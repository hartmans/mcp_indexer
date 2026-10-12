import re
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from mcp_indexer.plugins.base import (
    ChunkInfo,
    ContentBridge,
    DocumentHandle,
    DocumentSource,
)
from mcp_indexer.plugins.file_source import FileBridge, FileSource, FileSourceConfig


def split_text(
    text: str,
    encoding: str,
    semantic_boundary_regexps: tuple[str, ...],
    embedding_boundary_regexps: tuple[str, ...],
    min_size: int,
) -> list[tuple[dict[str, Any], list[str]]]:
    """Split text into semantic chunks of candidate embedding chunks.

    Operates on text: all boundary matching and candidate slicing are
    char-based, so a cut never lands mid-character. Each semantic chunk's
    metadata carries ``b``/``e`` byte offsets into the encoded content
    (computed by encoding each piece in ``encoding``), which are the
    source's own retrieval coordinates, not the app's.

    Only the semantic-split minimum is enforced here: semantic chunks are
    cut at the first boundary that yields a chunk of at least ``min_size``
    characters, and a trailing remnant below that is merged into the
    previous chunk. ``max_size`` is not enforced here; it is
    ``create_embedding_chunks``' job, which receives the candidate
    embedding chunks as a hint.
    """
    if not text:
        return []
    semantic_chunks = []
    start = 0
    matches = []
    for pattern in semantic_boundary_regexps:
        for m in re.finditer(pattern, text):
            matches.append(m.start())
    matches.sort()
    for pos in matches:
        if pos <= start:
            continue
        chunk = text[start:pos]
        if len(chunk) >= min_size:
            semantic_chunks.append(chunk)
            start = pos
    final_chunk = text[start:]
    if final_chunk:
        if semantic_chunks and len(final_chunk) < min_size:
            semantic_chunks[-1] += final_chunk
        else:
            semantic_chunks.append(final_chunk)

    results: list[tuple[dict[str, Any], list[str]]] = []
    current_byte_start = 0
    for s_chunk in semantic_chunks:
        # Byte offsets for the source's own retrieval (b/e): encoded
        # piece length, cumulative from the start of the content.
        byte_len = len(s_chunk.encode(encoding))
        # Candidate embedding chunks at the embedding boundaries, with
        # no sizing: min_size/max_size enforcement is the
        # create_embedding_chunks step's job, which gets the list of
        # candidates as a hint.
        e_chunks = []
        e_start = 0
        e_matches = []
        for pattern in embedding_boundary_regexps:
            for m in re.finditer(pattern, s_chunk):
                e_matches.append(m.end())
        e_matches.sort()
        for pos in e_matches:
            if pos <= e_start:
                continue
            e_chunks.append(s_chunk[e_start:pos])
            e_start = pos
        remnant = s_chunk[e_start:]
        if remnant:
            e_chunks.append(remnant)
        if e_chunks:
            chunk_metadata = {'b': current_byte_start,
                              'e': current_byte_start + byte_len}
            results.append((chunk_metadata, e_chunks))
        current_byte_start += byte_len
    return results


class TextDocument(DocumentHandle):
    """A document whose content is text.

    Text interpretation (semantic/embedding boundaries, encoding, summary
    limits) is supplied at construction so a document can be built from any
    content access, not only a :class:`FileSource`.
    """

    def __init__(
        self,
        source: DocumentSource,
        document_id: str,
        last_modified: datetime,
        bridge: ContentBridge,
        *,
        title: str,
        encoding: str = "utf-8",
        semantic_boundary_regexps: tuple[str, ...] = (),
        embedding_boundary_regexps: tuple[str, ...] = (r"\n\s*\n",),
        summary_length: int = 65536,
    ):
        super().__init__(source, document_id, last_modified)
        self.bridge = bridge
        self.title = title
        self.encoding = encoding
        self.semantic_boundary_regexps = semantic_boundary_regexps
        self.embedding_boundary_regexps = embedding_boundary_regexps
        self.summary_length = summary_length

    async def get_metadata(self) -> dict[str, str]:
        """Title from the source; last_modified from the resolved mtime."""
        return {
            "title": self.title,
            "last_modified": str(self.last_modified),
        }

    async def _text(self, errors: str | None = None) -> str:
        """Read the content bytes and decode them.

        ``errors`` is passed to ``decode``; ``None`` means strict.
        read_bytes + decode, not text-mode read: text-mode newline
        translation would make the decoded string (and the byte offsets
        computed from it) disagree with the underlying content.
        """
        data = await self.bridge.read_bytes()
        if errors is None:
            return data.decode(self.encoding)
        return data.decode(self.encoding, errors)

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> str:
        """Return the text of the semantic chunk, decoded in the document's
        encoding. This is the bytes-to-text boundary."""
        b = chunk_metadata.get("b", 0)
        e = chunk_metadata.get("e")
        if e is None:
            raise ValueError("Chunk metadata must contain 'e' (end) byte offset")
        data = await self.bridge.read_byte_range(b, e)
        return data.decode(self.encoding, errors="replace")

    async def get_chunks(self, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        text = await self._text()
        for chunk_info in split_text(
            text,
            self.encoding,
            self.semantic_boundary_regexps,
            self.embedding_boundary_regexps,
            min_size,
        ):
            yield chunk_info

    async def get_document_summary(self) -> str:
        """Summarize the document with the LLM if it is small enough."""
        text = await self._text(errors="replace")
        llm = self.source.context.llm
        if len(text) <= self.summary_length and llm is not None:
            prompt = [
                ("system", self.source.config.doc_summary_prompt),
                ("user", "Write no more than two paragraphs to summarize the "
                 "following document:\n\n" + text),
            ]
            summaries = await llm([prompt])
            return summaries[0]
        return ""


class TextFileSourceConfig(FileSourceConfig):
    """Configuration for the TextFileSource plugin."""
    summary_length: int = 65536


class TextFileSource(FileSource[TextDocument]):
    """
    A concrete FileSource for handling simple text files (Markdown, RST, TXT).
    """

    source_prefix = "text"

    # Semantic boundaries:
    # 1. RST style headers (underline of = or -)
    # 2. Markdown style headers (# Header)
    # 3. Large gaps of whitespace (though handled by embedding boundaries,
    #    we might want them as semantic breaks)
    semantic_boundary_regexps = (
        r"\n={3,}\n",           # RST Level 1
        r"\n-{3,}\n",           # RST Level 2
        r"\n\s*#{1,6}\s+.*",    # Markdown Headers
    )

    # Embedding boundaries:
    # 1. Double newlines (paragraphs)
    # 2. Horizontal rules
    embedding_boundary_regexps = (
        r"\n\s*\n",             # Paragraphs
        r"\n\s*-{3,}\s*\n",     # MD/RST Horizontal rules
    )

    def __init__(self, collection_id: str, *, context, collection_config, reranker=None):
        super().__init__(
            collection_id,
            context=context,
            collection_config=collection_config,
            reranker=reranker,
        )
        # Resolve the specific TextFileSourceConfig
        self.source_config = collection_config.resolve_source_config(TextFileSourceConfig)

    def make_document(self, document_id: str, mtime: datetime, path: Path) -> TextDocument:
        return TextDocument(
            source=self,
            document_id=document_id,
            last_modified=mtime,
            bridge=FileBridge(self, path),
            title=path.name,
            encoding=self.source_config.encoding,
            semantic_boundary_regexps=self.semantic_boundary_regexps,
            embedding_boundary_regexps=self.embedding_boundary_regexps,
            summary_length=self.source_config.summary_length,
        )
