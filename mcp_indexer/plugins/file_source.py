import os
import re
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Generic, TypeVar, Any
from urllib.parse import quote, unquote
from pydantic import BaseModel, Field
from mcp_indexer.plugins.base import DocumentSource, DocumentPointer, ChunkInfo, DocumentNotFoundError

class FileSourceConfig(BaseModel):
    """Base configuration for directory-based sources."""
    directory: Path
    include: List[str] = Field(default_factory=list, description="Glob patterns to include")
    exclude: List[str] = Field(default_factory=list, description="Glob patterns to exclude")
    # Text encoding of the source's files. b/e metadata and chunk decoding
    # are byte-based; everything else in the pipeline is char-based.
    encoding: str = "utf-8"

class FileSourcePointer(DocumentPointer):
    """
    A DocumentPointer for a file on disk.
    """
    def __init__(self, source: "FileSource", document_id: str, path: Path):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
        except FileNotFoundError as exc:
            source._check_root()
            raise DocumentNotFoundError(document_id) from exc
        super().__init__(source, document_id, last_modified=mtime)
        self.path = path

    async def get_metadata(self) -> dict[str, str]:
        return {
            "title": self.path.name,
            "last_modified": str(self.last_modified)
        }

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> str:
        """Return the text of the semantic chunk, decoded in the source's
        encoding. This is the bytes-to-text boundary for file sources."""
        b = chunk_metadata.get('b', 0)
        e = chunk_metadata.get('e')
        if e is None:
            raise ValueError("Chunk metadata must contain 'e' (end) byte offset")

        try:
            with open(self.path, 'rb') as f:
                f.seek(b)
                chunk_bytes = f.read(e - b)
        except FileNotFoundError as exc:
            self.source._check_root()
            raise DocumentNotFoundError(self.document_id) from exc
        return chunk_bytes.decode(self.source.source_config.encoding, errors='replace')

    async def get_chunks(self, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        try:
            # read_bytes + decode, not read_text: text-mode newline
            # translation would make the decoded string (and the byte
            # offsets computed from it) disagree with the file.
            text = self.path.read_bytes().decode(self.source.source_config.encoding)
        except FileNotFoundError as exc:
            self.source._check_root()
            raise DocumentNotFoundError(self.document_id) from exc
        async for chunk_info in self.source.split_text({}, text, min_size, max_size):
            yield chunk_info

P = TypeVar("P", bound=DocumentPointer)

class FileSource(DocumentSource[P]):
    """
    An abstract DocumentSource for files on disk.
    """
    safe_document_path_chars = "/-_.!@+[]()"
    semantic_boundary_regexps: tuple[str, ...] = ()
    embedding_boundary_regexps: tuple[str, ...] = (r"\n\s*\n",)

    def __init__(self, collection_id: str, *, context, collection_config, reranker=None):
        super().__init__(
            collection_id,
            context=context,
            collection_config=collection_config,
            reranker=reranker,
        )
        self.source_config = collection_config.resolve_source_config(FileSourceConfig)

    def _is_included(self, path: Path) -> bool:
        if self.source_config.include:
            if not any(path.match(pattern) for pattern in self.source_config.include):
                return False
        if any(path.match(pattern) for pattern in self.source_config.exclude):
            return False
        return True

    def encode_document_path(self, relative_path: str) -> str:
        return quote(relative_path, safe=self.safe_document_path_chars)

    def decode_document_path(self, encoded_path: str) -> str:
        return unquote(encoded_path)

    def _check_root(self) -> None:
        if not self.source_config.directory.is_dir():
            raise OSError(f"Source directory unavailable: {self.source_config.directory}")

    def _document_path(self, document_id: str) -> Path:
        relative = Path(self.decode_document_path(document_id))
        root = self.source_config.directory.resolve()
        path = root / relative
        if relative.is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Document path outside source: {document_id}")
        if not self._is_included(relative):
            self._check_root()
            raise DocumentNotFoundError(document_id, "Document excluded from collection")
        return path

    async def get_documents(self, last_modified: Optional[datetime] = None) -> AsyncGenerator[P, None]:
        root = self.source_config.directory
        self._check_root()
        for path in root.rglob("*"):
            if path.is_file() and self._is_included(path):
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if last_modified is None or mtime > last_modified:
                    relative_path = path.relative_to(root).as_posix()
                    encoded_path = self.encode_document_path(relative_path)
                    full_doc_id = encoded_path
                    yield self.fetch_document(full_doc_id)

    async def split_text(self, metadata: dict[str, Any], text: str, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        """Split a document into semantic chunks of candidate embedding
        chunks, as (metadata, candidate_text) pairs.

        Operates on text: all boundary matching and candidate slicing are
        char-based, so a cut never lands mid-character. The metadata is
        enriched with ``b``/``e`` byte offsets into the encoded file
        (computed by encoding each piece in the source's encoding), which
        are the source's own retrieval coordinates, not the app's.
        """
        if not text:
            return
        semantic_chunks = []
        start = 0
        matches = []
        for pattern in self.semantic_boundary_regexps:
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

        encoding = self.source_config.encoding
        current_byte_start = 0
        for s_chunk in semantic_chunks:
            # Byte offsets for the source's own retrieval (b/e): encoded
            # piece length, cumulative from the start of the file.
            byte_len = len(s_chunk.encode(encoding))
            # Candidate embedding chunks at the embedding boundaries, with
            # no sizing: min_size/max_size enforcement is the
            # create_embedding_chunks step's job, which gets the list of
            # candidates as a hint.
            e_chunks = []
            e_start = 0
            e_matches = []
            for pattern in self.embedding_boundary_regexps:
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
                chunk_metadata = {**metadata, 'b': current_byte_start,
                                  'e': current_byte_start + byte_len}
                yield chunk_metadata, e_chunks
            current_byte_start += byte_len

    def fetch_document(self, document_id: str) -> P:
        absolute_path = self._document_path(document_id)
        return FileSourcePointer(self, document_id, absolute_path) # type: ignore
