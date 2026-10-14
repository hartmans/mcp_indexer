import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from typing import Any, AsyncGenerator, List, Optional, TypeVar
from urllib.parse import quote, unquote
from pydantic import BaseModel, Field
from mcp_indexer.plugins.base import (
    ContentBridge,
    DocumentHandle,
    DocumentNotFoundError,
    DocumentSource,
)
from mcp_indexer.plugins.plugin import DocumentPlugin, HANDLER_REGISTRY


class FileSourceConfig(BaseModel):
    """Base configuration for directory-based sources.

    Carries the source-level text settings: the encoding of the source's
    files and ``summary_length``, the size cap below which a document
    summary may be generated. b/e metadata and chunk decoding are
    byte-based; everything else in the pipeline is char-based.
    """
    directory: Path
    include: List[str] = Field(default_factory=list, description="Glob patterns to include")
    exclude: List[str] = Field(default_factory=list, description="Glob patterns to exclude")
    encoding: str = "utf-8"
    summary_length: int = 65536


class FileBridge(ContentBridge):
    """Content access for a real filesystem object.

    The bridge opens and closes files inside each operation and never owns
    the file: ``path`` yields the existing path without copying or deleting
    it. Filesystem I/O is offloaded to a thread so it does not block the
    event loop. If the file has disappeared, the underlying source is asked
    to confirm the root is still available (raising ``OSError`` if not) and
    a :class:`DocumentNotFoundError` is raised.
    """

    def __init__(self, source, path: Path):
        self._source = source
        self._path = path

    async def read_bytes(self) -> bytes:
        try:
            return await asyncio.to_thread(self._path.read_bytes)
        except FileNotFoundError as exc:
            self._source._check_root()
            raise DocumentNotFoundError(str(self._path)) from exc

    async def read_byte_range(self, start: int, end: int) -> bytes:
        def _read() -> bytes:
            with open(self._path, "rb") as f:
                f.seek(start)
                return f.read(end - start)
        try:
            return await asyncio.to_thread(_read)
        except FileNotFoundError as exc:
            self._source._check_root()
            raise DocumentNotFoundError(str(self._path)) from exc

    @asynccontextmanager
    async def path(self):
        yield self._path


# ---------------------------------------------------------------------------
# Code-defined extension-to-policy discovery.
#
# The extension picks a policy; the policy's ``handler`` name picks the
# plugin class. These mappings are separate from the handler-name registry.
# Suffixes compare case-insensitively; unknown extensions and extensionless
# files fall back to :data:`TEXT_POLICY`.
# ---------------------------------------------------------------------------

# Markdown (``.md``, ``.markdown``): semantic boundaries are level-1 and
# level-2 headings -- ATX ``#``/``##`` and Setext underlines. Deeper ATX
# headings are not semantic boundaries.
MARKDOWN_POLICY: dict[str, Any] = {
    "handler": "text",
    "semantic_boundary_regexps": (
        r"(?m)^#{1,2}[ \t]+\S",
        r"(?m)^={3,}[ \t]*\r?$",
        r"(?m)^-{3,}[ \t]*\r?$",
    ),
    "embedding_boundary_regexps": (r"\n\s*\n",),
}

# HTML (``.html``, ``.htm``): semantic boundaries are h1 and h2 elements,
# not h3 through h6. The markup is preserved; no conversion is introduced.
HTML_POLICY: dict[str, Any] = {
    "handler": "text",
    "semantic_boundary_regexps": (
        r"(?im)^<h[12][\s>].*$",
    ),
    "embedding_boundary_regexps": (
        r"(?im)</(?:p|div|section|article|li|h[1-6])>\s*",
    ),
}

# RST (``.rst``): ``=`` and ``-`` title adornments are semantic-boundary
# hints, adopting the documented section/subsection convention (Sphinx RST
# primer). RST assigns hierarchy by order of adornment styles, not fixed
# characters; this does not claim to infer actual heading levels.
RST_POLICY: dict[str, Any] = {
    "handler": "text",
    "semantic_boundary_regexps": (
        r"(?m)^={3,}[ \t]*\r?$",
        r"(?m)^-{3,}[ \t]*\r?$",
    ),
    "embedding_boundary_regexps": (r"\n\s*\n",),
}

# Default text fallback: a line of common separator punctuation (three or
# more of ``-``/``=``/``*``/``_``/``+``) is a semantic boundary; paragraph
# breaks are the embedding boundaries.
TEXT_POLICY: dict[str, Any] = {
    "handler": "text",
    "semantic_boundary_regexps": (
        r"(?m)^[-=*+_]{3,}[ \t]*\r?$",
    ),
    "embedding_boundary_regexps": (r"\n\s*\n",),
}

# Docling-handled formats: PDF and selected office formats. Interpretation
# is fixed by the plugin (conversion, Markdown delegation, figure
# subdocuments); the policy carries only the handler name.
DOCLING_POLICY: dict[str, Any] = {
    "handler": "docling",
}

_EXTENSION_POLICIES: dict[str, dict[str, Any]] = {
    ".md": MARKDOWN_POLICY,
    ".markdown": MARKDOWN_POLICY,
    ".html": HTML_POLICY,
    ".htm": HTML_POLICY,
    ".rst": RST_POLICY,
    ".pdf": DOCLING_POLICY,
    ".doc": DOCLING_POLICY,
    ".docx": DOCLING_POLICY,
    ".odt": DOCLING_POLICY,
    ".ods": DOCLING_POLICY,
    ".ppt": DOCLING_POLICY,
    ".pptx": DOCLING_POLICY,
    ".rtf": DOCLING_POLICY,
}


def extension_policy(path: Path) -> dict[str, Any]:
    """Select the interpretation policy for a file by its extension.

    Code-defined mapping: unknown extensions and extensionless files fall
    back to :data:`TEXT_POLICY`. The suffix is compared case-insensitively,
    so ``NOTES.MD`` selects the Markdown policy.
    """
    return _EXTENSION_POLICIES.get(path.suffix.lower(), TEXT_POLICY)


P = TypeVar("P", bound=DocumentPlugin)


class FileSource(DocumentSource[P]):
    """An DocumentSource for files on disk.

    Discovery maps each file's extension to an interpretation policy:
    the source's extension map starts from the code-defined mapping and
    overrides it with the collection's configured plugin entries. The
    policy's ``handler`` name picks the plugin class, resolved through
    the handler-name registry, so enumeration and direct fetch select
    consistently. The document's encoding and summary length come from
    the source's config, not per-document arguments.
    """
    source_prefix = "file"
    safe_document_path_chars = "/-_.!@+[]()"

    def __init__(self, collection_id: str, *, context, collection_config, reranker=None):
        super().__init__(
            collection_id,
            context=context,
            collection_config=collection_config,
            reranker=reranker,
        )
        self.source_config = collection_config.resolve_source_config(FileSourceConfig)
        # Code-defined map first; the collection's resolved
        # extension-to-policy map (merged defaults and collection
        # entries) overrides per extension.
        self._extension_map = {**_EXTENSION_POLICIES, **collection_config.plugin}

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

    def policy_for_path(self, path: Path) -> dict[str, Any]:
        """Choose the interpretation policy for a file.

        The source's extension map starts from the code-defined
        extension-to-policy mapping, overridden by the collection's
        configured plugin entries; unknown extensions and extensionless
        files fall back to :data:`TEXT_POLICY`. A source that
        interprets its files specially may override this (or
        :meth:`make_document` directly).
        """
        return self._extension_map.get(path.suffix.lower(), TEXT_POLICY)

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
                    yield await self.fetch_document(full_doc_id)

    def _fetch_document(self, document_id: str) -> P:
        path = self._document_path(document_id)
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
        except FileNotFoundError as exc:
            self._check_root()
            raise DocumentNotFoundError(document_id) from exc
        return self.make_document(document_id, mtime, path)

    async def _fetch_root(self, document_id: str) -> P:
        """Acquire a root (non-subdocument) document by its local ID.

        Resolve with the current mtime; raise DocumentNotFoundError if absent.
        The synchronous path work runs in a thread so the event loop is not
        blocked by filesystem stats.
        """
        return await asyncio.to_thread(self._fetch_document, document_id)

    def make_document(self, document_id: str, mtime: datetime, path: Path) -> P:
        """Construct the selected plugin for a resolved path.

        The policy comes from :meth:`policy_for_path`; its ``handler`` name
        is resolved through the handler-name registry, and the source
        supplies identity, the file bridge, the mtime, and the initial
        metadata. Enumeration (via ``get_documents``) and direct fetch (via
        ``fetch_document``) both go through this hook, so they select
        consistently. A source that knows its document type may override
        this and construct it directly with a specialized policy.
        """
        policy = self.policy_for_path(path)
        handler = policy.get("handler")
        cls = HANDLER_REGISTRY.get(handler) if handler is not None else None
        if cls is None:
            raise ValueError(
                f"No document plugin registered for handler {handler!r} "
                f"(file {path.name!r})"
            )
        return cls(
            document_id,
            source=self,
            bridge=FileBridge(self, path),
            policy=policy,
            last_modified=mtime,
            metadata={"title": path.name},
        )
