import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator, List, Optional, TypeVar
from urllib.parse import quote, unquote
from pydantic import BaseModel, Field
from mcp_indexer.plugins.base import (
    ContentBridge,
    DocumentHandle,
    DocumentNotFoundError,
    DocumentSource,
)


class FileSourceConfig(BaseModel):
    """Base configuration for directory-based sources."""
    directory: Path
    include: List[str] = Field(default_factory=list, description="Glob patterns to include")
    exclude: List[str] = Field(default_factory=list, description="Glob patterns to exclude")
    # Text encoding of the source's files. b/e metadata and chunk decoding
    # are byte-based; everything else in the pipeline is char-based.
    encoding: str = "utf-8"


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


P = TypeVar("P", bound=DocumentHandle)


class FileSource(DocumentSource[P]):
    """
    An abstract DocumentSource for files on disk.
    """
    safe_document_path_chars = "/-_.!@+[]()"
    # Default chunking boundaries for the plain-text document a FileSource
    # produces by default. Concrete sources override these.
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
                    yield await self.fetch_document(full_doc_id)

    def _fetch_document(self, document_id: str) -> P:
        path = self._document_path(document_id)
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
        except FileNotFoundError as exc:
            self._check_root()
            raise DocumentNotFoundError(document_id) from exc
        return self.make_document(document_id, mtime, path)

    async def fetch_document(self, document_id: str) -> P:
        """Resolve a local ID with its current mtime; raise
        DocumentNotFoundError if absent."""
        return await asyncio.to_thread(self._fetch_document, document_id)

    def make_document(self, document_id: str, mtime: datetime, path: Path) -> P:
        """Construct this source's document handle for a resolved path.

        ``_fetch_document`` and ``get_documents`` both go through this hook
        so subclasses can substitute their own document class without
        duplicating path/mtime resolution.
        """
        ...
