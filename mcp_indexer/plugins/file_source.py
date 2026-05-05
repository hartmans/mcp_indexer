import os
import re
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Generic, TypeVar, Any
from urllib.parse import quote, unquote
from pydantic import BaseModel, Field
from mcp_indexer.plugins.base import DocumentSource, DocumentPointer, ChunkInfo

class FileSourceConfig(BaseModel):
    """Base configuration for directory-based sources."""
    directory: Path
    include: List[str] = Field(default_factory=list, description="Glob patterns to include")
    exclude: List[str] = Field(default_factory=list, description="Glob patterns to exclude")

class FileSourcePointer(DocumentPointer):
    """
    A DocumentPointer for a file on disk.
    """
    def __init__(self, source: "FileSource", document_id: str, path: Path):
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        super().__init__(source, document_id, last_modified=mtime)
        self.path = path

    async def get_metadata(self) -> dict[str, str]:
        return {
            "title": self.path.name,
            "last_modified": str(self.last_modified)
        }

    async def fetch_semantic_chunk(self, chunk_metadata: dict[str, Any]) -> list[str]:
        b = chunk_metadata.get('b', 0)
        e = chunk_metadata.get('e')
        if e is None:
            raise ValueError("Chunk metadata must contain 'e' (end) byte offset")

        with open(self.path, 'rb') as f:
            f.seek(b)
            chunk_bytes = f.read(e - b)
        return [chunk_bytes.decode('utf-8', errors='replace')]

    async def get_chunks(self, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        with open(self.path, 'rb') as f:
            text_bytes = f.read()
        async for chunk_info in self.source.split_text({}, text_bytes, min_size, max_size):
            yield chunk_info

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> list[str]:
        return await self.fetch_semantic_chunk(chunk_metadata)

P = TypeVar("P", bound=DocumentPointer)

class FileSource(DocumentSource[P]):
    """
    An abstract DocumentSource for files on disk.
    """
    safe_document_path_chars = "/-_.!@+[]()"
    semantic_boundary_regexps: tuple[bytes, ...] = ()
    embedding_boundary_regexps: tuple[bytes, ...] = (rb"\n\s*\n",)

    def __init__(self, collection_id: str, *, context, collection_config):
        super().__init__(collection_id, context=context, collection_config=collection_config)
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

    async def get_documents(self, last_modified: Optional[datetime] = None) -> AsyncGenerator[P, None]:
        root = self.source_config.directory
        for path in root.rglob("*"):
            if path.is_file() and self._is_included(path):
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if last_modified is None or mtime > last_modified:
                    relative_path = path.relative_to(root).as_posix()
                    encoded_path = self.encode_document_path(relative_path)
                    full_doc_id = encoded_path
                    yield self.fetch_document(full_doc_id)

    async def split_text(self, metadata: dict[str, Any], text_bytes: bytes, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        if not bytes(text_bytes):
            return
        semantic_chunks = []
        start = 0
        matches = []
        for pattern in self.semantic_boundary_regexps:
            for m in re.finditer(pattern, text_bytes):
                matches.append(m.start())
        matches.sort()
        for pos in matches:
            if pos <= start:
                continue
            chunk = text_bytes[start:pos]
            if len(chunk) >= min_size:
                semantic_chunks.append(chunk)
                start = pos
        final_chunk = text_bytes[start:]
        if final_chunk:
            if semantic_chunks and len(final_chunk) < min_size:
                semantic_chunks[-1] += final_chunk
            else:
                semantic_chunks.append(final_chunk)

        current_semantic_start = 0
        for s_chunk in semantic_chunks:
            current_semantic_end = current_semantic_start + len(s_chunk)
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
                chunk = s_chunk[e_start:pos]
                if len(chunk) >= min_size:
                    e_chunks.append(chunk)
                    e_start = pos
            remnant = s_chunk[e_start:]
            if remnant:
                if e_chunks and len(remnant) < min_size:
                    e_chunks[-1] += remnant
                else:
                    e_chunks.append(remnant)
            if e_chunks:
                chunk_metadata = {**metadata, 'b': current_semantic_start, 'e': current_semantic_end}
                yield chunk_metadata, [c.decode('utf-8', errors='replace') for c in e_chunks]
            current_semantic_start = current_semantic_end

    def fetch_document(self, document_id: str) -> P:
        relative_path_str = self.decode_document_path(document_id)
        absolute_path = self.source_config.directory / relative_path_str
        return FileSourcePointer(self, document_id, absolute_path) # type: ignore
