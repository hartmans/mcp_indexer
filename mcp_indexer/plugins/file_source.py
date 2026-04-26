import os
import re
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Generic, TypeVar, Any
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
        """Returns basic file metadata."""
        return {
            "title": self.path.name,
            "last_modified": str(self.last_modified)
        }

    async def fetch_semantic_chunk(self, chunk_metadata: dict[str, Any]) -> list[str]:
        """
        Reads a specific semantic chunk from the file using the 'b' and 'e' byte offsets.
        """
        b = chunk_metadata.get('b', 0)
        e = chunk_metadata.get('e')
        
        if e is None:
            raise ValueError("Chunk metadata must contain 'e' (end) byte offset")

        with open(self.path, 'rb') as f:
            f.seek(b)
            chunk_bytes = f.read(e - b)
        
        return [chunk_bytes.decode('utf-8', errors='replace')]

    async def get_chunks(self) -> AsyncGenerator[ChunkInfo, None]:
        """
        Reads the full file as bytes and uses the source's split_text helper.
        """
        with open(self.path, 'rb') as f:
            text_bytes = f.read()
        
        # Pass the byte stream and an empty dict for metadata
        async for chunk_info in self.source.split_text({}, text_bytes, 100, 1000):
            yield chunk_info

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> list[str]:
        """
        Retrieve the semantic chunk identified by the metadata.
        """
        return await self.fetch_semantic_chunk(chunk_metadata)

P = TypeVar("P", bound=DocumentPointer)

class FileSource(DocumentSource[P]):
    """
    An abstract DocumentSource for files on disk.
    Handles directory traversal, filtering, and change detection.
    """

    # Boundaries are now byte strings.
    semantic_boundary_regexps: tuple[bytes, ...] = ()
    embedding_boundary_regexps: tuple[bytes, ...] = (rb"\n\s*\n",)

    def __init__(self, collection_id: str, *, context, collection_config):
        super().__init__(collection_id, context=context, collection_config=collection_config)
        
        # Ensure the config is validated using the Pydantic model
        self.source_config = collection_config.resolve_source_config(FileSourceConfig)

    def _is_included(self, path: Path) -> bool:
        """Check if a path matches include patterns and does not match exclude patterns."""
        if self.source_config.include:
            if not any(path.match(pattern) for pattern in self.source_config.include):
                return False
        
        if any(path.match(pattern) for pattern in self.source_config.exclude):
            return False
            
        return True

    async def get_documents(self, last_modified: Optional[datetime] = None) -> AsyncGenerator[P, None]:
        """
        Walks the directory and yields documents modified since last_modified.
        """
        root = self.source_config.directory
        for path in root.rglob("*"):
            if path.is_file() and self._is_included(path):
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if last_modified is None or mtime > last_modified:
                    relative_path = str(path.relative_to(root))
                    full_doc_id = f"{self.id_prefix}{relative_path}"
                    yield self.fetch_document(full_doc_id)

    async def split_text(self, metadata: dict[str, Any], text_bytes: bytes, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        """
        Splits byte text into semantic chunks, and within those, identifies embedding boundaries.
        """
        if not text_bytes:
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
                # Metadata now contains byte offsets
                chunk_metadata = {**metadata, 'b': current_semantic_start, 'e': current_semantic_end}
                # Decode fragments to strings for the indexer
                yield chunk_metadata, [c.decode('utf-8', errors='replace') for c in e_chunks]
            
            current_semantic_start = current_semantic_end

    def fetch_document(self, document_id: str) -> P:
        """
        Recover a DocumentPointer from a document_id.
        """
        relative_path_str = self.strip_id_prefix(document_id)
        absolute_path = self.source_config.directory / relative_path_str
        return FileSourcePointer(self, document_id, absolute_path) # type: ignore
