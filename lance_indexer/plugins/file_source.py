import os
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from lance_indexer.plugins.base import DocumentSource, DocumentPointer, Document, DocumentChunk

class FileSourceConfig(BaseModel):
    """Configuration for directory-based sources."""
    directory: Path
    include: List[str] = Field(default_factory=list, description="Glob patterns to include")
    exclude: List[str] = Field(default_factory=list, description="Glob patterns to exclude")

P = TypeVar("P", bound=DocumentPointer)

class FileSource(DocumentSource[P]):
    """
    An abstract DocumentSource for files on disk.
    Handles directory traversal, filtering, and change detection.
    """

    def __init__(self, context, collection_config):
        super().__init__(context, collection_config)
        # Validate the source_blob into FileSourceConfig
        self.file_config = collection_config.resolve_source_config(FileSourceConfig)

    def _is_included(self, path: Path) -> bool:
        """Check if a path matches include patterns and does not match exclude patterns."""
        # If include patterns are defined, path must match at least one
        if self.file_config.include:
            if not any(path.match(pattern) for pattern in self.file_config.include):
                return False
        
        # Path must not match any exclude patterns
        if any(path.match(pattern) for pattern in self.file_config.exclude):
            return False
            
        return True

    async def get_documents(self, last_modified: Optional[datetime] = None) -> AsyncGenerator[P, None]:
        """
        Walks the directory and yields documents modified since last_modified.
        """
        root = self.file_config.directory
        for path in root.rglob("*"):
            if path.is_file() and self._is_included(path):
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if last_modified is None or mtime > last_modified:
                    yield self.create_document_pointer(path)

    def create_document_pointer(self, path: Path) -> P:
        """
        Factory method to create a DocumentPointer for a file.
        Must be implemented by the child plugin.
        """
        raise NotImplementedError("FileSource plugins must implement create_document_pointer")

    async def index_document(self, pointer: P) -> None:
        """
        High-level workflow to index a single file:
        1. Get metadata.
        2. Extract chunks.
        3. Save to LanceDB.
        """
        # This would be called by the Indexer
        # It uses the abstract methods implemented by the child plugin
        doc_metadata = await self.extract_metadata(pointer)
        async for chunk in self.extract_chunks(pointer):
            # Save logic here (or delegated to indexer)
            pass

    # --- Abstract Methods for Downstream Plugins ---

    async def extract_metadata(self, pointer: P) -> Document:
        """Extract the Document-level metadata (title, keywords, etc.) from the file."""
        raise NotImplementedError

    async def extract_chunks(self, pointer: P) -> AsyncGenerator[DocumentChunk, None]:
        """Read the file and yield chunks with embeddings and summaries."""
        raise NotImplementedError

    def fetch_document(self, document_id: str) -> P:
        """Implement how to recover a pointer from a document_id (e.g. mapping ID back to path)."""
        raise NotImplementedError

    def fetch_chunk(self, chunk_id: str) -> str:
        """Implement how to retrieve the text of a specific chunk from the file."""
        raise NotImplementedError

    def document_id(self, chunk_id: str) -> str:
        """Recover the document_id from a chunk_id."""
        raise NotImplementedError
