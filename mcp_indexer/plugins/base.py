import dataclasses
import json
from datetime import datetime
from types import MappingProxyType
from typing import List, AsyncGenerator, Generic, TypeVar, Any, AsyncIterator, Literal, Dict, Type, Mapping

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..context import Context
from ..config import CollectionConfig
from ..llm import VECTOR_DIMENSIONS
from ..context import SOURCE_REGISTRY
from ..models import Base, DocumentChunk, Document, ChunkSummary, serialize_metadata, deserialize_metadata

ChunkInfo = tuple[dict[str, Any], list[str]]


@dataclasses.dataclass
class DocumentPointer:
    """A document to be indexed, returned from DocumentSource."""

    source: "DocumentSource"
    document_id: str
    last_modified: datetime

    async def get_metadata(self) -> dict[str, str]:
        ...

    async def get_chunks(self, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        ...

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> list[str]:
        ...


def create_embedding_chunks(
    metadata: dict,
    text_list: list[str],
    min_size: int,
    max_size: int,
) -> List[tuple[dict, str]]:
    """
    Processes a single semantic chunk into embedding-sized chunks.
    Returns a list of (metadata, text) tuples.
    """
    text = "".join(text_list)
    text_len = len(text)

    results = []
    offset = 0

    while offset < text_len:
        end = min(offset + max_size, text_len)
        if offset > 0 and (end - offset) < min_size:
            if results:
                last_meta, last_text = results[-1]
                results[-1] = (last_meta, last_text + text[offset:end])
                offset = end
                continue

        chunk_text = text[offset:end]
        chunk_meta = {**metadata, "o": offset, "s": len(chunk_text)}
        results.append((chunk_meta, chunk_text))
        offset = end

    return results


p = TypeVar("p", bound=DocumentPointer)


class DocumentSource(Generic[p]):
    """An abstract plugin representing a source of documents."""

    def __init__(
        self, collection_id: str, *, context: Context, collection_config: CollectionConfig
    ):
        self.context = context
        self.config = collection_config
        self.id = collection_id

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prefix = getattr(cls, "source_prefix", cls.__name__.lower())
        if prefix not in SOURCE_REGISTRY:
            SOURCE_REGISTRY[prefix] = cls

    async def get_documents(self, last_modified: datetime | None = None) -> AsyncGenerator[p, None]:
        ...

    async def fetch_document(document_id: str) -> p:
        ...

    async def get_document_summary(self, document_id: str) -> str:
        return ""

    async def fetch_chunk(
        self, document_id: str, chunk_metadata: dict[str, Any], scope: Literal["semantic", "embedding"]
    ) -> str:
        pointer = self.fetch_document(document_id)
        semantic_chunk_list = await pointer.fetch_chunk(chunk_metadata)
        full_text = "".join(semantic_chunk_list)

        if scope == "semantic":
            return full_text

        offset = chunk_metadata.get("o", 0)
        span = chunk_metadata.get("s", len(full_text))
        return full_text[offset : offset + span]

    def document_id(self, chunk_id: str) -> str:
        return chunk_id.rsplit("?", 1)[0]