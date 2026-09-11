import dataclasses
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generic, List, Literal, TypeVar

from ..context import SOURCE_REGISTRY

if TYPE_CHECKING:
    from ..config import CollectionConfig
    from ..context import Context
    from ..rerank import AbstractReranker
    from ..search import SearchHit

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

    reranker: "AbstractReranker | None"

    def __init__(
        self,
        collection_id: str,
        *,
        context: "Context",
        collection_config: "CollectionConfig",
        reranker: "AbstractReranker | None" = None,
    ):
        self.context = context
        self.config = collection_config
        self.id = collection_id
        self.reranker = reranker

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prefix = getattr(cls, "source_prefix", cls.__name__.lower())
        if prefix not in SOURCE_REGISTRY:
            SOURCE_REGISTRY[prefix] = cls

    async def get_documents(self, last_modified: datetime | None = None) -> AsyncGenerator[p, None]:
        ...

    def fetch_document(document_id: str) -> p:
        ...

    async def get_document_summary(self, document_id: str) -> str:
        return ""

    async def native_search(
        self, query: str, *, document_limit: int, chunk_limit: int
    ) -> list["SearchHit"]:
        """Return ranked document and embedding-chunk hits, documents first.

        Limits apply separately to each hit type. Chunk metadata uses the same
        retrieval convention as get_chunks/fetch_chunk, even before indexing.
        Scores are meaningful within each ranked list only.
        """
        return []

    def chunk_lengths(
        self, chunk_metadata: dict[str, Any]
    ) -> tuple[int | None, int | None]:
        semantic_length: int | None = None
        if "e" in chunk_metadata:
            # XXX refactoring needed: b/e is currently a FileSource-specific
            # semantic-span convention, not a DocumentSource abstraction.
            # A truly generic semantic_length likely requires fetching the
            # semantic chunk text (or storing/memoizing the length in schema).
            semantic_start = int(chunk_metadata.get("b", 0))
            semantic_end = int(chunk_metadata["e"])
            semantic_length = semantic_end - semantic_start

        embedding_length = (
            int(chunk_metadata["s"]) if "s" in chunk_metadata else None
        )

        if semantic_length is None:
            semantic_length = embedding_length

        return semantic_length, embedding_length

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
