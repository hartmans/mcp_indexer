import bisect
import dataclasses
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generic, List, Literal, TypeVar

from ..context import SOURCE_REGISTRY

if TYPE_CHECKING:
    from ..config import CollectionConfig
    from ..context import Context
    from ..rerank import AbstractReranker
    from ..search import NativeSearchBatch

# The metadata for a semantic chunk, and a proposed set of good splits into embedding chunks.
ChunkInfo = tuple[dict[str, Any], list[str]]


class DocumentNotFoundError(LookupError):
    """A collection-local document is absent from an available source."""

    def __init__(self, document_id: str, message: str = "Document not found"):
        self.document_id = document_id
        super().__init__(f"{message}: {document_id}")


@dataclasses.dataclass
class DocumentPointer:
    """A document to be indexed, returned from DocumentSource."""

    source: "DocumentSource"
    document_id: str
    last_modified: datetime

    async def get_metadata(self) -> dict[str, str]:
        ...

    async def get_chunks(self, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        """Yield (metadata, candidate_text) per semantic chunk.

        The candidates form a partition of the semantic chunk's text:
        contiguous, non-overlapping, and ``"".join(candidates)`` equals
        exactly what ``fetch_chunk`` returns at semantic scope. This
        invariant is untestable across the interface (candidates carry no
        offsets) and the chunker, stored offsets, and search dedup all
        rely on it: a source must not yield overlapping pieces.
        """
        ...

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> str:
        """Return the text of the semantic chunk identified by metadata.

        The bytes-to-text boundary is this method: the pointer decodes the
        stored span in the source's encoding and the app never sees bytes.
        """
        ...


def create_embedding_chunks(
    text_list: list[str],
    min_size: int,
    max_size: int,
) -> List[str]:
    """
    Sizes a single semantic chunk's candidate embedding chunks.

    ``text_list`` is the list of candidate embedding chunks proposed by the
    source (at its embedding boundaries). The candidates are a hint: each
    output chunk is cut at the first candidate boundary that yields a chunk
    of at least ``min_size`` (absorbing any smaller leading fragments), so
    with a wide min_size/max_size gap the cut lands near min_size and
    ``max_size`` binds only when no usable boundary appears. Where none
    does, a hard cut is made at ``max_size``; after such a slice the next
    cut resumes at the next candidate boundary, so chunks re-align with the
    source's boundaries as soon as the remainder allows. ``max_size``
    is a constraint (no output chunk exceeds it); ``min_size`` is a hint
    (undersized chunks are emitted only when a boundary cut or tail would
    otherwise violate ``max_size``).

    The output chunks partition the input text: they are contiguous,
    non-overlapping, and rejoin to the input. Offsets within the semantic
    text are the caller's bookkeeping (the indexer stores them in
    ``embedding_offset``); this function reports sizes only.
    """
    text = "".join(text_list)
    text_len = len(text)

    # Candidate boundary positions: cumulative end of each proposed chunk.
    boundaries = []
    pos = 0
    for piece in text_list:
        pos += len(piece)
        boundaries.append(pos)

    results: List[str] = []
    offset = 0
    while offset < text_len:
        # First candidate boundary at which the chunk reaches min_size.
        idx = bisect.bisect_left(boundaries, offset + min_size)
        b = boundaries[idx] if idx < len(boundaries) else None
        if b is None or b - offset > max_size:
            # No boundary in [offset + min_size, offset + max_size]:
            # hard cut at the cap.
            cut = min(offset + max_size, text_len)
        else:
            cut = b
        results.append(text[offset:cut])
        offset = cut

    # Tail: merge an undersized final chunk into the previous one if it
    # fits without exceeding max_size; otherwise emit it as-is.
    if len(results) >= 2 and len(results[-1]) < min_size:
        if len(results[-2]) + len(results[-1]) <= max_size:
            results[-1] = results[-2] + results[-1]
            results.pop(-2)

    return results


p = TypeVar("p", bound=DocumentPointer)


class DocumentSource(Generic[p]):
    """An abstract plugin representing a source of documents."""

    reranker: "AbstractReranker | None"
    default_indexing_mode: Literal["full", "indexed"] = "full"
    separate_rerank: bool = False

    @property
    def indexing_mode(self) -> Literal["full", "indexed"]:
        return self.config.indexing_mode or self.default_indexing_mode

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
        from ..search_state import SearchSessionStore
        self._search_sessions = SearchSessionStore()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prefix = getattr(cls, "source_prefix", cls.__name__.lower())
        if prefix not in SOURCE_REGISTRY:
            SOURCE_REGISTRY[prefix] = cls

    async def get_documents(self, last_modified: datetime | None = None) -> AsyncGenerator[p, None]:
        ...

    def fetch_document(self, document_id: str) -> p:
        """Resolve a local ID with current mtime; raise DocumentNotFoundError if absent."""
        ...

    async def get_document_summary(self, document_id: str) -> str:
        return ""

    async def native_search(
        self, query: str, *, document_limit: int, chunk_limit: int,
        document_offset: int = 0, chunk_offset: int = 0,
    ) -> "NativeSearchBatch | None":
        """Return a bounded batch of ranked hits, or None for exhaustion.

        Limits apply separately; zero disables a hit type. Offsets are
        source-defined nonnegative positions, initially zero, passed back from
        the previous batch unchanged. Every batch, including one with no hits,
        advances at least one enabled position and never decreases positions.
        Return final hits before returning None on a subsequent call. Raise on
        failure; empty batches do not mean exhaustion.

        Unchanged inputs/corpus require deterministic traversal and stable ties.
        Return documents first, then chunks. Search orchestration suppresses
        duplicates and already returned documents. Chunk metadata follows the
        get_chunks/fetch_chunk convention; equivalent spans must use consistent
        metadata across discovery mechanisms. Scores are meaningful within each
        ranked list only.
        """
        return None

    async def fetch_chunk(
        self, document_id: str, chunk_metadata: dict[str, Any],
        scope: Literal["semantic", "embedding"],
        offset: int | None = None, length: int | None = None,
    ) -> str:
        """Fetch one chunk's text.

        ``scope="semantic"`` returns the whole semantic chunk; the
        chunk-level ``offset``/``length`` are ignored (callers may still
        supply them, e.g. copied from a database row, for uniformity).
        ``scope="embedding"`` returns the slice
        ``semantic_text[offset:offset + length]``; both are required, in
        characters, and an embedding fetch without them is an error.
        """
        pointer = self.fetch_document(document_id)
        full_text = await pointer.fetch_chunk(chunk_metadata)

        if scope == "semantic":
            return full_text

        if offset is None or length is None:
            raise ValueError(
                "Embedding-scope fetch_chunk requires offset and length"
            )
        return full_text[offset:offset + length]

    def document_id(self, chunk_id: str) -> str:
        return chunk_id.rsplit("?", 1)[0]
