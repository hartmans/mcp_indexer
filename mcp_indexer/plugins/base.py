import asyncio
import bisect
import dataclasses
import time
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generic, List, Literal, TypeVar, cast

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


class ContentBridge:
    """How a :class:`DocumentHandle` reaches a document's underlying bytes.

    A bridge is a durable description of *how* to acquire content, not an
    already-open resource. Concrete bridges may read from the filesystem,
    hold decoded bytes, or produce the bytes (for example from a conversion
    command). All access is async, and ``path`` exposes a filesystem path
    that is only valid for the duration of its context.
    """

    async def read_bytes(self) -> bytes:
        """Return the full content of the document."""
        ...

    async def read_byte_range(self, start: int, end: int) -> bytes:
        """Return the bytes in the half-open range ``[start, end)``."""
        ...

    def path(self) -> AbstractAsyncContextManager[Path]:
        """Return an async context manager yielding a path to the content.

        The yielded path is valid only for the duration of the context.
        Concrete bridges implement this by decorating an async generator
        with ``@asynccontextmanager``.
        """
        ...


@dataclasses.dataclass
class DocumentHandle:
    """A bound document returned from a :class:`DocumentSource`.

    A handle is a durable description of access: it may be retained and used
    independently of the call that produced it.
    """

    source: "DocumentSource"
    document_id: str
    last_modified: datetime
    # Whether the indexer persists each embedding chunk's exact text in
    # ``DocumentChunk.text``. Output retention, independent of the bridge's
    # input capabilities.
    save_text: bool = False

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

        The bytes-to-text boundary is this method: the handle decodes the
        stored span in the source's encoding and the app never sees bytes.
        """
        ...

    async def get_sub_documents(self) -> AsyncGenerator["DocumentHandle", None]:
        """Yield this document's subdocuments; leaf documents yield nothing."""
        return
        yield

    async def fetch_sub_document(self, document_id: str) -> "DocumentHandle":
        """Resolve an immediate child by its full collection-local ID.

        ``document_id`` is the child's complete collection-local ID; with the
        final ``:`` component removed it is this handle's own
        ``document_id``. The returned child handle retains that full ID and
        uses it to identify its own descendants. The default reports the
        child as absent; container documents override it.
        """
        raise DocumentNotFoundError(
            document_id, f"has no subdocument {document_id.split(':')[-1]}"
        )

    async def get_document_summary(self) -> str:
        ...


class HandleCache:
    """Process-local cache of :class:`DocumentHandle` objects.

    Entries are keyed by full collection-local document ID and carry a
    monotonic last-used timestamp, updated on insertion and on a cache hit.
    Idle entries are evicted by a single recurring expiration callback
    scheduled on the running event loop: at most one callback is scheduled
    per cache, and it reschedules itself only while entries remain. Eviction
    releases cache references without invalidating handles still held by
    running work. This is idle expiration for memory management, not
    source-freshness validation.
    """

    def __init__(self, *, interval: float = 300.0):
        self._interval = interval
        self._entries: dict[str, tuple[Any, float]] = {}
        self._scheduled = False

    def get(self, document_id: str) -> Any:
        """Return the cached handle for ``document_id`` (refreshing its
        last-used timestamp) or None if absent."""
        entry = self._entries.get(document_id)
        if entry is None:
            return None
        handle, _ = entry
        self._entries[document_id] = (handle, time.monotonic())
        return handle

    def cache_document(self, handle: Any) -> None:
        """Insert or refresh a handle, scheduling the idle-expiration
        callback if one is not already scheduled."""
        self._entries[handle.document_id] = (handle, time.monotonic())
        self._schedule()

    def _schedule(self) -> None:
        if self._scheduled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (synchronous context); the cache still works,
            # it just performs no idle eviction.
            return
        self._scheduled = True
        loop.call_later(self._interval, self._expire)

    def _expire(self) -> None:
        self._scheduled = False
        now = time.monotonic()
        stale = [
            key for key, (_, last_used) in self._entries.items()
            if now - last_used >= self._interval
        ]
        for key in stale:
            del self._entries[key]
        if self._entries:
            self._schedule()

    def __len__(self) -> int:
        return len(self._entries)


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


p = TypeVar("p", bound=DocumentHandle)


class DocumentSource(Generic[p]):
    """An abstract plugin representing a source of documents."""

    reranker: "AbstractReranker | None"
    default_indexing_mode: Literal["full", "indexed"] = "full"
    separate_rerank: bool = False
    # Idle-expiration interval (seconds) for the process-local handle cache.
    # Overridable by tests.
    handle_cache_interval: float = 300.0

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
        self._handle_cache = HandleCache(interval=self.handle_cache_interval)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prefix = getattr(cls, "source_prefix", cls.__name__.lower())
        if prefix not in SOURCE_REGISTRY:
            SOURCE_REGISTRY[prefix] = cls

    async def get_documents(self, last_modified: datetime | None = None) -> AsyncGenerator[p, None]:
        ...

    async def fetch_document(self, document_id: str) -> p:
        """Resolve a full collection-local ID, root or nested subdocument.

        A handle already in the process-local cache is returned directly.
        Otherwise a root ID (no hierarchy separator) is acquired via
        :meth:`_fetch_root`; an ID containing a ``:`` hierarchy separator is
        split at the rightmost separator and resolved by acquiring the parent
        (recursively) and delegating to the parent's handle-level
        :meth:`DocumentHandle.fetch_sub_document`. The resolved handle is
        cached before being returned. It is durable and may be retained
        independently of this call.
        """
        cached = self._handle_cache.get(document_id)
        if cached is not None:
            return cached
        parent_id, separator, _component = document_id.rpartition(":")
        if not separator:
            handle = await self._fetch_root(document_id)
        else:
            parent = await self.fetch_document(parent_id)
            # A child resolved through this source is a handle of the same
            # source; fetch_sub_document is typed against the base class.
            handle = cast(p, await parent.fetch_sub_document(document_id))
        self._handle_cache.cache_document(handle)
        return handle

    async def _fetch_root(self, document_id: str) -> p:
        """Acquire a root (non-subdocument) document by its local ID.

        Implemented by concrete sources. Resolve with the current mtime and
        raise DocumentNotFoundError if the root is absent from an available
        source. Acquisition is async: it may perform asynchronous work such
        as preparing content.
        """
        ...

    def cache_document(self, handle: "DocumentHandle") -> None:
        """Register a handle in this source's process-local cache.

        Called by subdocument enumeration and any other producer of handles
        so that later fetches of the same full collection-local ID reuse the
        handle instead of re-acquiring it.
        """
        self._handle_cache.cache_document(handle)

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

    def document_id(self, chunk_id: str) -> str:
        return chunk_id.rsplit("?", 1)[0]
