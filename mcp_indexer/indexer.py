import argparse
import asyncio
import heapq
import itertools
import json
import logging
import re
import traceback
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, List, Optional

from sqlalchemy import delete, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from mcp_indexer.context import Context
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.models import ChunkSummary, Document, DocumentChunk, FailedDocument, IndexingWork, deserialize_metadata
from mcp_indexer.plugins.base import DocumentHandle, DocumentSource, DocumentNotFoundError, create_embedding_chunks
from mcp_indexer.retrieval import fetch_chunk

logger = logging.getLogger(__name__)

def normalize_keywords(keywords: str | list[str]) -> list[str]:
    """
    Maps a string or list of keywords to a sorted list of normalized keywords.
    Ignores whitespace in strings, splits on commas, lowercases,
    replaces '_' with '-', and validates against /[-a-z0-9.]+/ .
    """
    if isinstance(keywords, str):
        # Ignore whitespace and split on comma
        raw_list = keywords.replace(" ", "").split(",") if keywords else []
    elif isinstance(keywords, list):
        raw_list = keywords
    else:
        return []

    normalized = set()
    for k in raw_list:
        k = k.strip().lower().replace("_", "-")
        if not k:
            continue
        # Confirm it matches /[-a-z0-9.]+/
        if re.fullmatch(r"[-a-z0-9.]+", k):
            normalized.add(k)

    return sorted(list(normalized))


INDEXING_WORKERS = 64
MONITOR_INTERVAL_SECONDS = 30.0
MONITOR_STALL_THRESHOLD_SECONDS = 40.0
DEBUG_STATS_FILENAME = "indexer-stats.jsonl"

def monitor_time() -> float:
    import time

    return time.monotonic()


@dataclass
class SemanticChunkPlan:
    # Sized embedding chunks for this semantic chunk, in order. The
    # join equals the semantic chunk's text; each chunk's char offset
    # within it is cumulative over the list.
    embedding_chunks: list[str]


@dataclass
class ReconstructedSemanticChunk:
    rows: list[DocumentChunk]
    text: str


class WorkKind(IntEnum):
    """Indexer work, ordered by dispatch priority (smaller is more urgent).

    ``ENSURE`` and ``SUMMARIZE_RESULTS`` are unbounded orchestration
    entries: they queue for priority but run without holding worker
    capacity. They await the content work they submit, so holding capacity
    while awaiting could starve the pool they draw from (with one worker
    slot it would deadlock). Content kinds consume a worker slot each.
    """
    ENSURE = 0
    SUMMARIZE_RESULTS = 1
    DOC_SUMMARY_TRIGGER = 2
    CHUNK_SUMMARY = 3
    DOC_SUMMARY = 4
    INDEX = 5


_UNBOUNDED_KINDS = frozenset({WorkKind.ENSURE, WorkKind.SUMMARIZE_RESULTS})
_DOC_SUMMARY_KINDS = frozenset(
    {WorkKind.DOC_SUMMARY, WorkKind.DOC_SUMMARY_TRIGGER}
)


def _mark_exception_retrieved(fut: asyncio.Future) -> None:
    """Keep a fire-and-forget failure from warning on collection."""
    if not fut.cancelled():
        with suppress(Exception):
            fut.exception()


@dataclass
class _WorkItem:
    """One unit of queued work: its dispatch order and how to run it."""
    order: int
    kind: WorkKind
    factory: Callable[[], Awaitable]
    future: asyncio.Future
    source: DocumentSource | None = None
    document_id: str = ""
    stats: "DocumentIndexingStats | None" = None

    @property
    def bounded(self) -> bool:
        return self.kind not in _UNBOUNDED_KINDS


class DocumentIndexingStats:
    def __init__(
        self,
        document_id: str,
        operation: str = "index",
        semantic_chunks: int = 0,
        embedding_chunks: int = 0,
        summary_spans: int = 0,
        source: Any = None,
    ):
        self.document_id = document_id
        self.operation = operation
        self.semantic_chunks = semantic_chunks
        self.embedding_chunks = embedding_chunks
        self.summary_spans = summary_spans
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "operation": self.operation,
            "semantic_chunks": self.semantic_chunks,
            "embedding_chunks": self.embedding_chunks,
            "summary_spans": self.summary_spans,
        }


class Indexer:
    """
    Orchestrates indexing, chunk summarization, and document summarization.
    """

    def __init__(self, context: Context, debug: bool = False):
        self.context = context
        self.server_config = context.config.get_server_config()
        self.debug = debug
        self._work_capacity = INDEXING_WORKERS
        self._order = itertools.count()
        # Queued work: the heap orders it, the pending map keeps items
        # alive until dispatch (a pop must only ever find live items).
        self._heap: list[tuple[int, int]] = []
        self._pending: dict[int, _WorkItem] = {}
        # Dispatched, unfinished work; the single source of truth for what
        # is running, for the idle signal, and for the capacity gate
        # (only bounded items count against capacity).
        self._in_flight: dict[asyncio.Task, _WorkItem] = {}
        self._capacity_in_use = 0
        # Document summary runs are joined per (collection, document): one
        # queued or in-flight summary per key, regardless of the trigger
        # that scheduled it.
        self._keyed: dict[tuple[str, str], _WorkItem] = {}
        # _wake wakes the dispatcher (submission, or capacity freed);
        # _idle signals wait_for_idle (nothing queued or running).
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._dispatcher: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._debug_stats_path = Path(DEBUG_STATS_FILENAME)
        # Pass-level document summarization; the trigger consults it. The MCP
        # path runs no passes and treats summarization as enabled.
        self._document_summaries_enabled = True

    def _require_built_collections(self) -> None:
        if not self.context.collections:
            raise ValueError(
                "Context has no collections; call Context.build_collections() before creating an Indexer."
            )

    async def _get_failed_documents(self, source: DocumentSource) -> set[str]:
        """Get set of document_ids that failed to process for this collection."""
        async with self.context.get_session() as session:
            result = await session.execute(
                select(FailedDocument.document_id).where(
                    FailedDocument.collection_id == source.id
                )
            )
            return set(result.scalars().all())

    async def _record_failure(self, source: DocumentSource, document_id: str, reason: str) -> None:
        """Record a failure for a document.

        Also retires any pending work entry targeting the failed document:
        handled failures are recorded, not retried, and must not block the
        parent's summarization readiness or reappear on every restart.
        """
        async with self.context.get_session() as session:
            failed = FailedDocument(
                collection_id=source.id,
                document_id=document_id,
                failure_reason=reason,
            )
            await session.merge(failed)
            await session.execute(delete(IndexingWork).where(
                IndexingWork.collection_id == source.id,
                IndexingWork.document_id == document_id,
            ))
            await session.commit()
        # A failed target no longer blocks its parent's readiness (failed
        # children are excluded), so the parent may be ready right now.
        self._trigger_document_summary(source, document_id, for_parent=True)

    async def index_all(
        self,
        *,
        index: bool = True,
        summarize_chunks: bool = True,
        summarize_documents: bool = True,
        maintenance: bool = False,
    ):
        await self.context.build_collections()
        # Pass-level document summarization; completion triggers consult it,
        # so a pass with summarize_documents off stays silent about document
        # summaries even while its indexing and chunk work completes.
        self._document_summaries_enabled = summarize_documents
        loops = []
        for source in self.context.collections.values():
            if maintenance:
                loops.append(self.maintain_collection(source))
            if index:
                loops.append(self.index_collection(source))
            if summarize_chunks:
                loops.append(self.summarize_chunks_collection(source))
            if summarize_documents:
                loops.append(self.summarize_documents_collection(source))

        if loops:
            await asyncio.gather(*loops)
        await self.wait_for_idle()

    async def index_collection(self, source: DocumentSource):
        if source.indexing_mode != "full":
            return
        # Durable work published by earlier parent commits (and not yet
        # retired) is processed first, then source discovery proceeds.
        await self._index_pending_work(source)
        pointers = []
        async for pointer in source.get_documents(last_modified=None):
            pointers.append(pointer)
            if len(pointers) == 256:
                await self._index_discovered_documents(source, pointers)
                pointers = []
        if pointers:
            await self._index_discovered_documents(source, pointers)

    async def _index_pending_work(self, source: DocumentSource) -> None:
        """Process pending indexing_work rows until none remain.

        Work rows are created atomically with a parent's commit; the parent
        then schedules its children directly. Rows therefore survive only
        across process stops, so this pass is the recovery path. Each batch
        is awaited before the next scan: every row seen at the start of an
        iteration is retired by the end of its gather (the index commit or
        the failure record retires it), and newly published rows come only
        from the batch just awaited, so the pass is bounded by the trees it
        recovers.
        """
        while True:
            async with self.context.get_session() as session:
                rows = (await session.execute(
                    select(IndexingWork.document_id)
                    .where(IndexingWork.collection_id == source.id)
                    .order_by(IndexingWork.document_id)
                    .limit(256),
                )).scalars().all()
            if not rows:
                return
            failed_docs = await self._get_failed_documents(source)
            futures = []
            for document_id in rows:
                future = await self._index_work_row(source, document_id, failed_docs)
                if future is not None:
                    futures.append(future)
            if futures:
                await asyncio.gather(*futures, return_exceptions=True)

    async def _retire_work(self, source: DocumentSource, document_id: str) -> None:
        """Delete the pending work entry targeting document_id, if any."""
        async with self.context.get_session() as session:
            await session.execute(delete(IndexingWork).where(
                IndexingWork.collection_id == source.id,
                IndexingWork.document_id == document_id,
            ))
            await session.commit()

    async def _index_discovered_documents(self, source: DocumentSource, pointers) -> None:
        document_ids = [pointer.document_id for pointer in pointers]
        async with self.context.get_session() as session:
            existing_docs = set((await session.execute(select(Document.document_id).where(
                Document.collection_id == source.id,
                Document.document_id.in_(document_ids),
            ))).scalars().all())
            failed_docs = set((await session.execute(select(FailedDocument.document_id).where(
                FailedDocument.collection_id == source.id,
                FailedDocument.document_id.in_(document_ids),
            ))).scalars().all())
        for pointer in pointers:
            if pointer.document_id in existing_docs:
                continue
            if pointer.document_id in failed_docs:
                logger.info(
                    "Skipping failed document %s in collection %s",
                    pointer.document_id,
                    source.id,
                )
                continue

            stats = DocumentIndexingStats(
                document_id=pointer.document_id, operation="index", source=source
            )
            self.submit(
                WorkKind.INDEX,
                lambda pointer=pointer, stats=stats: self.index_document(pointer, stats),
                source=source, document_id=pointer.document_id, stats=stats,
            )

    async def maintain_collection(self, source: DocumentSource) -> None:
        """Refresh/remove stored local IDs, without enumerating the source corpus."""
        after = None
        while True:
            statement = select(Document.document_id, Document.last_modified).where(
                Document.collection_id == source.id,
            ).order_by(Document.document_id).limit(256)
            if after is not None:
                statement = statement.where(Document.document_id > after)
            async with self.context.get_session() as session:
                rows = (await session.execute(statement)).all()
            if not rows:
                return
            for document_id, last_modified in rows:
                try:
                    pointer = await source.fetch_document(document_id)
                except DocumentNotFoundError:
                    async with self.context.get_session() as session:
                        await session.execute(delete(Document).where(
                            Document.collection_id == source.id,
                            Document.document_id == document_id,
                        ))
                        await session.execute(delete(FailedDocument).where(
                            FailedDocument.collection_id == source.id,
                            FailedDocument.document_id == document_id,
                        ))
                        await session.commit()
                    logger.info("Removed %s:%s", source.id, document_id)
                    continue
                except Exception:
                    logger.exception("Could not maintain %s:%s", source.id, document_id)
                    continue
                if pointer.last_modified != last_modified:
                    stats = DocumentIndexingStats(document_id, "refresh", source=source)
                    self.submit(
                        WorkKind.INDEX,
                        lambda pointer=pointer, stats=stats: self.index_document(pointer, stats),
                        source=source, document_id=document_id, stats=stats,
                    )
            after = rows[-1][0]

    async def ensure_document_indexed(
        self, source: DocumentSource, document_id: str, *, retry_failed: bool = False,
    ) -> None:
        """Persist a local document ID if absent; never probe an existing document.

        A child row implies its parent row exists, so when the requested ID
        has no ancestors in the database the missing ancestors are indexed
        first, from the root down to the requested child. Sibling subtrees
        are not completed; they remain ordinary discovered child work.

        The request is queued ahead of other work (unbounded: it holds no
        worker slot while it awaits the index runs it initiates).
        """
        async with self.context.get_session() as session:
            if await session.get(Document, (source.id, document_id)) is not None:
                return
        future = self.submit(
            WorkKind.ENSURE,
            lambda: self._ensure_indexed_walk(source, document_id, retry_failed=retry_failed),
            source=source, document_id=document_id,
        )
        await future

    async def _ensure_indexed_walk(
        self, source: DocumentSource, document_id: str, *, retry_failed: bool,
    ) -> None:
        """Index missing ancestors, from the root down to the requested ID."""
        # Walk up the ID hierarchy, collecting missing ancestors.
        missing_ancestors: list[str] = []
        node = document_id
        while True:
            parent, _, _ = node.rpartition(":")
            if not parent:
                break
            async with self.context.get_session() as session:
                if await session.get(Document, (source.id, parent)) is not None:
                    break
            missing_ancestors.append(parent)
            node = parent
        for ancestor in reversed(missing_ancestors):
            await self._ensure_document_row(source, ancestor, retry_failed=retry_failed)
        await self._ensure_document_row(source, document_id, retry_failed=retry_failed)

    async def _ensure_document_row(
        self, source: DocumentSource, document_id: str, *, retry_failed: bool = False,
    ) -> None:
        """Index a single document ID (callers establish ancestor rows).

        The index run is queued content work whose failure record is owned
        by its completion, so a failure simply raises to the caller.
        """
        async with self.context.get_session() as session:
            failure = await session.get(FailedDocument, (source.id, document_id))
        if failure is not None and not retry_failed:
            raise RuntimeError(f"Previously failed document {document_id}: {failure.failure_reason}")
        stats = DocumentIndexingStats(document_id, "index", source=source)
        await self.submit(
            WorkKind.INDEX,
            lambda: self._index_fetched(source, document_id, stats),
            source=source, document_id=document_id, stats=stats,
        )

    async def _index_fetched(
        self, source: DocumentSource, document_id: str, stats: DocumentIndexingStats,
    ) -> None:
        """Acquire the document and index it; the run's work item awaits it."""
        await self.index_document(await source.fetch_document(document_id), stats)

    async def summarize_chunks_collection(self, source: DocumentSource):
        # Get set of failed document_ids to skip
        failed_docs = await self._get_failed_documents(source)
        
        async with self.context.get_session() as session:
            stmt = (
                select(DocumentChunk.document_id)
                .where(
                    DocumentChunk.collection_id == source.id,
                    DocumentChunk.summary_span.is_(None),
                )
                .distinct()
                .order_by(DocumentChunk.document_id)
            )
            if failed_docs:
                stmt = stmt.where(DocumentChunk.document_id.not_in(failed_docs))
            result = await session.execute(stmt)
            document_ids = result.scalars().all()

        if not document_ids:
            return

        for document_id in document_ids:
            stats = DocumentIndexingStats(
                document_id=document_id, operation="chunk_summary", source=source
            )
            self.submit(
                WorkKind.CHUNK_SUMMARY,
                lambda document_id=document_id: self.summarize_document_chunks(source, document_id),
                source=source, document_id=document_id, stats=stats,
            )

        await self.wait_for_idle()

    async def summarize_results(
        self, collection_id: str, document_ids: Iterable[str]
    ) -> None:
        """Bring the requested documents' summary trees to summarized state.

        Each requested document is indexed first (establishing its row and
        its ancestors'), and its tree is then summarized bottom-up to the
        ultimate parent -- the full subtree, since a parent's summary is
        built from its children's.

        The request is queued as a single unbounded entry behind only
        other requests of its kind; the work it initiates -- ancestor
        indexing, pending child work, chunk summaries, and document
        summaries -- joins the queue at its own priorities. The entry
        awaits only the work it initiates (and runs already queued or in
        flight for the same document); it never waits on the process idle
        mechanism, and duplicate work with concurrent requests is accepted
        (document summary runs are joined instead). Failed documents
        follow the general failure rules.
        """
        document_ids = set(document_ids)
        if not document_ids:
            return

        source = self.context.collections[collection_id]
        future = self.submit(
            WorkKind.SUMMARIZE_RESULTS,
            lambda: self._summarize_results_walk(source, sorted(document_ids)),
            source=source,
            document_id=",".join(sorted(document_ids)),
        )
        await future

    async def _summarize_results_walk(
        self, source: DocumentSource, document_ids: list[str],
    ) -> None:
        """Converge each requested document's tree, bottom-up, awaited."""
        failed_docs = await self._get_failed_documents(source)
        for document_id in document_ids:
            if document_id in failed_docs:
                continue
            await self.ensure_document_indexed(source, document_id)

        live_ids = [document_id for document_id in document_ids if document_id not in failed_docs]
        ultimates = sorted({
            await self._ultimate_parent(source, document_id)
            for document_id in live_ids
        })
        for ultimate in ultimates:
            await self._summarize_subtree(source, ultimate)

    async def _ultimate_parent(self, source: DocumentSource, document_id: str) -> str:
        """Walk the stored parent chain up to the root of the tree."""
        node = document_id
        while True:
            async with self.context.get_session() as session:
                doc = await session.get(Document, (source.id, node))
            parent = doc.parent_document_id if doc is not None else None
            if parent is None:
                return node
            node = parent

    async def _summarize_subtree(self, source: DocumentSource, document_id: str) -> None:
        """Converge the document's subtree to summarized state, bottom-up.

        Pending child work is indexed first, the children's subtrees are
        converged, then the document's own chunk summaries and document
        summary run. A summary that is not ready when it runs simply yields
        no summary (the readiness gate is not overridden); completion
        triggers keep the tree converging in the background.
        """
        failed_docs = await self._get_failed_documents(source)
        if document_id in failed_docs:
            return
        await self._index_subtree_work(source, document_id)
        async with self.context.get_session() as session:
            child_ids = (await session.execute(
                select(Document.document_id).where(
                    Document.collection_id == source.id,
                    Document.parent_document_id == document_id,
                )
            )).scalars().all()
        if child_ids:
            await asyncio.gather(*(
                self._summarize_subtree(source, child_id)
                for child_id in child_ids
            ))
        await self._summarize_result_chunks(source, document_id)
        async with self.context.get_session() as session:
            needs_summary = (await session.execute(
                select(Document.summary).where(
                    Document.collection_id == source.id,
                    Document.document_id == document_id,
                )
            )).scalar_one_or_none()
        if needs_summary == "":
            await self._submit_document_summary(
                source, document_id, WorkKind.DOC_SUMMARY, join=True,
            )

    async def _index_subtree_work(
        self, source: DocumentSource, document_id: str,
    ) -> None:
        """Index the document's pending direct-child work targets, awaited.

        The work rows were published atomically with the parent's last
        index commit; each target is resolved once here, and its row retires
        with the index commit (or the failure record). Index failures are
        absorbed: the failed target is excluded from the tree per the
        general failure rules.
        """
        async with self.context.get_session() as session:
            rows = (await session.execute(
                select(IndexingWork.document_id).where(
                    IndexingWork.collection_id == source.id,
                    IndexingWork.parent_document_id == document_id,
                )
            )).scalars().all()
        if not rows:
            return
        failed_docs = await self._get_failed_documents(source)
        futures = []
        for child_id in rows:
            future = await self._index_work_row(source, child_id, failed_docs)
            if future is not None:
                futures.append(future)
        if futures:
            await asyncio.gather(*futures, return_exceptions=True)

    async def _index_work_row(
        self,
        source: DocumentSource,
        document_id: str,
        failed_docs: set[str],
    ) -> asyncio.Future | None:
        """Resolve one pending work row targeting document_id.

        Retires the row when the target is failed, gone, or unchanged since
        it was last indexed; otherwise queues an index run and returns its
        future. The row retires atomically with the index commit (or the
        failure record).
        """
        if document_id in failed_docs:
            # Failure exclusion: a failed target never stalls a parent or
            # reappears on every restart.
            await self._retire_work(source, document_id)
            return None
        try:
            pointer = await source.fetch_document(document_id)
        except DocumentNotFoundError:
            # The child is gone; the parent's next successful enumeration
            # (or maintenance) reconciles any row.
            await self._retire_work(source, document_id)
            return None
        async with self.context.get_session() as session:
            stored = await session.get(Document, (source.id, document_id))
        if stored is not None and stored.last_modified == pointer.last_modified:
            # Unchanged since it was indexed; the work is done.
            await self._retire_work(source, document_id)
            return None
        stats = DocumentIndexingStats(
            document_id=document_id, operation="index", source=source,
        )
        return self.submit(
            WorkKind.INDEX,
            lambda pointer=pointer, stats=stats: self.index_document(pointer, stats),
            source=source, document_id=document_id, stats=stats,
        )

    async def _summarize_result_chunks(
        self, source: DocumentSource, document_id: str,
    ) -> None:
        """Chunk-level summaries for one document, awaited to completion."""
        async with self.context.get_session() as session:
            result = await session.execute(
                select(DocumentChunk.document_id).where(
                    DocumentChunk.collection_id == source.id,
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.summary_span.is_(None),
                ).distinct()
            )
            needs_chunks = result.first() is not None
        if not needs_chunks:
            return
        stats = DocumentIndexingStats(
            document_id=document_id, operation="chunk_summary", source=source,
        )
        await self.submit(
            WorkKind.CHUNK_SUMMARY,
            lambda: self.summarize_document_chunks(source, document_id),
            source=source, document_id=document_id, stats=stats,
        )

    async def _summarization_ready(self, source: DocumentSource, document_id: str) -> bool:
        """Tree readiness for bottom-up document summarization.

        Ready when no work entry targets the document or any descendant
        (failed targets retire their work entries when the failure is
        recorded, so they neither block nor retry) and every non-failed
        child row has a summary. Never waits; a not-ready document is left
        for a completion trigger or a later pass.
        """
        if await self._has_pending_subtree_work(source, document_id):
            return False
        failed_docs = await self._get_failed_documents(source)
        async with self.context.get_session() as session:
            rows = (await session.execute(
                select(Document.document_id, Document.summary).where(
                    Document.collection_id == source.id,
                    Document.parent_document_id == document_id,
                )
            )).all()
        for child_id, child_summary in rows:
            if child_id in failed_docs:
                continue
            if child_summary == "":
                return False
        return True

    async def _has_pending_subtree_work(
        self, source: DocumentSource, document_id: str
    ) -> bool:
        """True while any work entry targets the document or a descendant.

        Descendant IDs extend the document's ID with hierarchy separators,
        so the exact ID plus a prefix match covers the subtree; LIKE
        metacharacters in the ID are escaped.
        """
        escaped = (
            document_id.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        async with self.context.get_session() as session:
            result = await session.execute(
                select(IndexingWork.document_id).where(
                    IndexingWork.collection_id == source.id,
                    or_(
                        IndexingWork.document_id == document_id,
                        IndexingWork.document_id.like(
                            f"{escaped}:%", escape="\\",
                        ),
                    ),
                ).limit(1)
            )
            return result.first() is not None

    async def summarize_documents_collection(self, source: DocumentSource):
        """Single catch-up pass: schedule a summary for every doc ready now.

        Not a pump: documents that become ready while this pass's tasks run
        are picked up by the completion triggers (which fire on the parent
        when a child's summary lands), and work that finished in an earlier
        process is caught here because triggers are process-local. The pass
        boundary's wait_for_idle converges the tasks the triggers schedule.
        Failed documents follow the general failure rules and are excluded.
        """
        failed_docs = await self._get_failed_documents(source)
        async with self.context.get_session() as session:
            result = await session.execute(
                select(Document.document_id).where(
                    Document.collection_id == source.id,
                    Document.summary == "",
                )
            )
            candidates = result.scalars().all()
        for document_id in candidates:
            if document_id in failed_docs:
                continue
            if not await self._summarization_ready(source, document_id):
                continue
            await self._submit_document_summary(
                source, document_id, WorkKind.DOC_SUMMARY, join=False,
            )

    def submit(
        self,
        kind: WorkKind,
        factory: Callable[[], Awaitable],
        *,
        source: DocumentSource | None = None,
        document_id: str = "",
        stats: Optional[DocumentIndexingStats] = None,
    ) -> asyncio.Future:
        """Queue one unit of work and return a future for its result.

        A single dispatcher pulls work from a priority heap, spawning one
        task per item; done callbacks free capacity and wake it. Bounded
        kinds (content work) dispatch as worker slots free; unbounded kinds
        (the ensure_document_indexed and summarize_results entries)
        dispatch whenever their priority turn comes. The factory is called
        at dispatch time, so an item's inputs are captured by closure.

        Document summary runs are keyed by (collection, document): a
        submission for a summary already queued or in flight returns that
        run's future instead of scheduling a duplicate.
        """
        loop = asyncio.get_running_loop()
        if kind in _DOC_SUMMARY_KINDS and source is not None:
            existing = self._keyed.get((source.id, document_id))
            if existing is not None:
                return existing.future
        item = _WorkItem(
            order=next(self._order),
            kind=kind,
            factory=factory,
            future=loop.create_future(),
            source=source,
            document_id=document_id,
            stats=stats,
        )
        item.future.add_done_callback(_mark_exception_retrieved)
        if kind in _DOC_SUMMARY_KINDS and source is not None:
            self._keyed[(source.id, document_id)] = item
        self._pending[item.order] = item
        heapq.heappush(self._heap, (kind, item.order))
        self._idle.clear()
        self._wake.set()
        self._ensure_dispatcher()
        return item.future

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._dispatch_loop())

    async def _dispatch_loop(self) -> None:
        try:
            while True:
                self._dispatch_ready()
                if not self._in_flight and not self._pending:
                    return
                self._wake.clear()
                if self._dispatch_ready():
                    # A submission raced the clear; loop and dispatch it.
                    continue
                await self._wake.wait()
        finally:
            if self._dispatcher is asyncio.current_task():
                self._dispatcher = None

    def _dispatch_ready(self) -> bool:
        """Dispatch every queued item that can run now; report progress."""
        dispatched = False
        while self._heap:
            item = self._pending.get(self._heap[0][1])
            if item is None:
                heapq.heappop(self._heap)
                continue
            if item.bounded and self._capacity_in_use >= self._work_capacity:
                break
            heapq.heappop(self._heap)
            del self._pending[item.order]
            self._dispatch_item(item)
            dispatched = True
        return dispatched

    def _dispatch_item(self, item: _WorkItem) -> None:
        # Submission never waits for a slot (a parent scheduling children
        # must not block on child capacity), so the capacity gate is only
        # the check above; the work itself runs in its own task. The runner
        # keeps a synchronous factory failure from killing the dispatcher:
        # it lands as an ordinary task failure.
        task = asyncio.create_task(self._run_item(item))
        task.add_done_callback(lambda done, item=item: self._on_work_done(item, done))
        self._in_flight[task] = item
        if item.bounded:
            self._capacity_in_use += 1
        self._ensure_monitor_task()

    async def _run_item(self, item: _WorkItem):
        return await item.factory()

    async def _submit_document_summary(
        self,
        source: DocumentSource,
        document_id: str,
        kind: WorkKind,
        *,
        join: bool,
    ) -> bool | None:
        """Queue a document summary run, joining any queued or in-flight one.

        Runs for one (collection, document) are submitted at most once: the
        key lookup in ``submit`` returns the existing run's future when a
        concurrent trigger, catch-up scan, or on-demand walk asks for the
        same summary. With ``join`` the call awaits the run and returns
        whether a summary was committed; otherwise it returns once the run
        is queued.
        """
        stats = DocumentIndexingStats(
            document_id=document_id, operation="document_summary", source=source,
        )
        future = self.submit(
            kind,
            lambda: self.summarize_document(source, document_id),
            source=source, document_id=document_id, stats=stats,
        )
        if not join:
            return None
        try:
            return bool(await future)
        except Exception:
            # The failure is recorded by the run's completion; a failed run
            # simply yields no summary.
            return False

    def _trigger_document_summary(
        self, source: DocumentSource, document_id: str, *, for_parent: bool,
    ) -> None:
        """Completion trigger.

        When content work succeeds on a document -- or a failure is recorded
        for one -- exactly one document's summary readiness just changed: the
        document itself for its own indexing/chunk work, its parent for a
        finished (or failed) child. If document summarization is enabled for
        this run, queue a summary on it; the keyed join in ``submit``
        deduplicates against concurrent triggers.
        """
        if not self._document_summaries_enabled:
            return
        if for_parent:
            parent = self._parent_of(document_id)
            if parent is None:
                return
            document_id = parent
        stats = DocumentIndexingStats(
            document_id=document_id, operation="document_summary", source=source,
        )
        self.submit(
            WorkKind.DOC_SUMMARY_TRIGGER,
            lambda: self.summarize_document(source, document_id),
            source=source, document_id=document_id, stats=stats,
        )

    def _ensure_monitor_task(self) -> None:
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(
                self._monitor_running_documents()
            )

    async def _monitor_running_documents(self) -> None:
        last_run = monitor_time()
        previous_tasks: set[asyncio.Task] = set()
        try:
            while True:
                running_items = list(self._in_flight.items())
                if not running_items:
                    return
                await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
                now = monitor_time()
                elapsed = now - last_run
                last_run = now

                if elapsed > MONITOR_STALL_THRESHOLD_SECONDS:
                    logger.error(
                        "Indexing monitor woke after %.2fs, exceeding the %.2fs threshold",
                        elapsed,
                        MONITOR_STALL_THRESHOLD_SECONDS,
                    )

                running_items = list(self._in_flight.items())
                if not running_items:
                    return

                current_tasks = {task for task, _ in running_items}
                tasks_to_log = current_tasks & previous_tasks
                previous_tasks = current_tasks

                for task, item in running_items:
                    if task not in tasks_to_log:
                        continue
                    stats = item.stats
                    logger.info(
                        "Still running %s document_id=%s semantic_chunks=%d embedding_chunks=%d summary_spans=%d",
                        item.kind.name.lower(),
                        item.document_id,
                        stats.semantic_chunks if stats else 0,
                        stats.embedding_chunks if stats else 0,
                        stats.summary_spans if stats else 0,
                    )
        except asyncio.CancelledError:
            raise
        finally:
            if self._monitor_task is asyncio.current_task():
                self._monitor_task = None
            # Set idle if monitor task completes and there is no queued
            # or running work
            if not self._in_flight and not self._pending:
                self._idle.set()

    def _on_work_done(self, item: _WorkItem, task: asyncio.Task) -> None:
        self._in_flight.pop(task, None)
        if item.bounded:
            self._capacity_in_use -= 1
        if item.source is not None:
            key = (item.source.id, item.document_id)
            if self._keyed.get(key) is item:
                del self._keyed[key]

        if task.cancelled():
            if not item.future.done():
                item.future.cancel()
        else:
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "%s work failed for %s:%s",
                    item.kind.name.lower(),
                    item.source.id if item.source is not None else "?",
                    item.document_id or "?",
                    exc_info=exc,
                )
                # A failed content run is recorded (not retried) so a failed
                # target never blocks its parent's readiness or reappears on
                # every restart; orchestration entries own their own errors.
                if item.source is not None and item.kind not in _UNBOUNDED_KINDS:
                    tb_str = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    # Truncate long tracebacks
                    if len(tb_str) > 1000:
                        tb_str = tb_str[:997] + "..."
                    asyncio.create_task(self._record_failure(
                        item.source, item.document_id,
                        f"{type(exc).__name__}: {exc}\n{tb_str}",
                    ))
            if not item.future.done():
                if exc is not None:
                    item.future.set_exception(exc)
                else:
                    item.future.set_result(task.result())

        if item.stats is not None and self.debug:
            self._append_debug_stats(item.stats)

        self._wake.set()
        if not self._in_flight and not self._pending:
            self._idle.set()
        if not self._in_flight:
            if self._monitor_task is not None and not self._monitor_task.done():
                self._monitor_task.cancel()

    def _append_debug_stats(self, stats: DocumentIndexingStats) -> None:
        import json

        with self._debug_stats_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stats.to_dict()))
            handle.write("\n")
            handle.flush()

    async def wait_for_idle(self):
        while True:
            await self._idle.wait()

            if self._monitor_task is not None:
                with suppress(asyncio.CancelledError):
                    await self._monitor_task

            if not self._in_flight:
                return

    async def index_document(self, pointer, stats: DocumentIndexingStats):
        source = pointer.source
        embedding_texts: list[str] = []
        embedding_offsets: list[int] = []
        embedding_metadata: list[dict[str, Any]] = []
        semantic_chunk_ids: list[int] = []
        semantic_lengths: list[int] = []
        async for meta, text_list in pointer.get_chunks(
            self.server_config.min_size, self.server_config.max_size
        ):
            plan = self._build_semantic_chunk_plan(
                meta,
                text_list,
                self.server_config.min_size,
                self.server_config.max_size,
            )
            stats.semantic_chunks += 1
            stats.embedding_chunks += len(plan.embedding_chunks)
            # The first embedding chunk of this semantic chunk is the next
            # global row; every embedding chunk in the chunk points back to it.
            first_order = len(embedding_texts)
            # Denormalized semantic length: the length of the semantic
            # chunk's text, duplicated across every embedding chunk in it.
            semantic_length = len("".join(text_list))
            offset = 0
            for e_text in plan.embedding_chunks:
                embedding_texts.append(e_text)
                embedding_offsets.append(offset)
                embedding_metadata.append(meta)
                offset += len(e_text)
                semantic_chunk_ids.append(first_order)
                semantic_lengths.append(semantic_length)

        batch_embeddings = (
            await self._embed_batch(embedding_texts)
            if embedding_texts
            else []
        )
        if len(batch_embeddings) != len(embedding_texts):
            raise ValueError(
                f"Embedding batch returned {len(batch_embeddings)} vectors for {len(embedding_texts)} chunks"
            )

        all_chunks = []
        for order, (e_text, embedding_offset, e_meta, embedding,
                    semantic_chunk_id, semantic_length) in enumerate(
            zip(embedding_texts, embedding_offsets, embedding_metadata,
                batch_embeddings, semantic_chunk_ids, semantic_lengths,
                strict=True),
        ):
            all_chunks.append(
                DocumentChunk(
                    collection_id=source.id,
                    document_id=pointer.document_id,
                    order=order,
                    chunk_id=f"{pointer.document_id}?c={order}",
                    text=e_text if pointer.save_text else None,
                    embedding=embedding,
                    summary_span=None,
                    semantic_chunk_id=semantic_chunk_id,
                    embedding_offset=embedding_offset,
                    semantic_length=semantic_length,
                    embedding_length=len(e_text),
                    metadata_str=json.dumps(e_meta),
                )
            )

        # Enumerate the immediate subdocuments. Their work entries are
        # published atomically with this document's own rows, and a
        # complete enumeration is what permits reconciling removed
        # children. An enumeration failure propagates through the general
        # failure handling; it never establishes absence.
        child_handles: list[DocumentHandle] = [
            child async for child in pointer.get_sub_documents()
        ]

        metadata_dict = await pointer.get_metadata()
        doc_record = Document(
            collection_id=source.id,
            document_id=pointer.document_id,
            parent_document_id=self._parent_of(pointer.document_id),
            title=metadata_dict.get("title", "Untitled"),
            title_strength=int(metadata_dict.get("title_strength", 0)),
            last_modified=pointer.last_modified,
            summary="",
            embedding=[0.0] * VECTOR_DIMENSIONS,
            keywords=normalize_keywords(metadata_dict.get("keywords", [])),
            metadata_dict={
                key: value
                for key, value in metadata_dict.items()
                if key not in ("title", "title_strength", "keywords", "last_modified")
            },
        )
        # Attach chunks to document; the merge cascades them on insert and
        # replaces the previous chunk set on refresh.
        doc_record.chunks = all_chunks
        # Existing child mtimes are read before the commit so the
        # post-commit scheduling below can skip unchanged children.
        stored_children = await self._direct_child_mtimes(source, pointer.document_id)
        await self._index_document(
            source, doc_record,
            child_ids=[child.document_id for child in child_handles],
        )

        # Post-commit scheduling: reuse the already-enumerated handles
        # (the process-local cache keeps them cheap) through the priority
        # queue, without waiting for child capacity or completion. Failure
        # exclusion applies here as well as to restart work.
        failed_docs = await self._get_failed_documents(source)
        for child in child_handles:
            child_id = child.document_id
            if child_id in failed_docs:
                continue
            stored_mtime = stored_children.get(child_id)
            if stored_mtime is not None and stored_mtime == child.last_modified:
                continue
            stats = DocumentIndexingStats(
                document_id=child_id, operation="index", source=source,
            )
            self.submit(
                WorkKind.INDEX,
                lambda child=child, stats=stats: self.index_document(child, stats),
                source=source, document_id=child_id, stats=stats,
            )
        # Indexing complete: this document's own summary inputs may now
        # exist (a summary-only leaf, or freshly ingested chunks).
        self._trigger_document_summary(
            source, pointer.document_id, for_parent=False,
        )

    @staticmethod
    def _parent_of(document_id: str) -> str | None:
        """The parent ID of a collection-local ID, or None for a root.

        The rightmost hierarchy separator (``:``) separates a child from
        its parent; the parent is always an exact ID prefix of the child.
        """
        parent, _, _ = document_id.rpartition(":")
        return parent or None

    async def _direct_child_mtimes(
        self, source: DocumentSource, document_id: str
    ) -> dict[str, datetime]:
        """Stored last_modified values of the direct children with rows."""
        async with self.context.get_session() as session:
            rows = (await session.execute(
                select(Document.document_id, Document.last_modified).where(
                    Document.collection_id == source.id,
                    Document.parent_document_id == document_id,
                )
            )).all()
        return {row[0]: row[1] for row in rows}

    async def summarize_document_chunks(
        self, source: DocumentSource, document_id: str
    ) -> bool:
        async with self.context.get_session() as session:
            doc = await session.get(Document, (source.id, document_id))
        if doc is None:
            return False
        expected_mtime = doc.last_modified
        semantic_chunks = await self._reconstruct_semantic_chunks(source, document_id)

        if not semantic_chunks:
            return False

        summary_prompt = source.config.chunk_summary_prompt or source.config.doc_summary_prompt

        # Plan the summary spans: one per semantic chunk, split only when the
        # semantic chunk exceeds max_to_summarize, and never splitting an
        # embedding chunk. Returns the texts to summarize and each embedding
        # chunk's span index.
        span_texts, chunk_spans = self._plan_summarization_spans(
            semantic_chunks, self.server_config.max_to_summarize
        )

        # Submit across semantic boundaries so the model batcher can stay full.
        summaries = await asyncio.gather(*[
            self._summarize_text(text, summary_prompt) for text in span_texts
        ])
        doc_summaries = [
            ChunkSummary(
                collection_id=source.id,
                document_id=document_id,
                summary_span=offset,
                summary=summary,
            )
            for offset, summary in enumerate(summaries)
        ]

        # Write in a short session after the model work. The row lock makes the
        # mtime read and the write atomic: if a maintenance refresh lands first,
        # we see its new last_modified and skip; if we win, the refresh blocks
        # on the lock and then overwrites with the current content. A stale
        # summary can therefore never be committed.
        async with self.context.get_session() as session:
            doc = await session.get(
                Document, (source.id, document_id),
                options=[selectinload(Document.chunks),
                         selectinload(Document.summaries)],
                with_for_update=True,
            )
            if doc is None or doc.last_modified != expected_mtime:
                return False
            for chunk in doc.chunks:
                span_index = chunk_spans.get(chunk.chunk_id)
                if span_index is not None:
                    chunk.summary_span = span_index
            doc.summaries = doc_summaries
            await session.commit()
        # Span summaries are the document's own summary inputs.
        self._trigger_document_summary(
            source, document_id, for_parent=False,
        )
        return True

    async def summarize_document(
        self, source: DocumentSource, document_id: str
    ) -> bool:
        async with self.context.get_session() as session:
            doc = await session.get(
                Document, (source.id, document_id),
                options=[selectinload(Document.summaries)],
            )
            child_rows = (await session.execute(
                select(Document.document_id, Document.summary).where(
                    Document.collection_id == source.id,
                    Document.parent_document_id == document_id,
                )
            )).all()
        if doc is None:
            return False
        # Tree readiness: descendant work has completed or failed, and
        # available child summaries have been produced. The check is a pure
        # read: it never waits on descendant work, so a not-ready document
        # yields its run (and its capacity) to the queue; the document is
        # left for a completion trigger or a later pass.
        if not await self._summarization_ready(source, document_id):
            return False
        expected_mtime = doc.last_modified
        pointer = await source.fetch_document(document_id)
        metadata_dict = await pointer.get_metadata()

        doc_summary = await pointer.get_document_summary()
        child_summaries = [
            (child_id, child_summary)
            for child_id, child_summary in child_rows
            if child_summary
        ]
        # Summary-only documents (e.g. an image) use the handle-provided
        # summary as-is, with no second LLM pass over it. Everything else is
        # composed by the LLM from the available inputs below; a
        # handle-provided summary never bypasses child-summary incorporation
        # for a parent with subdocuments.
        if not (doc_summary and not child_summaries):
            span_summaries = [s.summary for s in sorted(
                doc.summaries, key=lambda s: s.summary_span
            ) if s.summary]
            sections: list[str] = []
            if doc_summary:
                # Handles supply a bounded excerpt, not the full text;
                # label it as such.
                sections.append(
                    f"Excerpt of the document's text:\n{doc_summary}"
                )
            if span_summaries:
                sections.append(
                    "Summaries of parts of the document:\n"
                    + "\n\n".join(span_summaries)
                )
            if child_summaries:
                # Descendant information propagates through immediate-child
                # summaries, not by flattening the whole tree.
                sections.append(
                    "Summaries of subdocuments within the document:\n"
                    + "\n".join(
                        f"{child_id}: {child_summary}"
                        for child_id, child_summary in child_summaries
                    )
                )
            if not sections:
                return False
            doc_summary_list = await self.context.llm(
                [[
                    (
                        "system",
                        source.config.doc_summary_prompt,
                    ),
                    (
                        "user",
                        "Write no more than two paragraphs to summarize the "
                        "following document:\n\n" + "\n\n".join(sections),
                    ),
                ]]
            )
            doc_summary = doc_summary_list[0]

        # Extract title and keywords from the summary if present.
        summary_lines = doc_summary.splitlines()
        extracted_title: str | None = None
        extracted_keywords: list[str] = []
        filtered_lines: list[str] = []

        for line in summary_lines:
            stripped = line.strip()
            # Check for "title:" line (case-insensitive)
            if stripped.lower().startswith("title:") and extracted_title is None:
                title_part = stripped[6:].strip()  # After "title:"
                if title_part:
                    extracted_title = title_part
                    # Don't include this line in the stored summary
                    continue
            # Extract keywords line
            if stripped.lower().startswith("keywords:"):
                kw_part = stripped[9:].strip() # After "keywords:"
                extracted_keywords = normalize_keywords(kw_part)
                continue
            filtered_lines.append(line)

        # Rebuild the summary without the extracted lines
        final_summary = "\n".join(filtered_lines).strip()

        doc_vector = (await self.context.embedding([final_summary]))[0]

        # Use extracted title if found, otherwise fall back to metadata
        title = extracted_title if extracted_title else metadata_dict.get("title", "Untitled")
        title_strength = 9 if extracted_title else int(metadata_dict.get("title_strength", 0))

        # Write through the loaded document: changed attributes only. The row
        # lock makes the mtime read and the write atomic, so this can never
        # commit a summary for content that is no longer current.
        async with self.context.get_session() as session:
            doc = await session.get(
                Document, (source.id, document_id), with_for_update=True
            )
            if doc is None or doc.last_modified != expected_mtime:
                return False
            doc.title = title
            doc.title_strength = title_strength
            doc.summary = final_summary
            doc.embedding = doc_vector
            doc.keywords = (extracted_keywords if extracted_keywords else
                            normalize_keywords(metadata_dict.get("keywords", [])))
            await session.commit()
        # This summary is now an input to the parent's summary.
        self._trigger_document_summary(
            source, document_id, for_parent=True,
        )
        return True

    async def _index_document(
        self, source: DocumentSource, doc: Document, child_ids: list[str],
    ) -> None:
        """Persist the prepared document, fully replacing any previous version.

        The document, its own chunk rows, and the ``indexing_work`` entries
        for the enumerated child IDs are committed together. A complete
        enumeration also reconciles removed children: their subtrees and
        work entries are deleted in the same transaction.
        """
        async with self.context.get_session() as session:
            # The transient clobbers the previous version: every field is
            # written, the chunk set is replaced (delete-orphan), and previous
            # chunk summaries are cleared so summarization re-queues.
            doc.summaries = []
            for _ in range(2):
                try:
                    # A refresh may delete a semantic head while its
                    # members still reference it (delete-orphan flush
                    # order); the NO ACTION self-FK forbids that
                    # mid-transaction, so defer it. Both rows leave the
                    # table in this transaction, so commit-time
                    # validation still rejects a real violation.
                    await session.execute(text(
                        "SET CONSTRAINTS fk_document_chunk_semantic DEFERRED"))
                    await session.merge(doc)
                    await self._publish_child_work(
                        session, source.id, doc.document_id, child_ids,
                    )
                    await session.execute(delete(FailedDocument).where(
                        FailedDocument.collection_id == source.id,
                        FailedDocument.document_id == doc.document_id,
                    ))
                    await session.commit()
                    return
                except IntegrityError:
                    # A concurrent index committed this document first; re-merge
                    # the same prepared document so the latest write wins.
                    await session.rollback()
            raise RuntimeError(
                f"Could not index {source.id}:{doc.document_id} after concurrent conflict"
            )

    async def _publish_child_work(
        self, session, collection_id: str, document_id: str, child_ids: list[str],
    ) -> None:
        """Reconcile and publish child work within the parent's transaction.

        Successful enumeration permits deletion: direct children absent from
        the enumeration lose their Document subtrees (the parent FK cascades)
        and their work entries. The document's own pending work is retired,
        and a work entry is (re)published for every discovered child --
        including children with rows or failures, which selection-time
        exclusion handles.
        """
        if child_ids:
            stmt = select(Document.document_id).where(
                Document.collection_id == collection_id,
                Document.parent_document_id == document_id,
                Document.document_id.not_in(child_ids),
            )
        else:
            stmt = select(Document.document_id).where(
                Document.collection_id == collection_id,
                Document.parent_document_id == document_id,
            )
        stale_children = (await session.execute(stmt)).scalars().all()
        if stale_children:
            await session.execute(delete(IndexingWork).where(
                IndexingWork.collection_id == collection_id,
                IndexingWork.document_id.in_(stale_children),
            ))
            # The Document rows; fk_document_parent's ON DELETE CASCADE removes
            # each child's subtree, chunks, summaries, and work entries where
            # it is the parent.
            await session.execute(delete(Document).where(
                Document.collection_id == collection_id,
                Document.document_id.in_(stale_children),
            ))
        # The document's own pending work is complete with this commit.
        await session.execute(delete(IndexingWork).where(
            IndexingWork.collection_id == collection_id,
            IndexingWork.document_id == document_id,
        ))
        for child_id in child_ids:
            await session.merge(IndexingWork(
                collection_id=collection_id,
                document_id=child_id,
                parent_document_id=document_id,
            ))

    async def _reconstruct_semantic_chunks(
        self, source: DocumentSource, document_id: str
    ) -> list[ReconstructedSemanticChunk]:
        async with self.context.get_session() as session:
            result = await session.execute(
                select(DocumentChunk).where(
                    DocumentChunk.collection_id == source.id,
                    DocumentChunk.document_id == document_id,
                )
            )
            document_rows = result.scalars().all()

        document_rows.sort(key=lambda row: row.order)

        grouped_rows: dict[int, list[DocumentChunk]] = {}
        for row in document_rows:
            grouped_rows.setdefault(row.semantic_chunk_id, []).append(row)

        semantic_chunks = [
            ReconstructedSemanticChunk(
                rows=rows,
                text=await fetch_chunk(source, rows[0].chunk_id, scope="semantic"),
            )
            for rows in grouped_rows.values()
        ]
        return semantic_chunks

    async def _summarize_text(self, text: str, prompt: str) -> str:
        res = await self.context.llm(
            [[
                ("system", prompt),
                ("user", f"Write no more than three sentences to summarize this chunk:\n\n{text}"),
            ]]
        )
        return res[0]

    def _build_semantic_chunk_plan(
        self,
        metadata: dict[str, Any],
        text_list: list[str],
        min_size: int,
        max_size: int,
    ) -> SemanticChunkPlan:
        return SemanticChunkPlan(
            embedding_chunks=create_embedding_chunks(
                text_list, min_size, max_size
            ),
        )

    def _plan_summarization_spans(
        self,
        semantic_chunks: list[ReconstructedSemanticChunk],
        max_to_summarize: int,
    ) -> tuple[list[str], dict[str, int]]:
        """Plan summary spans and map each embedding chunk to its span.

        A semantic chunk normally yields one span. It is split into several
        spans only when it exceeds max_to_summarize, and the split groups whole
        embedding chunks (ordered by their in-chunk offset) so a single
        embedding chunk is never split across spans. Returns the span texts to
        summarize and a chunk_id -> absolute span index mapping.
        """
        span_texts: list[str] = []
        chunk_by_id: dict[str, int] = {}

        def size_of(group: list[str]) -> int:
            return sum(len(part) for part in group)

        for semantic_chunk in semantic_chunks:
            base = len(span_texts)
            text = semantic_chunk.text
            rows = sorted(
                semantic_chunk.rows,
                key=lambda r: r.embedding_offset,
            )

            # Group whole embedding chunks under the budget; a chunk that
            # alone exceeds the budget becomes its own single span.
            spans: list[list[str]] = []
            current: list[str] = []
            chunk_span: list[int] = []
            for row in rows:
                o = row.embedding_offset
                s = row.embedding_length or 0
                if current and size_of(current) + s > max_to_summarize:
                    spans.append(current)
                    current = []
                current.append(text[o:o + s])
                chunk_span.append(len(spans))
            spans.append(current)

            for group in spans:
                span_texts.append("".join(group))
            for row, gi in zip(rows, chunk_span):
                chunk_by_id[row.chunk_id] = base + gi

        return span_texts, chunk_by_id

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return await self.context.embedding(texts)

    def _deserialize_metadata(self, metadata_str: str | None) -> dict[str, Any]:
        if metadata_str is None:
            return {}
        if isinstance(metadata_str, dict):
            return metadata_str
        return deserialize_metadata(metadata_str)

async def main():
    parser = argparse.ArgumentParser(
        description="SQLAlchemy PostgreSQL Indexer CLI"
    )
    parser.add_argument("-c", "--config", action="append", default=[],
                        metavar="PATH", help="Config TOML file (repeatable; last wins)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Write per-document indexing statistics to {DEBUG_STATS_FILENAME}",
    )
    parser.add_argument("--index", action="store_true", help="Run chunk indexing")
    parser.add_argument("--maintenance", action="store_true",
                        help="Refresh changed and remove missing indexed documents")
    parser.add_argument(
        "--summarize-chunks", action="store_true", help="Run chunk summarization"
    )
    parser.add_argument(
        "--summarize-documents", action="store_true", help="Run document summarization"
    )
    args = parser.parse_args()
    operation_flags = (
        args.index,
        args.summarize_chunks,
        args.summarize_documents,
        args.maintenance,
    )
    if any(operation_flags):
        run_index = args.index
        run_summarize_chunks = args.summarize_chunks
        run_summarize_documents = args.summarize_documents
    else:
        run_index = True
        run_summarize_chunks = True
        run_summarize_documents = True

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.ERROR)

    ctx = Context.build_context(args.config)
    await ctx.init_db()
    indexer = Indexer(ctx, debug=args.debug)
    await indexer.index_all(
        index=run_index,
        summarize_chunks=run_summarize_chunks,
        summarize_documents=run_summarize_documents,
        maintenance=args.maintenance,
    )
    await indexer.wait_for_idle()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
