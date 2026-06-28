import argparse
import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from sqlalchemy import String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from mcp_indexer.context import Context
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.models import ChunkSummary, Document, DocumentChunk, FailedDocument, deserialize_metadata
from mcp_indexer.plugins.base import DocumentSource, create_embedding_chunks

logger = logging.getLogger(__name__)

INDEXING_WORKERS = 64
MONITOR_INTERVAL_SECONDS = 30.0
MONITOR_STALL_THRESHOLD_SECONDS = 40.0
DEBUG_STATS_FILENAME = "indexer-stats.jsonl"


def monitor_time() -> float:
    import time

    return time.monotonic()


@dataclass
class SummarySpan:
    offset: int
    size: int
    text: str


@dataclass
class SemanticChunkPlan:
    embedding_chunks: list[tuple[dict[str, Any], str]]


@dataclass
class ReconstructedSemanticChunk:
    metadata: dict[str, Any]
    embedding_rows: list[dict[str, Any]]
    text: str


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
        self.sem = asyncio.Semaphore(INDEXING_WORKERS)
        self._idle = asyncio.Event()
        self._idle.set()
        self._running_documents: dict[asyncio.Task, DocumentIndexingStats] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._debug_stats_path = Path(DEBUG_STATS_FILENAME)

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
        """Record a failure for a document."""
        async with self.context.get_session() as session:
            failed = FailedDocument(
                collection_id=source.id,
                document_id=document_id,
                failure_reason=reason,
            )
            await session.merge(failed)
            await session.commit()

    async def index_all(
        self,
        *,
        index: bool = True,
        summarize_chunks: bool = True,
        summarize_documents: bool = True,
    ):
        await self.context.build_collections()
        loops = []
        for source in self.context.collections.values():
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
        # Get set of failed document_ids to skip
        failed_docs = await self._get_failed_documents(source)
        
        async with self.context.get_session() as session:
            # Get existing document IDs for this collection
            result = await session.execute(
                select(Document.document_id).where(Document.collection_id == source.id)
            )
            existing_docs = set(result.scalars().all())

        async for pointer in source.get_documents(last_modified=None):
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
            await self._schedule_document_task(
                self.index_document(pointer, stats),
                stats,
            )

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
            await self._schedule_document_task(
                self.summarize_document_chunks(source, document_id),
                stats,
            )

        await self.wait_for_idle()

    async def summarize_results(
        self, collection_id: str, document_ids: Iterable[str]
    ) -> None:
        document_ids = set(document_ids)
        if not document_ids:
            return

        source = self.context.collections[collection_id]
        failed_docs = await self._get_failed_documents(source)

        async with self.context.get_session() as session:
            stmt = (
                select(DocumentChunk.document_id)
                .where(
                    DocumentChunk.collection_id == collection_id,
                    DocumentChunk.document_id.in_(document_ids),
                    DocumentChunk.summary_span.is_(None),
                )
                .distinct()
                .order_by(DocumentChunk.document_id)
            )
            if failed_docs:
                stmt = stmt.where(DocumentChunk.document_id.not_in(failed_docs))
            result = await session.execute(stmt)
            pending_document_ids = result.scalars().all()

        if not pending_document_ids:
            return

        tasks = []
        for document_id in pending_document_ids:
            stats = DocumentIndexingStats(
                document_id=document_id, operation="chunk_summary", source=source
            )
            tasks.append(
                self._schedule_unbounded_document_task(
                    self.summarize_document_chunks(source, document_id),
                    stats,
                )
            )

        await asyncio.gather(*tasks)

    async def summarize_documents_collection(self, source: DocumentSource):
        # Get set of failed document_ids to skip
        failed_docs = await self._get_failed_documents(source)
        
        async with self.context.get_session() as session:
            result = await session.execute(
                select(Document).where(
                    Document.collection_id == source.id,
                    Document.summary == "",
                )
            )
            documents = result.scalars().all()

        if not documents:
            return

        # Filter out failed documents
        document_ids = sorted(d.document_id for d in documents if d.document_id not in failed_docs)
        for document_id in document_ids:
            stats = DocumentIndexingStats(
                document_id=document_id, operation="document_summary", source=source
            )
            await self._schedule_document_task(
                self.summarize_document(source, document_id),
                stats,
            )

        await self.wait_for_idle()

    async def _schedule_document_task(self, coro, stats: DocumentIndexingStats) -> None:
        await self.sem.acquire()
        task = asyncio.create_task(self._run_with_semaphore(coro, stats))
        self._track_task(task, stats)
        task.add_done_callback(self._on_task_done)

    def _schedule_unbounded_document_task(
        self, coro, stats: DocumentIndexingStats
    ) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._track_task(task, stats)
        task.add_done_callback(self._on_task_done)
        return task

    async def _run_with_semaphore(self, coro, stats: DocumentIndexingStats):
        try:
            return await coro
        except Exception as e:
            logger.error(f"Error in {stats.operation} for {stats.document_id}: {e}")
            raise
        finally:
            self.sem.release()

    def _track_task(self, task: asyncio.Task, stats: DocumentIndexingStats) -> None:
        self._running_documents[task] = stats
        self._idle.clear()
        self._ensure_monitor_task()

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
                running_items = list(self._running_documents.items())
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

                running_items = list(self._running_documents.items())
                if not running_items:
                    return

                current_tasks = {task for task, _ in running_items}
                tasks_to_log = current_tasks & previous_tasks
                previous_tasks = current_tasks

                for task, stats in running_items:
                    if task not in tasks_to_log:
                        continue
                    logger.info(
                        "Still running %s document_id=%s semantic_chunks=%d embedding_chunks=%d summary_spans=%d",
                        stats.operation,
                        stats.document_id,
                        stats.semantic_chunks,
                        stats.embedding_chunks,
                        stats.summary_spans,
                    )
        except asyncio.CancelledError:
            raise
        finally:
            if self._monitor_task is asyncio.current_task():
                self._monitor_task = None
            # Set idle if monitor task completes and there are no running documents
            if not self._running_documents:
                self._idle.set()

    def _on_task_done(self, fut: asyncio.Task) -> None:
        stats = self._running_documents.pop(fut, None)
        if stats is not None and self.debug:
            self._append_debug_stats(stats)

        if not self._running_documents:
            if self._monitor_task is not None and not self._monitor_task.done():
                self._monitor_task.cancel()
            self._idle.set()

        try:
            fut.result()
        except Exception as e:
            logger.exception(f"Task failed: {e}")
            # Record failure if we have stats and a document_id
            if stats is not None and hasattr(stats, "document_id") and stats.document_id:
                import traceback
                tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                # Truncate long tracebacks
                if len(tb_str) > 1000:
                    tb_str = tb_str[:997] + "..."
                # Get source from stats.source if available, otherwise try to infer
                source = getattr(stats, "source", None)
                if source is not None:
                    asyncio.create_task(
                        self._record_failure(source, stats.document_id, f"{type(e).__name__}: {str(e)}\n{tb_str}")
                    )
                else:
                    logger.warning(f"Cannot record failure for {stats.document_id}: no source available")

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

            if not self._running_documents:
                return

    async def index_document(self, pointer, stats: DocumentIndexingStats):
        source = pointer.source
        embedding_chunks: list[tuple[dict[str, Any], str]] = []
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
            embedding_chunks.extend(plan.embedding_chunks)

        batch_embeddings = (
            await self._embed_batch([text for _, text in embedding_chunks])
            if embedding_chunks
            else []
        )
        if len(batch_embeddings) != len(embedding_chunks):
            raise ValueError(
                f"Embedding batch returned {len(batch_embeddings)} vectors for {len(embedding_chunks)} chunks"
            )

        all_chunks = []
        for order, ((e_meta, _e_text), embedding) in enumerate(
            zip(embedding_chunks, batch_embeddings, strict=True)
        ):
            all_chunks.append(
                DocumentChunk(
                    collection_id=source.id,
                    document_id=pointer.document_id,
                    order=order,
                    chunk_id=f"{pointer.document_id}?c={order}",
                    embedding=embedding,
                    summary_span=None,
                    metadata_str=json.dumps(e_meta),
                )
            )

        metadata_dict = await pointer.get_metadata()
        doc_record = Document(
            collection_id=source.id,
            document_id=pointer.document_id,
            title=metadata_dict.get("title", "Untitled"),
            title_strength=int(metadata_dict.get("title_strength", 0)),
            last_modified=pointer.last_modified,
            summary="",
            embedding=[0.0] * VECTOR_DIMENSIONS,
            keywords=metadata_dict.get("keywords", []),
        )
        # Attach chunks to document for cascade merge
        doc_record.chunks = all_chunks

        # Upsert document (cascades to chunks via relationship)
        await self._upsert_document(source, doc_record)

    async def summarize_document_chunks(
        self, source: DocumentSource, document_id: str
    ) -> bool:
        semantic_chunks = await self._reconstruct_semantic_chunks(source, document_id)
        if not semantic_chunks:
            return False

        summary_prompt = source.config.chunk_summary_prompt or source.config.doc_summary_prompt

        updated_rows: list[dict[str, Any]] = []
        summary_records: list[ChunkSummary] = []
        next_summary_span = 0

        for semantic_chunk in semantic_chunks:
            spans = self._split_summary_spans(
                semantic_chunk.text, self.server_config.max_to_summarize
            )
            summaries = await asyncio.gather(
                *[
                    self._summarize_text(span.text, summary_prompt)
                    for span in spans
                ]
            )

            for offset, summary in enumerate(summaries):
                summary_records.append(
                    ChunkSummary(
                        collection_id=source.id,
                        document_id=document_id,
                        summary_span=next_summary_span + offset,
                        summary=summary,
                    )
                )

            for row in semantic_chunk.embedding_rows:
                metadata = self._deserialize_metadata(row.metadata_str)
                local_index = self._summary_span_for_embedding_chunk(metadata, spans)
                updated = {
                    "collection_id": row.collection_id,
                    "document_id": row.document_id,
                    "chunk_id": row.chunk_id,
                    "summary_span": next_summary_span + local_index,
                }
                updated_rows.append(updated)

            next_summary_span += len(spans)

        # Insert summaries first (chunks reference summaries via foreign key)
        await self._insert_summaries(source, summary_records)
        await self._update_chunks(source, updated_rows)
        return True

    async def summarize_document(
        self, source: DocumentSource, document_id: str
    ) -> bool:
        pointer = source.fetch_document(document_id)
        metadata_dict = await pointer.get_metadata()

        doc_summary = await source.get_document_summary(document_id)
        if not doc_summary:
            span_summaries = await self._load_summary_texts(source, document_id)
            if not span_summaries:
                return False
            doc_summary_text = "\n\n".join(span_summaries)
            doc_summary_list = await self.context.llm(
                [
                    (
                        "system",
                        source.config.doc_summary_prompt,
                    ),
                    (
                        "user",
                        f"Write no more than two paragraphs to summarize the following document:\n\n{doc_summary_text}",
                    ),
                ]
            )
            doc_summary = doc_summary_list[0]

        doc_vector = (await self.context.embedding([doc_summary]))[0]
        doc_record = Document(
            collection_id=source.id,
            document_id=document_id,
            title=metadata_dict.get("title", "Untitled"),
            title_strength=int(metadata_dict.get("title_strength", 0)),
            last_modified=pointer.last_modified,
            summary=doc_summary,
            embedding=doc_vector,
            keywords=metadata_dict.get("keywords", []),
        )

        await self._update_document(source, doc_record)
        return True

    async def _upsert_document(
        self, source: DocumentSource, doc: Document
    ) -> None:
        async with self.context.get_session() as session:
            await session.merge(doc)
            await session.commit()

    async def _insert_summaries(
        self, source: DocumentSource, summaries: list[ChunkSummary]
    ) -> None:
        if not summaries:
            return
        async with self.context.get_session() as session:
            for summary in summaries:
                await session.merge(summary)
            await session.commit()

    async def _update_chunks(
        self, source: DocumentSource, updates: list[dict[str, Any]]
    ) -> None:
        if not updates:
            return
        async with self.context.get_session() as session:
            for row in updates:
                stmt = select(DocumentChunk).where(
                    DocumentChunk.collection_id == source.id,
                    DocumentChunk.document_id == row["document_id"],
                    DocumentChunk.chunk_id == row["chunk_id"],
                )
                result = await session.execute(stmt)
                chunk = result.scalar_one_or_none()
                if chunk:
                    chunk.summary_span = row["summary_span"]
            await session.commit()

    async def _update_document(
        self, source: DocumentSource, doc: Document
    ) -> None:
        async with self.context.get_session() as session:
            await session.merge(doc)
            await session.commit()

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

        grouped_rows: dict[str, list[DocumentChunk]] = {}
        grouped_metadata: dict[str, dict[str, Any]] = {}
        for row in document_rows:
            embedding_metadata = self._deserialize_metadata(row.metadata_str)
            semantic_metadata = {
                key: value
                for key, value in embedding_metadata.items()
                if key not in ("o", "s")
            }
            key = self._metadata_identity(semantic_metadata)
            grouped_rows.setdefault(key, []).append(row)
            grouped_metadata.setdefault(key, semantic_metadata)

        semantic_chunks = []
        for key, rows_for_semantic_chunk in grouped_rows.items():
            text = await source.fetch_chunk(
                document_id, grouped_metadata[key], scope="semantic"
            )
            semantic_chunks.append(
                ReconstructedSemanticChunk(
                    metadata=grouped_metadata[key],
                    embedding_rows=rows_for_semantic_chunk,
                    text=text,
                )
            )
        return semantic_chunks

    async def _summarize_text(self, text: str, prompt: str) -> str:
        res = await self.context.llm(
            [
                ("system", prompt),
                ("user", f"Write no more than three sentences to summarize this chunk:\n\n{text}"),
            ]
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
                metadata, text_list, min_size, max_size
            ),
        )

    def _split_summary_spans(
        self, text: str, max_to_summarize: int
    ) -> list[SummarySpan]:
        if len(text) <= max_to_summarize:
            return [SummarySpan(offset=0, size=len(text), text=text)]

        split_size = max(1, max_to_summarize // 2)
        spans = []
        for offset in range(0, len(text), split_size):
            chunk_text = text[offset : offset + split_size]
            spans.append(SummarySpan(offset=offset, size=len(chunk_text), text=chunk_text))
        return spans

    def _summary_span_for_embedding_chunk(
        self,
        embedding_metadata: dict[str, Any],
        summary_spans: list[SummarySpan],
    ) -> int:
        if len(summary_spans) == 1:
            return 0

        start = int(embedding_metadata.get("o", 0))
        end = start + int(embedding_metadata.get("s", 0))
        best_index = 0
        best_overlap = -1

        for index, span in enumerate(summary_spans):
            span_end = span.offset + span.size
            overlap = max(0, min(end, span_end) - max(start, span.offset))
            if overlap > best_overlap:
                best_index = index
                best_overlap = overlap

        return best_index

    async def _load_summary_texts(
        self, source: DocumentSource, document_id: str
    ) -> list[str]:
        async with self.context.get_session() as session:
            result = await session.execute(
                select(ChunkSummary).where(
                    ChunkSummary.collection_id == source.id,
                    ChunkSummary.document_id == document_id,
                )
            )
            summaries = result.scalars().all()
            if not summaries:
                return []

            sorted_summaries = sorted(summaries, key=lambda s: s.summary_span)
            return [s.summary for s in sorted_summaries if s.summary]

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return await self.context.embedding(texts)

    def _metadata_identity(self, metadata: dict[str, Any]) -> str:
        return json.dumps(metadata, sort_keys=True, separators=(",", ":"))

    def _deserialize_metadata(self, metadata_str: str | None) -> dict[str, Any]:
        if metadata_str is None:
            return {}
        if isinstance(metadata_str, dict):
            return metadata_str
        return deserialize_metadata(metadata_str)

    async def search(
        self,
        collection_id: str,
        query: str,
        limit: int = 5,
        document_candidate_limit: int | None = None,
        chunk_candidate_limit: int | None = None,
    ):
        query_vector = await self.context.embedding.query(query)
        document_candidate_limit = document_candidate_limit or limit
        chunk_candidate_limit = chunk_candidate_limit or limit

        document_distance = Document.embedding.cosine_distance(query_vector)
        chunk_distance = DocumentChunk.embedding.cosine_distance(query_vector)

        document_candidates = (
            select(
                Document.collection_id.label("collection_id"),
                Document.document_id.label("document_id"),
                func.cast(None, String).label("chunk_id"),
                document_distance.label("distance"),
            )
            .where(Document.collection_id == collection_id)
            .order_by(document_distance, Document.document_id)
            .limit(document_candidate_limit)
            .cte("document_candidates")
        )
        chunk_candidates = (
            select(
                DocumentChunk.collection_id.label("collection_id"),
                DocumentChunk.document_id.label("document_id"),
                DocumentChunk.chunk_id.label("chunk_id"),
                chunk_distance.label("distance"),
            )
            .where(DocumentChunk.collection_id == collection_id)
            .order_by(chunk_distance, DocumentChunk.document_id, DocumentChunk.chunk_id)
            .limit(chunk_candidate_limit)
            .cte("chunk_candidates")
        )
        candidate_hits = (
            select(
                document_candidates.c.collection_id,
                document_candidates.c.document_id,
                document_candidates.c.chunk_id,
                document_candidates.c.distance,
            )
            .union_all(
                select(
                    chunk_candidates.c.collection_id,
                    chunk_candidates.c.document_id,
                    chunk_candidates.c.chunk_id,
                    chunk_candidates.c.distance,
                )
            )
            .cte("candidate_hits")
        )
        ranked_hits = (
            select(
                candidate_hits.c.collection_id,
                candidate_hits.c.document_id,
                candidate_hits.c.chunk_id,
                candidate_hits.c.distance,
                func.row_number()
                .over(
                    order_by=(
                        candidate_hits.c.distance,
                        candidate_hits.c.document_id,
                        candidate_hits.c.chunk_id.asc().nullsfirst(),
                    )
                )
                .label("hit_rank"),
            )
            .cte("ranked_hits")
        )
        top_hits = (
            select(
                ranked_hits.c.collection_id,
                ranked_hits.c.document_id,
                ranked_hits.c.chunk_id,
                ranked_hits.c.distance,
                ranked_hits.c.hit_rank,
            )
            .where(ranked_hits.c.hit_rank <= limit)
            .cte("top_hits")
        )
        grouped_hits = (
            select(
                top_hits.c.collection_id,
                top_hits.c.document_id,
                top_hits.c.chunk_id,
                top_hits.c.distance,
                top_hits.c.hit_rank,
                func.min(top_hits.c.hit_rank)
                .over(
                    partition_by=(
                        top_hits.c.collection_id,
                        top_hits.c.document_id,
                    )
                )
                .label("document_group_rank"),
            )
            .cte("grouped_hits")
        )

        async with self.context.get_session() as session:
            result = await session.execute(
                select(
                    grouped_hits.c.collection_id,
                    grouped_hits.c.document_id,
                    grouped_hits.c.chunk_id,
                )
                .order_by(
                    grouped_hits.c.document_group_rank,
                    grouped_hits.c.hit_rank,
                )
            )
            ranked_hits = result.mappings().all()
            if not ranked_hits:
                return []

            document_ids = {row["document_id"] for row in ranked_hits}
            await self.summarize_results(collection_id, document_ids)
            document_result = await session.execute(
                select(Document)
                .where(
                    Document.collection_id == collection_id,
                    Document.document_id.in_(document_ids),
                )
                .options(
                    selectinload(Document.chunks).selectinload(DocumentChunk.summary)
                )
            )
            documents = {
                document.document_id: document
                for document in document_result.scalars().unique().all()
            }

            grouped_documents: list[tuple[str, list[DocumentChunk]]] = []
            document_chunks: dict[str, list[DocumentChunk]] = {}
            for row in ranked_hits:
                document_id = row["document_id"]
                chunks = document_chunks.get(document_id)
                if chunks is None:
                    chunks = []
                    document_chunks[document_id] = chunks
                    grouped_documents.append((document_id, chunks))

                chunk_id = row["chunk_id"]
                if chunk_id is None:
                    continue

                document = documents[document_id]
                for chunk in document.chunks:
                    if chunk.chunk_id == chunk_id:
                        chunks.append(chunk)
                        break
                else:
                    raise LookupError(
                        f"Missing hydrated chunk {chunk_id!r} for document {document_id!r}"
                    )

            return [
                (documents[document_id], chunks)
                for document_id, chunks in grouped_documents
            ]


async def main():
    parser = argparse.ArgumentParser(
        description="SQLAlchemy PostgreSQL Indexer CLI"
    )
    parser.add_argument("--config", required=True, help="Path to the config TOML file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Write per-document indexing statistics to {DEBUG_STATS_FILENAME}",
    )
    parser.add_argument("--index", action="store_true", help="Run chunk indexing")
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
    indexer = Indexer(ctx, debug=args.debug)
    await indexer.index_all(
        index=run_index,
        summarize_chunks=run_summarize_chunks,
        summarize_documents=run_summarize_documents,
    )
    await indexer.wait_for_idle()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
