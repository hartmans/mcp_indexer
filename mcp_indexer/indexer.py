import argparse
import asyncio
import json
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from mcp_indexer.context import Context
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.models import ChunkSummary, Document, DocumentChunk, FailedDocument, deserialize_metadata
from mcp_indexer.plugins.base import DocumentSource, DocumentNotFoundError, create_embedding_chunks

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
        maintenance: bool = False,
    ):
        await self.context.build_collections()
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
        pointers = []
        async for pointer in source.get_documents(last_modified=None):
            pointers.append(pointer)
            if len(pointers) == 256:
                await self._index_discovered_documents(source, pointers)
                pointers = []
        if pointers:
            await self._index_discovered_documents(source, pointers)

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
            await self._schedule_document_task(
                self.index_document(pointer, stats),
                stats,
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
                    pointer = source.fetch_document(document_id)
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
                    await self._schedule_document_task(
                        self.index_document(pointer, stats, replace=True), stats,
                    )
            after = rows[-1][0]

    async def ensure_document_indexed(
        self, source: DocumentSource, document_id: str, *, retry_failed: bool = False,
    ) -> None:
        """Persist a local document ID if absent; never probe an existing document."""
        async with self.context.get_session() as session:
            if await session.get(Document, (source.id, document_id)) is not None:
                return
            failure = await session.get(FailedDocument, (source.id, document_id))
            if failure is not None and not retry_failed:
                raise RuntimeError(f"Previously failed document {document_id}: {failure.failure_reason}")
        try:
            await self.index_document(
                source.fetch_document(document_id),
                DocumentIndexingStats(document_id, "index", source=source),
            )
        except Exception as exc:
            await self._record_failure(source, document_id, str(exc))
            raise

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


            # Check which documents have an empty summary.
            needs_doc_summary: set[str] = set()
            doc_stmt = (
                select(Document.document_id)
                .where(
                    Document.collection_id == collection_id,
                    Document.document_id.in_(document_ids),
                    Document.summary == "",
                )
            )
            if failed_docs:
                doc_stmt = doc_stmt.where(Document.document_id.not_in(failed_docs))
            result = await session.execute(doc_stmt)
            needs_doc_summary = set(result.scalars().all())

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
        tasks = []

        # After chunk-level summaries are done, summarize the document itself.
        for document_id in needs_doc_summary:
            stats = DocumentIndexingStats(
                document_id=document_id, operation="document_summary", source=source
            )
            tasks.append(
                self._schedule_unbounded_document_task(
                self.summarize_document(source, document_id),
                stats,
            ))

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

    async def index_document(self, pointer, stats: DocumentIndexingStats, *, replace: bool = False):
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
            keywords=normalize_keywords(metadata_dict.get("keywords", [])),
        )
        # Attach chunks to document for cascade merge
        doc_record.chunks = all_chunks

        # Persist the complete chunk set together, replacing only on maintenance.
        await self._upsert_document(source, doc_record, replace=replace)

    async def summarize_document_chunks(
        self, source: DocumentSource, document_id: str
    ) -> bool:
        expected_mtime = await self._document_mtime(source, document_id)
        if expected_mtime is None:
            return False
        semantic_chunks = await self._reconstruct_semantic_chunks(source, document_id)

        if not semantic_chunks:
            return False

        summary_prompt = source.config.chunk_summary_prompt or source.config.doc_summary_prompt

        updated_rows: list[dict[str, Any]] = []
        summary_records: list[ChunkSummary] = []
        next_summary_span = 0
        all_spans: list[SummarySpan] = []

        for semantic_chunk in semantic_chunks:
            spans = self._split_summary_spans(
                semantic_chunk.text, self.server_config.max_to_summarize
            )
            all_spans.extend(spans)

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

        # Submit across semantic boundaries so the model batcher can stay full.
        summaries = await asyncio.gather(*(
            self._summarize_text(span.text, summary_prompt) for span in all_spans
        ))
        summary_records = [
            ChunkSummary(
                collection_id=source.id,
                document_id=document_id,
                summary_span=offset,
                summary=summary,
            )
            for offset, summary in enumerate(summaries)
        ]

        # Guard and write summaries/references together; refresh may have replaced
        # the document while the model was working.
        async with self.context.get_session() as session:
            current = (await session.execute(select(Document).where(
                Document.collection_id == source.id,
                Document.document_id == document_id,
            ).with_for_update())).scalar_one_or_none()
            if current is None or current.last_modified != expected_mtime:
                return False
            for summary in summary_records:
                await session.merge(summary)
            await session.flush()
            for row in updated_rows:
                await session.execute(update(DocumentChunk).where(
                    DocumentChunk.collection_id == source.id,
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.chunk_id == row["chunk_id"],
                ).values(summary_span=row["summary_span"]))
            await session.commit()
        return True

    async def _document_mtime(self, source: DocumentSource, document_id: str):
        async with self.context.get_session() as session:
            return (await session.execute(select(Document.last_modified).where(
                Document.collection_id == source.id,
                Document.document_id == document_id,
            ))).scalar_one_or_none()

    async def summarize_document(
        self, source: DocumentSource, document_id: str
    ) -> bool:
        async with self.context.get_session() as session:
            existing = (await session.execute(select(Document).where(
                Document.collection_id == source.id,
                Document.document_id == document_id,
            ))).scalar_one_or_none()
        if existing is None:
            return False
        if existing.summary:
            return True
        expected_mtime = existing.last_modified
        pointer = source.fetch_document(document_id)
        metadata_dict = await pointer.get_metadata()

        doc_summary = await source.get_document_summary(document_id)
        if not doc_summary:
            span_summaries = await self._load_summary_texts(source, document_id)
            if not span_summaries:
                return False
            doc_summary_text = "\n\n".join(span_summaries)
            doc_summary_list = await self.context.llm(
                [[
                    (
                        "system",
                        source.config.doc_summary_prompt,
                    ),
                    (
                        "user",
                        f"Write no more than two paragraphs to summarize the following document:\n\n{doc_summary_text}",
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

        doc_record = Document(
            collection_id=source.id,
            document_id=document_id,
            title=title,
            title_strength=title_strength,
            last_modified=expected_mtime,
            summary=final_summary,
            embedding=doc_vector,
            keywords=(extracted_keywords if extracted_keywords else
                      normalize_keywords(metadata_dict.get("keywords", []))),
        )

        return await self._update_document(source, doc_record)

    async def _upsert_document(
        self, source: DocumentSource, doc: Document, *, replace: bool = False,
    ) -> None:
        async with self.context.get_session() as session:
            # The FK cascades remove old chunks and summary spans on refresh.
            # Preparation has already succeeded before this transaction begins.
            if replace:
                await session.execute(delete(Document).where(
                    Document.collection_id == source.id,
                    Document.document_id == doc.document_id,
                ))
            values = {column.key: getattr(doc, column.key) for column in Document.__table__.columns}
            inserted = (await session.execute(insert(Document).values(**values)
                .on_conflict_do_nothing(index_elements=["collection_id", "document_id"])
                .returning(Document.document_id))).scalar_one_or_none()
            if inserted is None:
                # Another search/indexing task already prepared this document.
                return
            if doc.chunks:
                await session.execute(insert(DocumentChunk), [
                    {column.key: getattr(chunk, column.key) for column in DocumentChunk.__table__.columns}
                    for chunk in doc.chunks
                ])
            await session.execute(delete(FailedDocument).where(
                FailedDocument.collection_id == source.id,
                FailedDocument.document_id == doc.document_id,
            ))
            await session.commit()

    async def _update_document(
        self, source: DocumentSource, doc: Document
    ) -> bool:
        async with self.context.get_session() as session:
            current = (await session.execute(select(Document).where(
                Document.collection_id == source.id,
                Document.document_id == doc.document_id,
            ).with_for_update())).scalar_one_or_none()
            if current is None or current.last_modified != doc.last_modified:
                return False
            if current.summary:
                return True
            for attribute in ("summary", "embedding", "title", "title_strength", "keywords"):
                setattr(current, attribute, getattr(doc, attribute))
            await session.commit()
            return True

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
