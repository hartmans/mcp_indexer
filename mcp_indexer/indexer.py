import argparse
import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from lancedb import col, lit
from pydantic import BaseModel

from mcp_indexer.context import Context
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.plugins.base import ChunkSummary, Document, DocumentChunk, DocumentSource, create_embedding_chunks

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


class DocumentIndexingStats(BaseModel):
    document_id: str
    operation: str = "index"
    semantic_chunks: int = 0
    embedding_chunks: int = 0
    summary_spans: int = 0


class Indexer:
    """
    Orchestrates indexing, chunk summarization, and document summarization.
    """

    def __init__(self, context: Context, debug: bool = False):
        self.context = context
        self._require_built_collections()
        self.server_config = context.config.get_server_config()
        self.debug = debug
        self.sem = asyncio.Semaphore(INDEXING_WORKERS)
        self._idle = asyncio.Event()
        self._idle.set()
        self._running_documents: dict[asyncio.Task, DocumentIndexingStats] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._debug_stats_path = Path(DEBUG_STATS_FILENAME)
        self.document_upsirt: list[tuple[DocumentSource, Document]] = []
        self.chunk_upsirt: list[tuple[DocumentSource, list[DocumentChunk]]] = []
        self.summary_upsirt: list[tuple[DocumentSource, list[ChunkSummary]]] = []
        self._upsirt_task: Optional[asyncio.Task] = None
        self._upsirt_lock = asyncio.Lock()

    def _require_built_collections(self) -> None:
        if not self.context.collections:
            raise ValueError("Context has no collections; call Context.build_collections() before creating an Indexer.")

        for collection_id, source in self.context.collections.items():
            missing = [
                name for name in ("chunk_table", "meta_table", "summary_table")
                if getattr(source, name, None) is None
            ]
            if missing:
                missing_names = ", ".join(missing)
                raise ValueError(
                    f"Collection '{collection_id}' has no built tables ({missing_names}); "
                    "call Context.build_collections() before creating an Indexer."
                )

    async def index_all(
        self,
        *,
        index: bool = True,
        summarize_chunks: bool = True,
        summarize_documents: bool = True,
    ):
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
        if self._upsirt_task is not None:
            await self._upsirt_task
        await self.wait_for_idle()

    async def index_collection(self, source: DocumentSource):
        meta_table = source.meta_table
        meta_table.optimize()
        source.chunk_table.optimize()
        source.summary_table.optimize()

        existing_docs = set()
        try:
            existing_docs = set(meta_table.to_pandas()["document_id"].tolist())
        except Exception as e:
            logger.debug(f"Could not load existing documents for {source.id}: {e}")

        async for pointer in source.get_documents(last_modified=None):
            if pointer.document_id in existing_docs:
                continue

            stats = DocumentIndexingStats(document_id=pointer.document_id, operation="index")
            await self._schedule_document_task(
                self.index_document(pointer, stats),
                stats,
            )

    async def summarize_chunks_collection(self, source: DocumentSource):
        while True:
            try:
                chunks = source.chunk_table.to_pandas()
            except Exception as e:
                logger.debug(f"Could not load chunks for {source.id}: {e}")
                return

            if "summary_span" not in chunks.columns or chunks.empty:
                if not self._running_documents:
                    return
                await self.wait_for_idle()
                continue

            unsummarized = chunks[chunks["summary_span"].isna()]
            document_ids = sorted(unsummarized["document_id"].dropna().unique())
            if not document_ids:
                if not self._running_documents:
                    return
                await self.wait_for_idle()
                continue

            for document_id in document_ids:
                stats = DocumentIndexingStats(document_id=document_id, operation="chunk_summary")
                await self._schedule_document_task(
                    self.summarize_document_chunks(source, document_id),
                    stats,
                )

            await self.wait_for_idle()
            return

    async def summarize_documents_collection(self, source: DocumentSource):
        while True:
            try:
                documents = source.meta_table.to_pandas()
            except Exception as e:
                logger.debug(f"Could not load document metadata for {source.id}: {e}")
                return

            if "summary" not in documents.columns or documents.empty:
                if not self._running_documents:
                    return
                await self.wait_for_idle()
                continue

            candidates = documents[documents["summary"].fillna("").astype(str) == ""]
            document_ids = sorted(candidates["document_id"].dropna().unique())
            if not document_ids:
                if not self._running_documents:
                    return
                await self.wait_for_idle()
                continue

            for document_id in document_ids:
                stats = DocumentIndexingStats(document_id=document_id, operation="document_summary")
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

    async def _run_with_semaphore(self, coro, stats: DocumentIndexingStats):
        try:
            return await coro
        except Exception as e:
            logger.error(f"Error in {stats.operation} for {stats.document_id}: {e}")
            raise
        finally:
            await self.run_upsirts()
            self.sem.release()

    def _track_task(self, task: asyncio.Task, stats: DocumentIndexingStats) -> None:
        self._running_documents[task] = stats
        self._idle.clear()
        self._ensure_monitor_task()

    def _ensure_monitor_task(self) -> None:
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_running_documents())

    async def _monitor_running_documents(self) -> None:
        last_run = monitor_time()
        previous_tasks: set[asyncio.Task] = set()
        try:
            while True:
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

    def _append_debug_stats(self, stats: DocumentIndexingStats) -> None:
        with self._debug_stats_path.open("a", encoding="utf-8") as handle:
            handle.write(stats.model_dump_json())
            handle.write("\n")
            handle.flush()

    async def wait_for_idle(self):
        while True:
            await self._idle.wait()

            if self._monitor_task is not None:
                with suppress(asyncio.CancelledError):
                    await self._monitor_task

            if self._upsirt_task is not None:
                await self._upsirt_task

            if not self._running_documents and (self._upsirt_task is None or self._upsirt_task.done()):
                return

    async def index_document(self, pointer, stats: DocumentIndexingStats):
        source = pointer.source
        embedding_chunks: list[tuple[dict[str, Any], str]] = []
        async for meta, text_list in pointer.get_chunks(self.server_config.min_size, self.server_config.max_size):
            plan = self._build_semantic_chunk_plan(
                meta,
                text_list,
                self.server_config.min_size,
                self.server_config.max_size,
            )
            stats.semantic_chunks += 1
            stats.embedding_chunks += len(plan.embedding_chunks)
            embedding_chunks.extend(plan.embedding_chunks)

        batch_embeddings = await self._embed_batch([text for _, text in embedding_chunks]) if embedding_chunks else []
        if len(batch_embeddings) != len(embedding_chunks):
            raise ValueError(
                f"Embedding batch returned {len(batch_embeddings)} vectors for {len(embedding_chunks)} chunks"
            )

        all_chunks_to_upsert = []
        for order, ((e_meta, _e_text), embedding) in enumerate(zip(embedding_chunks, batch_embeddings, strict=True)):
            all_chunks_to_upsert.append(DocumentChunk(
                document_id=pointer.document_id,
                order=order,
                chunk_id=f"{pointer.document_id}?c={order}",
                embedding=embedding,
                summary_span=None,
                metadata=e_meta,
            ))

        metadata_dict = await pointer.get_metadata()
        doc_record = Document(
            document_id=pointer.document_id,
            title=metadata_dict.get("title", "Untitled"),
            title_strength=int(metadata_dict.get("title_strength", 0)),
            last_modified=pointer.last_modified,
            summary="",
            embedding=[0.0] * VECTOR_DIMENSIONS,
            keywords=metadata_dict.get("keywords", []),
        )

        self.chunk_upsirt.append((source, all_chunks_to_upsert))
        self.document_upsirt.append((source, doc_record))

    async def summarize_document_chunks(self, source: DocumentSource, document_id: str) -> bool:
        semantic_chunks = await self._reconstruct_semantic_chunks(source, document_id)
        if not semantic_chunks:
            return False

        summary_prompt = source.config.chunk_summary_prompt or source.config.doc_summary_prompt

        updated_rows: list[dict[str, Any]] = []
        summary_records: list[ChunkSummary] = []
        next_summary_span = 0

        for semantic_chunk in semantic_chunks:
            spans = self._split_summary_spans(semantic_chunk.text, self.server_config.max_to_summarize)
            summaries = await asyncio.gather(*[
                self._summarize_text(span.text, summary_prompt)
                for span in spans
            ])

            for offset, summary in enumerate(summaries):
                summary_records.append(ChunkSummary(
                    document_id=document_id,
                    summary_span=next_summary_span + offset,
                    summary=summary,
                ))

            for row in semantic_chunk.embedding_rows:
                metadata = self._deserialize_metadata(row["metadata_str"])
                local_index = self._summary_span_for_embedding_chunk(metadata, spans)
                updated = dict(row)
                updated["summary_span"] = next_summary_span + local_index
                updated_rows.append(updated)

            next_summary_span += len(spans)

        chunk_records = [DocumentChunk.model_validate(row) for row in updated_rows]
        self.chunk_upsirt.append((source, chunk_records))
        self.summary_upsirt.append((source, summary_records))
        return True

    async def summarize_document(self, source: DocumentSource, document_id: str) -> bool:
        pointer = source.fetch_document(document_id)
        metadata_dict = await pointer.get_metadata()

        doc_summary = await source.get_document_summary(document_id)
        if not doc_summary:
            span_summaries = self._load_summary_texts(source, document_id)
            if not span_summaries:
                return False
            doc_summary_text = "\n\n".join(span_summaries)
            doc_summary_list = await self.context.llm([[
                ("system", source.config.doc_summary_prompt),
                ("user", f"Write no more than two paragraphs to summarize the following document:\n\n{doc_summary_text}"),
            ]])
            doc_summary = doc_summary_list[0]

        doc_vector = (await self.context.embedding([doc_summary]))[0]
        doc_record = Document(
            document_id=document_id,
            title=metadata_dict.get("title", "Untitled"),
            title_strength=int(metadata_dict.get("title_strength", 0)),
            last_modified=pointer.last_modified,
            summary=doc_summary,
            embedding=doc_vector,
            keywords=metadata_dict.get("keywords", []),
        )

        self.document_upsirt.append((source, doc_record))
        return True

    async def run_upsirts(self) -> Optional[asyncio.Task]:
        async with self._upsirt_lock:
            if self._upsirt_task is None or self._upsirt_task.done():
                if not (self.document_upsirt or self.chunk_upsirt or self.summary_upsirt):
                    self._upsirt_task = None
                    return None
                self._upsirt_task = asyncio.create_task(self._run_upsirts())
            return self._upsirt_task

    async def _run_upsirts(self) -> None:
        while True:
            async with self._upsirt_lock:
                document_work = self.document_upsirt[:]
                chunk_work = self.chunk_upsirt[:]
                summary_work = self.summary_upsirt[:]
                self.document_upsirt = []
                self.chunk_upsirt = []
                self.summary_upsirt = []

            if not document_work and not chunk_work and not summary_work:
                async with self._upsirt_lock:
                    if not self.document_upsirt and not self.chunk_upsirt and not self.summary_upsirt:
                        self._upsirt_task = None
                        return
                continue

            await self._execute_chunk_upsirts(chunk_work)
            await self._execute_summary_upsirts(summary_work)
            await self._execute_document_upsirts(document_work)

    async def _execute_chunk_upsirts(
        self,
        work: list[tuple[DocumentSource, list[DocumentChunk]]],
    ) -> None:
        grouped_chunks: dict[int, tuple[DocumentSource, list[DocumentChunk]]] = {}
        summary_deletes: dict[int, tuple[DocumentSource, set[str]]] = {}

        for source, chunks in work:
            if not chunks:
                continue

            source_key = id(source)
            grouped_chunks.setdefault(source_key, (source, []))[1].extend(chunks)
            summary_deletes.setdefault(source_key, (source, set()))[1].add(
                self._document_id_for_chunk_upsirt(chunks)
            )

        for source, document_ids in summary_deletes.values():
            for document_id in sorted(document_ids):
                await asyncio.to_thread(source.summary_table.delete, self._document_filter(document_id))

        for source, chunks in grouped_chunks.values():
            if not chunks:
                continue

            document_filters = [
                self._document_filter(document_id)
                for document_id in sorted({chunk.document_id for chunk in chunks})
            ]
            delete_filter = " OR ".join(f"({doc_filter})" for doc_filter in document_filters)

            await asyncio.to_thread(
                lambda source=source, chunks=chunks, delete_filter=delete_filter: source.chunk_table.merge_insert(
                    ["document_id", "chunk_id"]
                )
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(delete_filter)
                .execute(chunks)
            )

    async def _execute_summary_upsirts(
        self,
        work: list[tuple[DocumentSource, list[ChunkSummary]]],
    ) -> None:
        grouped_summaries: dict[int, tuple[DocumentSource, list[ChunkSummary]]] = {}

        for source, summaries in work:
            if not summaries:
                continue
            grouped_summaries.setdefault(id(source), (source, []))[1].extend(summaries)

        for source, summaries in grouped_summaries.values():
            await asyncio.to_thread(source.summary_table.add, summaries)

    async def _execute_document_upsirts(
        self,
        work: list[tuple[DocumentSource, Document]],
    ) -> None:
        grouped_documents: dict[int, tuple[DocumentSource, list[Document]]] = {}

        for source, document in work:
            grouped_documents.setdefault(id(source), (source, []))[1].append(document)

        for source, documents in grouped_documents.values():
            await asyncio.to_thread(
                lambda source=source, documents=documents: source.meta_table.merge_insert(["document_id"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(documents)
            )

    async def _reconstruct_semantic_chunks(self, source: DocumentSource, document_id: str) -> list[ReconstructedSemanticChunk]:
        document_rows = source.chunk_table.search().where(col('document_id')==lit(document_id)).to_pandas().to_dict("records")

        document_rows.sort(key=lambda row: int(row["order"]))

        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        grouped_metadata: dict[str, dict[str, Any]] = {}
        for row in document_rows:
            embedding_metadata = self._deserialize_metadata(row["metadata_str"])
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
            text = await source.fetch_chunk(document_id, grouped_metadata[key], scope="semantic")
            semantic_chunks.append(ReconstructedSemanticChunk(
                metadata=grouped_metadata[key],
                embedding_rows=rows_for_semantic_chunk,
                text=text,
            ))
        return semantic_chunks

    async def _summarize_text(self, text: str, prompt: str) -> str:
        res = await self.context.llm([[
            ("system", prompt),
            ("user", f"Write no more than three sentences to summarize this chunk:\n\n{text}"),
        ]])
        return res[0]

    def _build_semantic_chunk_plan(
        self,
        metadata: dict[str, Any],
        text_list: list[str],
        min_size: int,
        max_size: int,
    ) -> SemanticChunkPlan:
        return SemanticChunkPlan(
            embedding_chunks=create_embedding_chunks(metadata, text_list, min_size, max_size),
        )

    def _split_summary_spans(self, text: str, max_to_summarize: int) -> list[SummarySpan]:
        if len(text) <= max_to_summarize:
            return [SummarySpan(offset=0, size=len(text), text=text)]

        split_size = max(1, max_to_summarize // 2)
        spans = []
        for offset in range(0, len(text), split_size):
            chunk_text = text[offset:offset + split_size]
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

    def _load_summary_texts(self, source: DocumentSource, document_id: str) -> list[str]:
        summaries = source.summary_table.to_pandas()
        if summaries.empty:
            return []

        rows = summaries[summaries["document_id"] == document_id].sort_values("summary_span")
        return rows["summary"].dropna().astype(str).tolist()

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return await self.context.embedding(texts)

    def _document_filter(self, document_id: str) -> str:
        return (col("document_id") == lit(document_id)).to_sql()

    def _document_id_for_chunk_upsirt(self, chunks: list[DocumentChunk]) -> str:
        # Each queued chunk upsert unit is assumed to belong to a single document.
        # The caller is responsible for preserving that invariant.
        return chunks[0].document_id

    def _metadata_identity(self, metadata: dict[str, Any]) -> str:
        return json.dumps(metadata, sort_keys=True, separators=(",", ":"))

    def _deserialize_metadata(self, metadata_str: str | None) -> dict[str, Any]:
        return DocumentChunk._deserialize_metadata(metadata_str)

    async def search(self, collection_id: str, query: str, limit: int = 5):
        query_vector = await self.context.embedding.query(query)
        source = self.context.collections[collection_id]
        results = source.chunk_table.search(query_vector).limit(limit).to_pydantic(DocumentChunk)
        return results


async def main():
    parser = argparse.ArgumentParser(description="LanceDB Indexer CLI")
    parser.add_argument("--config", required=True, help="Path to the config TOML file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Write per-document indexing statistics to {DEBUG_STATS_FILENAME}",
    )
    parser.add_argument("--index", action="store_true", help="Run chunk indexing")
    parser.add_argument("--summarize-chunks", action="store_true", help="Run chunk summarization")
    parser.add_argument("--summarize-documents", action="store_true", help="Run document summarization")
    args = parser.parse_args()
    operation_flags = (args.index, args.summarize_chunks, args.summarize_documents)
    if any(operation_flags):
        run_index = args.index
        run_summarize_chunks = args.summarize_chunks
        run_summarize_documents = args.summarize_documents
    else:
        run_index = True
        run_summarize_chunks = True
        run_summarize_documents = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
