import asyncio
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
import lancedb
import logging
import argparse
import sys
import time
from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel
from mcp_indexer.context import Context
from mcp_indexer.config import ConfigManager
from mcp_indexer.plugins.base import Document, DocumentChunk, create_embedding_chunks

logger = logging.getLogger(__name__)

INDEXING_WORKERS = 10
MONITOR_INTERVAL_SECONDS = 30.0
MONITOR_STALL_THRESHOLD_SECONDS = 40.0
DEBUG_STATS_FILENAME = "indexer-stats.jsonl"


def monitor_time() -> float:
    return time.monotonic()


@dataclass
class SummarySpan:
    offset: int
    size: int
    text: str


@dataclass
class SemanticChunkPlan:
    embedding_chunks: list[tuple[dict[str, Any], str]]
    summary_spans: list[SummarySpan]


class DocumentIndexingStats(BaseModel):
    document_id: str
    semantic_chunks: int = 0
    embedding_chunks: int = 0
    summary_spans: int = 0

class Indexer:
    """
    Orchestrates the indexing process across multiple collections.
    """
    def __init__(self, context: Context, config_manager: ConfigManager, debug: bool = False):
        self.context = context
        self.config_manager = config_manager
        self.debug = debug
        self.sem = asyncio.Semaphore(INDEXING_WORKERS)
        self._idle = asyncio.Event()
        self._idle.set()
        self._running_documents: dict[asyncio.Task, DocumentIndexingStats] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._debug_stats_path = Path(DEBUG_STATS_FILENAME)

    async def index_all(self):
        """
        Indexes all configured collections.
        """
        tasks = []
        for collection_id, source in self.context.collections.items():
            tasks.append(self.index_collection(collection_id, source))
        
        if tasks:
            await asyncio.gather(*tasks)

    async def index_collection(self, collection_id: str, source: Any):
        """
        Indexes a specific collection using the provided source plugin.
        """
        col_config = self.config_manager.get_collection_config(collection_id)
        server_config = self.config_manager.get_server_config()
        
        table = self._get_or_create_table(
            collection_id, 
            schema=DocumentChunk, 
        )
        meta_table = self._get_or_create_table(
            f"{collection_id}_meta", 
            schema=Document, 
        )

        existing_docs = set()
        try:
            existing_docs = set(meta_table.to_pandas()["document_id"].tolist())
        except Exception as e:
            logger.debug(f"Could not load existing documents for {collection_id}: {e}")
            pass

        async for pointer in source.get_documents(last_modified=None):
            if pointer.document_id in existing_docs:
                continue

            await self.sem.acquire()
            stats = DocumentIndexingStats(document_id=pointer.document_id)
            task = asyncio.create_task(self._wrapped_process(
                pointer, col_config, table, meta_table,
                server_config.min_size, server_config.max_size, stats
            ))
            self._track_task(task, stats)
            task.add_done_callback(self._on_task_done)

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
                        "Still indexing document_id=%s semantic_chunks=%d embedding_chunks=%d summary_spans=%d",
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

    async def _wrapped_process(self, pointer, col_config, table, meta_table, min_size, max_size, stats):
        try:
            await self._process_document(
                pointer, col_config, table, meta_table, min_size, max_size, stats
            )
        except Exception as e:
            logger.error(f"Error in _wrapped_process for {pointer.document_id}: {e}")
            raise e
        finally:
            self.sem.release()

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
        await self._idle.wait()
        if self._monitor_task is not None:
            with suppress(asyncio.CancelledError):
                await self._monitor_task

    async def _process_document(self, pointer, col_config, table, meta_table, min_size, max_size, stats: DocumentIndexingStats):
        try:
            server_config = self.config_manager.get_server_config()
            plans = []
            async for meta, text_list in pointer.get_chunks(min_size, max_size):
                plan = self._build_semantic_chunk_plan(
                    meta,
                    text_list,
                    min_size,
                    max_size,
                    server_config.max_to_summarize,
                )
                plans.append(plan)
                stats.semantic_chunks += 1
                stats.embedding_chunks += len(plan.embedding_chunks)
                stats.summary_spans += len(plan.summary_spans)

            summary_prompt = col_config.chunk_summary_prompt or col_config.doc_summary_prompt
            summary_tasks = [self._summarize_text(span.text, summary_prompt) for plan in plans for span in plan.summary_spans]
            embedding_tasks = [
                self._embed_batch([text for _, text in plan.embedding_chunks])
                for plan in plans
            ]

            summary_results, embeddings_batches = await asyncio.gather(
                asyncio.gather(*summary_tasks),
                asyncio.gather(*embedding_tasks),
            )

            chunk_summary_groups: list[list[str]] = []
            summary_index = 0
            for plan in plans:
                span_count = len(plan.summary_spans)
                span_summaries = summary_results[summary_index:summary_index + span_count]
                summary_index += span_count
                chunk_summary_groups.append(span_summaries)

            all_chunks_to_upsert = []
            for i, plan in enumerate(plans):
                batch_embeddings = embeddings_batches[i]
                span_summaries = chunk_summary_groups[i]

                for j, (e_meta, e_text) in enumerate(plan.embedding_chunks):
                    chunk_summary = self._summary_for_embedding_chunk(
                        e_meta,
                        plan.summary_spans,
                        span_summaries,
                    )
                    all_chunks_to_upsert.append(DocumentChunk(
                        document_id=pointer.document_id,
                        chunk_id=f"{pointer.document_id}?c={len(all_chunks_to_upsert)}",
                        # We do not need to store text for now
                        # text=e_text,
                        summary=chunk_summary,
                        embedding=batch_embeddings[j],
                        metadata=e_meta
                    ))

            metadata_dict = await pointer.get_metadata()
            doc_summary = metadata_dict.get("summary")
            if not doc_summary:
                doc_summary_text = "\n\n".join(summary_results)
                doc_summary_list = await self.context.llm([[
                    ('system', col_config.doc_summary_prompt),
                    ('user', f"Write no more than two paragraphs to summarize the following document:\n\n{doc_summary_text}")]
                ])
                doc_summary = doc_summary_list[0]
            
            doc_vector = (await self.context.embedding([doc_summary]))[0]

            doc_record = Document(
                document_id=pointer.document_id,
                title=metadata_dict.get("title", "Untitled"),
                last_modified=pointer.last_modified,
                summary=doc_summary,
                embedding=doc_vector,
                keywords=metadata_dict.get("keywords", [])
            )
            
            table.merge_insert(["document_id", "chunk_id"])                 .when_matched_update_all()                 .when_not_matched_insert_all()                 .when_not_matched_by_source_delete(f"document_id = '{pointer.document_id}'")                 .execute(all_chunks_to_upsert)
            
            meta_table.merge_insert(["document_id"]).when_matched_update_all().when_not_matched_insert_all().execute([doc_record])

        except Exception as e:
            logger.error(f"Error processing document {pointer.document_id}: {e}")
            raise e

    async def _summarize_text(self, text: str, prompt: str) -> str:
        res = await self.context.llm([[
                                       ('system',prompt),
                                       ('user',f'Write no more than three sentences to summarize this chunk:\n\n{text}'),
                                       ]])
        return res[0]

    def _build_semantic_chunk_plan(self, metadata: dict[str, Any], text_list: list[str], min_size: int, max_size: int, max_to_summarize: int) -> SemanticChunkPlan:
        text = "".join(text_list)
        return SemanticChunkPlan(
            embedding_chunks=create_embedding_chunks(metadata, text_list, min_size, max_size),
            summary_spans=self._split_summary_spans(text, max_to_summarize),
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

    def _summary_for_embedding_chunk(
        self,
        embedding_metadata: dict[str, Any],
        summary_spans: list[SummarySpan],
        span_summaries: list[str],
    ) -> str:
        if len(span_summaries) == 1:
            return span_summaries[0]

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

        return span_summaries[best_index]

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return await self.context.embedding(texts)

    def _get_or_create_table(self, name: str, schema: Any):
        try:
            return self.context.db.open_table(name)
        except Exception as e:
            logger.debug(f"Table {name} not found, creating it. Error: {e}")
            return self.context.db.create_table(
                name, 
                schema=schema, 
            )
    async def search(self, collection_id: str, query: str, limit: int = 5):
        query_vector = await self.context.embedding.query(query)
        table = self.context.db.open_table(collection_id)
        results = table.search(query_vector).limit(limit).to_pydantic(DocumentChunk)
        return results

async def main():
    parser = argparse.ArgumentParser(description="LanceDB Indexer CLI")
    parser.add_argument("--config", required=True, help="Path to the config TOML file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=f"Write per-document indexing statistics to {DEBUG_STATS_FILENAME}",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('httpx').setLevel(logging.ERROR)
    
    ctx = Context.build_context(args.config)
    indexer = Indexer(ctx, ctx.config, debug=args.debug)
    await indexer.index_all()
    await indexer.wait_for_idle()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
