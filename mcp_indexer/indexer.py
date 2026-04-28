import asyncio
from dataclasses import dataclass
import lancedb
import logging
import argparse
import sys
from typing import List, Optional, Any
from datetime import datetime
from mcp_indexer.context import Context
from mcp_indexer.config import ConfigManager
from mcp_indexer.plugins.base import Document, DocumentChunk, create_embedding_chunks

logger = logging.getLogger(__name__)

INDEXING_WORKERS = 64


@dataclass
class SummarySpan:
    offset: int
    size: int
    text: str


@dataclass
class SemanticChunkPlan:
    embedding_chunks: list[tuple[dict[str, Any], str]]
    summary_spans: list[SummarySpan]

class Indexer:
    """
    Orchestrates the indexing process across multiple collections.
    """
    def __init__(self, context: Context, config_manager: ConfigManager):
        self.context = context
        self.config_manager = config_manager
        self.sem = asyncio.Semaphore(INDEXING_WORKERS)
        self._active_tasks = 0
        self._idle = asyncio.Event()
        self._idle.set()

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
            self._active_tasks += 1
            self._idle.clear()
            task = asyncio.create_task(self._wrapped_process(
                pointer, col_config, table, meta_table,
                server_config.min_size, server_config.max_size
            ))
            task.add_done_callback(lambda fut: self._on_task_done(fut))

    async def _wrapped_process(self, pointer, col_config, table, meta_table, min_size, max_size):
        try:
            await self._process_document(
                pointer, col_config, table, meta_table, min_size, max_size
            )
        except Exception as e:
            logger.error(f"Error in _wrapped_process for {pointer.document_id}: {e}")
            raise e
        finally:
            self.sem.release()
            self._active_tasks -= 1
            if self._active_tasks == 0:
                self._idle.set()

    def _on_task_done(self, fut):
        try:
            fut.result()
        except Exception as e:
            logger.exception(f"Task failed: {e}")

    async def wait_for_idle(self):
        await self._idle.wait()

    async def _process_document(self, pointer, col_config, table, meta_table, min_size, max_size):
        try:
            server_config = self.config_manager.get_server_config()
            plans = [
                self._build_semantic_chunk_plan(meta, text_list, min_size, max_size, server_config.max_to_summarize)
                async for meta, text_list in pointer.get_chunks(min_size, max_size)
            ]

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('httpx').setLevel(logging.ERROR)
    
    ctx = Context.build_context(args.config)
    indexer = Indexer(ctx, ctx.config)
    await indexer.index_all()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
