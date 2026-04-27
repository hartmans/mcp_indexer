import asyncio
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

INDEXING_WORKERS = 20

class Indexer:
    """
    Orchestrates the indexing process across multiple collections.
    """
    def __init__(self, context: Context, config_manager: ConfigManager):
        self.context = context
        self.config_manager = config_manager
        self.sem = asyncio.Semaphore(INDEXING_WORKERS)

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
            
            async with self.sem:
                task = asyncio.create_task(self._wrapped_process(
                    pointer, col_config, table, meta_table, 
                    server_config.min_size, server_config.max_size
                ))
                task.add_done_callback(lambda fut: self._on_task_done(fut))

    async def _wrapped_process(self, pointer, col_config, table, meta_table, min_size, max_size):
        async with self.sem:
            try:
                await self._process_document(
                    pointer, col_config, table, meta_table, min_size, max_size
                )
            except Exception as e:
                logger.error(f"Error in _wrapped_process for {pointer.document_id}: {e}")
                raise e

    def _on_task_done(self, fut):
        try:
            fut.result()
        except Exception as e:
            logger.error(f"Task failed: {e}")

    async def _process_document(self, pointer, col_config, table, meta_table, min_size, max_size):
        try:
            semantic_chunks = []
            async for meta, text_list in pointer.get_chunks(min_size, max_size):
                semantic_chunks.append((meta, text_list))

            summary_tasks = [
                self._summarize_semantic_chunk(text_list, col_config.chunk_summary_prompt)
                for _, text_list in semantic_chunks
            ]
            
            embedding_tasks = []
            for meta, text_list in semantic_chunks:
                embedding_chunks = create_embedding_chunks(meta, text_list, min_size, max_size)
                texts = [t for _, t in embedding_chunks]
                embedding_tasks.append(self._embed_batch(texts))

            summaries = await asyncio.gather(*summary_tasks)
            embeddings_batches = await asyncio.gather(*embedding_tasks)

            all_chunks_to_upsert = []
            for i, (meta, _) in enumerate(semantic_chunks):
                sem_summary = summaries[i]
                batch_embeddings = embeddings_batches[i]
                emb_chunks_info = create_embedding_chunks(meta, semantic_chunks[i][1], min_size, max_size)
                
                for j, (e_meta, _) in enumerate(emb_chunks_info):
                    all_chunks_to_upsert.append(DocumentChunk(
                        document_id=pointer.document_id,
                        chunk_id=f"{pointer.document_id}?c={len(all_chunks_to_upsert)}",
                        text=emb_chunks_info[j][1],
                        summary=sem_summary,
                        embedding=batch_embeddings[j],
                        metadata=e_meta
                    ))

            metadata_dict = await pointer.get_metadata()
            doc_summary = metadata_dict.get("summary")
            if not doc_summary:
                doc_summary_text = "\n\n".join(summaries)
                doc_summary_list = await self.context.llm([
                    f"{col_config.doc_summary_prompt}\n\n{doc_summary_text}"
                ])
                doc_summary = doc_summary_list[0]
            
            doc_vector = await self.context.embedding.query(doc_summary)

            doc_record = Document(
                document_id=pointer.document_id,
                title=metadata_dict.get("title", "Untitled"),
                last_modified=pointer.last_modified,
                summary=doc_summary,
                embedding=doc_vector,
                keywords=metadata_dict.get("keywords", [])
            )
            
            table.merge_insert(["document_id", "chunk_id"])                 .when_matched_update_all()                 .when_not_matched_insert_all()                 .when_not_matched_by_source_delete(f"target.document_id = '{pointer.document_id}'")                 .execute(all_chunks_to_upsert)
            
            meta_table.upsert([doc_record])

        except Exception as e:
            logger.error(f"Error processing document {pointer.document_id}: {e}")
            raise e

    async def _summarize_semantic_chunk(self, text_list: list[str], prompt: str) -> str:
        text = " ".join(text_list)
        res = await self.context.llm([f"{prompt}\n\n{text}"])
        return res[0]

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
    
    ctx = Context.build_context(args.config)
    indexer = Indexer(ctx, ctx.config)
    await indexer.index_all()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
