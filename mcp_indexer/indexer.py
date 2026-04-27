import asyncio
import lancedb
from typing import List, Optional, Any
from datetime import datetime
from mcp_indexer.context import Context
from mcp_indexer.config import ConfigManager
from mcp_indexer.plugins.base import Document, DocumentChunk, create_embedding_chunks

INDEXING_WORKERS = 20

class Indexer:
    """
    Orchestrates the indexing process across multiple collections.
    """
    def __init__(self, context: Context, config_manager: ConfigManager):
        self.context = context
        self.config_manager = config_manager
        self.sem = asyncio.Semaphore(INDEXING_WORKERS)

    async def index_collection(self, collection_id: str, source_plugin_class: Any):
        """
        Indexes a specific collection using the provided plugin class.
        """
        col_config = self.config_manager.get_collection_config(collection_id)
        server_config = self.config_manager.get_server_config()
        
        source = source_plugin_class(
            collection_id=collection_id,
            context=self.context,
            collection_config=col_config
        )
        
        table = self._get_or_create_table(
            collection_id, 
            schema=DocumentChunk, 
            primary_key="chunk_id"
        )
        meta_table = self._get_or_create_table(
            f"{collection_id}_meta", 
            schema=Document, 
            primary_key="document_id"
        )

        existing_docs = set()
        try:
            existing_docs = set(meta_table.to_pandas()["document_id"].tolist())
        except:
            pass

        async for pointer in source.get_documents(last_modified=None):
            if pointer.document_id in existing_docs:
                continue
            
            await self.sem.acquire()
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
        finally:
            self.sem.release()

    def _on_task_done(self, fut):
        try:
            fut.result()
        except Exception as e:
            # In a real system, we might log this to a file or monitoring service
            print(f"Task failed: {e}")

    async def _process_document(self, pointer, col_config, table, meta_table, min_size, max_size):
        """
        Processes a single document: generates summary, embedding, and chunks.
        """
        try:
            # 1. Extract all semantic chunks first
            semantic_chunks = []
            async for meta, text_list in pointer.get_chunks(min_size, max_size):
                semantic_chunks.append((meta, text_list))

            # 2. Parallelize: Semantic summaries vs Embedding calls
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

            # 3. Construct DocumentChunks
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

            # 4. Handle Document Summary
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
            
            # Atomic merge
            table.merge_insert(["document_id", "chunk_id"])                 .when_matched_update_all()                 .when_not_matched_insert_all()                 .when_not_matched_by_source_delete(f"target.document_id = '{pointer.document_id}'")                 .execute(all_chunks_to_upsert)
            
            meta_table.upsert([doc_record])

        except Exception as e:
            print(f"Error processing document {pointer.document_id}: {e}")
            raise e

    async def _summarize_semantic_chunk(self, text_list: list[str], prompt: str) -> str:
        text = " ".join(text_list)
        res = await self.context.llm([f"{prompt}\n\n{text}"])
        return res[0]

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return await self.context.embedding(texts)

    def _get_or_create_table(self, name: str, schema: Any, primary_key: Any):
        try:
            return self.context.db.open_table(name)
        except:
            return self.context.db.create_table(
                name, 
                schema=schema, 
                primary_key=primary_key
            )

    async def search(self, collection_id: str, query: str, limit: int = 5):
        query_vector = await self.context.embedding.query(query)
        table = self.context.db.open_table(collection_id)
        results = table.search(query_vector).limit(limit).to_pydantic(DocumentChunk)
        return results
