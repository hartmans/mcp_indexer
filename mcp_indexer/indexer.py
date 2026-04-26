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

    async def index_collection(self, collection_id: str, source_plugin_class: Any):
        """
        Indexes a specific collection using the provided plugin class.
        """
        # 1. Resolve configuration
        col_config = self.config_manager.get_collection_config(collection_id)
        server_config = self.config_manager.get_server_config()
        
        # 2. Initialize the plugin
        source = source_plugin_class(
            collection_id=collection_id,
            context=self.context,
            collection_config=col_config
        )
        
        # 3. Ensure the LanceDB tables exist for this collection
        table = self._get_or_create_table(
            collection_id, 
            schema=DocumentChunk, 
            primary_key="chunk_id"
        )
        
        # Metadata table
        meta_table = self._get_or_create_table(
            f"{collection_id}_meta", 
            schema=Document, 
            primary_key="document_id"
        )

        # 4. Process documents in parallel
        sem = asyncio.Semaphore(INDEXING_WORKERS)
        tasks = []

        async def wrapped_process(pointer):
            async with sem:
                await self._process_document(
                    pointer, 
                    col_config, 
                    table, 
                    meta_table, 
                    server_config.min_size, 
                    server_config.max_size
                )

        async for pointer in source.get_documents():
            tasks.append(asyncio.create_task(wrapped_process(pointer)))
        
        if tasks:
            await asyncio.gather(*tasks)

    async def _process_document(self, pointer, col_config, table, meta_table, min_size, max_size):
        """
        Processes a single document: generates summary, embedding, and chunks.
        """
        try:
            # a. Handle Document Metadata
            metadata_dict = await pointer.get_metadata()
            
            # Generate document summary and embedding
            doc_text = " ".join(await self._collect_all_text(pointer))
            doc_summary_list = await self.context.llm([
                f"{col_config.doc_summary_prompt}\n\n{doc_text}"
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
            meta_table.upsert([doc_record])

            # b. Handle Chunks
            chunks_to_index = []
            async for meta, text_list in pointer.get_chunks():
                # Transform semantic chunks -> embedding chunks
                async for e_meta, e_text_list in create_embedding_chunks(
                    self._wrap_semantic_chunk((meta, text_list)), 
                    min_size=min_size, 
                    max_size=max_size
                ):
                    chunk_text = "".join(e_text_list)
                    
                    # Generate chunk summary and embedding
                    chunk_summary_list = await self.context.llm([
                        f"{col_config.chunk_summary_prompt}\n\n{chunk_text}"
                    ])
                    chunk_summary = chunk_summary_list[0]
                    
                    chunk_vector = await self.context.embedding.query(chunk_summary)
                    
                    chunks_to_index.append(DocumentChunk(
                        document_id=pointer.document_id,
                        chunk_id=f"{pointer.document_id}?c={len(chunks_to_index)}",
                        text=chunk_text,
                        summary=chunk_summary,
                        embedding=chunk_vector,
                        metadata=e_meta
                    ))
            
            if chunks_to_index:
                table.upsert(chunks_to_index)
        except Exception as e:
            print(f"Error processing document {pointer.document_id}: {e}")

    def _get_or_create_table(self, name: str, schema: Any, primary_key: Any):
        """
        Returns the table if it exists, otherwise creates it.
        """
        try:
            return self.context.db.open_table(name)
        except:
            return self.context.db.create_table(
            name, 
            schema=schema, 
            primary_key=primary_key
        )

    async def _collect_all_text(self, pointer) -> List[str]:
        """Helper to collect all text from a document for high-level summary."""
        texts = []
        async for _, text_list in pointer.get_chunks():
            texts.extend(text_list)
        return texts

    async def _wrap_semantic_chunk(self, chunk: tuple):
        """Helper to turn a single chunk into an async iterator for create_embedding_chunks."""
        yield chunk

    async def search(self, collection_id: str, query: str, limit: int = 5):
        """
        Search for relevant chunks in a specific collection using explicit vectors.
        """
        # Generate query vector
        query_vector = await self.context.embedding.query(query)
        
        table = self.context.db.open_table(collection_id)
        # Use the vector explicitly for search
        results = table.search(query_vector).limit(limit).to_pydantic(DocumentChunk)
        return results
