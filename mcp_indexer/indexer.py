import lancedb
from typing import List, Optional, Any
from datetime import datetime
from mcp_indexer.context import Context
from mcp_indexer.config import ConfigManager
from mcp_indexer.plugins.base import Document, DocumentChunk, create_embedding_chunks

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
        
        # 2. Initialize the plugin
        source = source_plugin_class(
            collection_id=collection_id,
            context=self.context,
            collection_config=col_config
        )
        
        # 3. Ensure the LanceDB tables exist for this collection
        # We use the collection_id as the table name
        table = self.context.db.create_table(
            collection_id, 
            schema=DocumentChunk, 
            mode="overwrite"
        )
        
        # Metadata table
        meta_table = self.context.db.create_table(
            f"{collection_id}_meta", 
            schema=Document, 
            mode="overwrite"
        )

        # 4. Process documents
        async for pointer in source.get_documents():
            # a. Handle Document Metadata
            metadata_dict = await pointer.get_metadata()
            
            # Generate document summary and embedding
            # Using the prompt from the collection config
            doc_text = " ".join(await self._collect_all_text(pointer))
            doc_summary = await self.context.llm([
                f"{col_config.summary_prompt}\n\n{doc_text}"
            ])
            doc_summary = doc_summary[0]
            
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
                    min_size=100, 
                    max_size=1000
                ):
                    chunk_text = "".join(e_text_list)
                    
                    # Generate chunk summary and embedding
                    chunk_summary = await self.context.llm([
                        f"{col_config.summary_prompt}\n\n{chunk_text}"
                    ])
                    chunk_summary = chunk_summary[0]
                    
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
