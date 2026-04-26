import pytest
import asyncio
from typing import AsyncGenerator, Any
from lance_indexer.plugins.base import (
    DocumentSource, 
    DocumentPointer, 
    create_embedding_chunks, 
    ChunkInfo
)
from lance_indexer.context import Context
from lance_indexer.config import CollectionConfig

# Mock implementations for testing
class MockDocumentPointer(DocumentPointer):
    def __init__(self, source, document_id, title, chunks_data):
        super().__init__(source, document_id)
        self.title = title
        self.chunks_data = chunks_data # list of (metadata, text_list)

    async def get_chunks(self) -> AsyncGenerator[ChunkInfo, None]:
        for metadata, text_list in self.chunks_data:
            yield metadata, text_list

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> list[str]:
        # Simple mock: find the chunk that matches the metadata, ignoring offsets
        # Create a copy of the metadata without the 'o' and 's' keys for comparison
        search_meta = {k: v for k, v in chunk_metadata.items() if k not in ('o', 's')}
        for metadata, text_list in self.chunks_data:
            if metadata == search_meta:
                return text_list
        raise ValueError("Chunk not found")

class MockDocumentSource(DocumentSource):
    def __init__(self, collection_id, context, collection_config, documents_data):
        super().__init__(collection_id, context=context, collection_config=collection_config)
        self.documents_data = documents_data # dict of doc_id -> (title, chunks_data)

    def fetch_document(self, document_id: str) -> MockDocumentPointer:
        if document_id not in self.documents_data:
            raise ValueError("Document not found")
        title, chunks_data = self.documents_data[document_id]
        return MockDocumentPointer(self, document_id, title, chunks_data)

# Helpers
async def to_list(async_gen):
    return [item async for item in async_gen]

@pytest.mark.asyncio
async def test_create_embedding_chunks_basic():
    # Case: Semantic chunk fits within max_size, but exceeds min_size
    # min=10, max=20
    # Semantic chunk: ("meta", ["Hello ", "world!"]) -> 12 chars
    async def chunk_gen():
        yield {"id": "1"}, ["Hello ", "world!"]
    
    results = await to_list(create_embedding_chunks(chunk_gen(), 10, 20))
    assert len(results) == 1
    assert results[0] == ({"id": "1", "o": 0, "s": 12}, ["Hello world!"])

@pytest.mark.asyncio
async def test_create_embedding_chunks_multiple_small_items():
    # min=10, max=20
    # Semantic chunk: ("meta", ["A"]*15)
    # Should absorb the remainder of 5 into the first chunk of 10.
    async def chunk_gen():
        yield {"id": "1"}, ["A"]*15
    
    results = await to_list(create_embedding_chunks(chunk_gen(), 10, 20))
    # Should be one chunk of 15
    assert len(results) == 1
    assert results[0][0]["s"] == 15

@pytest.mark.asyncio
async def test_create_embedding_chunks_hard_slice():
    # min=10, max=20
    # Single item > max_size
    async def chunk_gen():
        yield {"id": "1"}, ["This is a very long string that exceeds max size"]
    
    results = await to_list(create_embedding_chunks(chunk_gen(), 10, 20))
    # 48 chars / 20 = 3 chunks (20, 20, 8)
    assert len(results) == 3
    assert results[0][0]["s"] == 20
    assert results[1][0]["s"] == 20
    assert results[2][0]["s"] == 8

@pytest.mark.asyncio
async def test_create_embedding_chunks_boundary_flush():
    # min=10, max=20
    # Item would push buffer over max_size -> flush buffer first
    async def chunk_gen():
        yield {"id": "1"}, ["Short", "MediumLength", "LongerThanMax"] 
        # "Short" (5), "MediumLength" (12), "LongerThanMax" (13)
        # 5 + 12 = 17 (<= 20), but 17 + 13 = 30 (> 20)
        # Buffer: ["Short", "MediumLength"] -> 17. 17 >= 10, so it would have been emitted already if we checked.
        # Wait, create_embedding_chunks emits if buffer_len >= min_size.
        # "Short" (5) < 10.
        # "Short" + "MediumLength" (17) >= 10 -> EMIT.
        # Buffer empty.
        # "LongerThanMax" (13) >= 10 -> EMIT.
    
    results = await to_list(create_embedding_chunks(chunk_gen(), 10, 20))
    assert len(results) == 2
    assert results[0][0]["s"] == 17
    assert results[1][0]["s"] == 13

@pytest.mark.asyncio
async def test_create_embedding_chunks_small_tail():
    # min=10, max=20
    # Text ends before reaching min_size
    async def chunk_gen():
        yield {"id": "1"}, ["Short"]
    
    results = await to_list(create_embedding_chunks(chunk_gen(), 10, 20))
    assert len(results) == 1
    assert results[0][0]["s"] == 5

@pytest.mark.asyncio
async def test_document_source_fetch_chunk():
    # Context is a dataclass; pass None for required fields
    ctx = Context(db=None, embedding_fn=None, llm=None, config=None)
    # Use a simple dict/mock for CollectionConfig
    cfg = MagicMock(spec=CollectionConfig) if 'MagicMock' in globals() else None
    docs_data = {
        "doc1": ("Title 1", [
            ({"meta1": "val1"}, ["Hello ", "World"]),
            ({"meta2": "val2"}, ["Foo ", "Bar"]),
        ])
    }
    source = MockDocumentSource("col1", ctx, cfg, docs_data)
    
    # Test semantic scope
    text_semantic = await source.fetch_chunk("doc1", {"meta1": "val1"}, scope='semantic')
    assert text_semantic == "Hello World"
    
    # Test embedding scope (full)
    text_emb_full = await source.fetch_chunk("doc1", {"meta1": "val1"}, scope='embedding')
    assert text_emb_full == "Hello World"
    
    # Test embedding scope (sliced)
    # Metadata is usually extended by create_embedding_chunks with 'o' and 's'
    text_emb_sliced = await source.fetch_chunk("doc1", {"meta1": "val1", "o": 6, "s": 5}, scope='embedding')
    assert text_emb_sliced == "World"

@pytest.mark.asyncio
async def test_document_source_fetch_chunk_not_found():
    ctx = Context(db=None, embedding_fn=None, llm=None, config=None)
    cfg = None
    source = MockDocumentSource("col1", ctx, cfg, {})
    
    with pytest.raises(ValueError):
        await source.fetch_chunk("doc_none", {}, scope='semantic')
