from datetime import datetime
from typing import Any, AsyncGenerator

import pytest

from mcp_indexer.context import Context
from mcp_indexer.plugins.base import ChunkInfo, DocumentPointer, DocumentSource, create_embedding_chunks


class MockDocumentPointer(DocumentPointer):
    def __init__(self, source, document_id, title, chunks_data):
        super().__init__(source=source, document_id=document_id, last_modified=datetime.now())
        self.title = title
        self.chunks_data = chunks_data

    async def get_chunks(self, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        del min_size, max_size
        for metadata, text_list in self.chunks_data:
            yield metadata, text_list

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> list[str]:
        search_meta = {k: v for k, v in chunk_metadata.items() if k not in ("o", "s")}
        for metadata, text_list in self.chunks_data:
            if metadata == search_meta:
                return text_list
        raise ValueError("Chunk not found")


class MockDocumentSource(DocumentSource):
    def __init__(self, collection_id, context, collection_config, documents_data):
        super().__init__(collection_id, context=context, collection_config=collection_config)
        self.documents_data = documents_data

    def fetch_document(self, document_id: str) -> MockDocumentPointer:
        if document_id not in self.documents_data:
            raise ValueError("Document not found")
        title, chunks_data = self.documents_data[document_id]
        return MockDocumentPointer(self, document_id, title, chunks_data)


def test_create_embedding_chunks_basic():
    results = create_embedding_chunks({"id": "1"}, ["Hello ", "world!"], 10, 20)
    assert results == [({"id": "1", "o": 0, "s": 12}, "Hello world!")]


def test_create_embedding_chunks_multiple_small_items():
    results = create_embedding_chunks({"id": "1"}, ["A"] * 15, 10, 20)
    assert len(results) == 1
    assert results[0][0]["s"] == 15
    assert results[0][1] == "A" * 15


def test_create_embedding_chunks_hard_slice():
    text = "This is a very long string that exceeds max size"
    results = create_embedding_chunks({"id": "1"}, [text], 10, 20)

    assert len(results) == 2
    assert [meta["s"] for meta, _ in results] == [20, 20]
    assert "".join(chunk for _, chunk in results) == text


def test_create_embedding_chunks_boundary_flush():
    text = "ShortMediumLengthLongerThanMax"
    results = create_embedding_chunks({"id": "1"}, ["Short", "MediumLength", "LongerThanMax"], 10, 20)

    assert len(results) == 2
    assert [meta["s"] for meta, _ in results] == [20, len(text) - 20]
    assert "".join(chunk for _, chunk in results) == text


def test_create_embedding_chunks_small_tail():
    results = create_embedding_chunks({"id": "1"}, ["Short"], 10, 20)
    assert len(results) == 1
    assert results[0] == ({"id": "1", "o": 0, "s": 5}, "Short")


@pytest.mark.asyncio
async def test_document_source_fetch_chunk(test_context):
    source = MockDocumentSource("col1", test_context, object(), {
        "doc1": ("Title 1", [
            ({"meta1": "val1"}, ["Hello ", "World"]),
            ({"meta2": "val2"}, ["Foo ", "Bar"]),
        ])
    })

    # Note: The new Context doesn't have a fetch_chunk method
    # This test verifies the MockDocumentSource works correctly
    chunk = source.fetch_document("doc1")
    text_list = await chunk.fetch_chunk({"meta1": "val1"})
    assert text_list == ["Hello ", "World"]


@pytest.mark.asyncio
async def test_document_source_fetch_chunk_not_found(test_context):
    source = MockDocumentSource("col1", test_context, object(), {})

    with pytest.raises(ValueError):
        await source.fetch_document("doc_none")