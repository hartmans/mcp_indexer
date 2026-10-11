from datetime import datetime
import json
from typing import Any, AsyncGenerator

import pytest

from mcp_indexer.models import Document, DocumentChunk
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.plugins.base import ChunkInfo, DocumentPointer, DocumentSource, create_embedding_chunks
from mcp_indexer.server import result_info


class MockDocumentPointer(DocumentPointer):
    def __init__(self, source, document_id, title, chunks_data):
        super().__init__(source=source, document_id=document_id, last_modified=datetime.now())
        self.title = title
        self.chunks_data = chunks_data

    async def get_chunks(self, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        del min_size, max_size
        for metadata, text_list in self.chunks_data:
            yield metadata, text_list

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> str:
        search_meta = {k: v for k, v in chunk_metadata.items() if k not in ("o", "s")}
        for metadata, text_list in self.chunks_data:
            if metadata == search_meta:
                return "".join(text_list)
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
    results = create_embedding_chunks(["Hello ", "world!"], 10, 20)
    assert results == ["Hello world!"]


def test_create_embedding_chunks_multiple_small_items():
    results = create_embedding_chunks(["A"] * 15, 10, 20)
    assert results == ["A" * 15]


def test_create_embedding_chunks_hard_slice():
    text = "This is a very long string that exceeds max size"
    # Single candidate, no internal boundaries: hard-sliced at max_size.
    # The 8-char remainder cannot merge (20 + 8 > max_size=20), so it is
    # emitted as an undersized tail chunk.
    results = create_embedding_chunks([text], 10, 20)

    assert [len(chunk) for chunk in results] == [20, 20, 8]
    assert "".join(results) == text


def test_create_embedding_chunks_boundary_flush():
    text = "ShortMediumLengthLongerThanMax"
    # Cut at the first candidate boundary reaching min_size: the 17-char
    # boundary (Short+MediumLength) rather than a hard cut at 20.
    results = create_embedding_chunks(["Short", "MediumLength", "LongerThanMax"], 10, 20)

    assert [len(chunk) for chunk in results] == [17, 13]
    assert "".join(results) == text


def test_create_embedding_chunks_small_tail():
    results = create_embedding_chunks(["Short"], 10, 20)
    assert results == ["Short"]


def test_create_embedding_chunks_tail_merges_when_it_fits():
    # 25-char text, max_size=30: the whole thing fits, but the single
    # candidate is split by a boundary into 20 + 5. The 5-char tail is
    # undersized and merges into the previous chunk (20 + 5 <= 30).
    results = create_embedding_chunks(["a" * 20, "b" * 5], 10, 30)
    assert results == ["a" * 20 + "b" * 5]


def test_create_embedding_chunks_oversized_candidate_realigned():
    # The first candidate is larger than max_size with no internal
    # boundaries, so it is hard-sliced at max_size; the next cut finds the
    # candidate boundary again and lands on it, re-aligning the output with
    # the source's boundaries.
    results = create_embedding_chunks(["x" * 35, "y" * 20], 10, 30)
    # Cut 1: hard cut at 30 (no boundary within [0, 30]).
    # Cut 2: the candidate boundary at 55 is within [30, 60] and yields
    # 25 >= min_size, so the cut lands exactly on it.
    assert [len(chunk) for chunk in results] == [30, 25]
    assert "".join(results) == "x" * 35 + "y" * 20


def test_create_embedding_chunks_candidates_are_a_hint():
    # Cut at the first boundary that yields >= min_size: the boundary after
    # "a"*20 (chunk of exactly min_size), absorbing nothing before it. The
    # following chunk then runs from that boundary to the next one.
    results = create_embedding_chunks(["a" * 20, "b" * 5, "c" * 20], 10, 30)
    assert [len(chunk) for chunk in results] == [20, 25]
    assert "".join(results) == "a" * 20 + "b" * 5 + "c" * 20


def test_document_source_reranker_defaults_to_none_and_can_be_supplied():
    reranker = object()

    default_source = DocumentSource("default", context=object(), collection_config=object())
    reranked_source = DocumentSource(
        "reranked",
        context=object(),
        collection_config=object(),
        reranker=reranker,
    )

    assert default_source.reranker is None
    assert reranked_source.reranker is reranker


def test_result_info_includes_chunk_lengths():
    source = MockDocumentSource("col1", object(), object(), {})
    document = Document(
        collection_id="col1",
        document_id="doc1",
        title="Title 1",
        title_strength=0,
        embedding=[0.0] * VECTOR_DIMENSIONS,
        keywords=["alpha"],
        summary="Document summary",
        last_modified=datetime.now(),
    )
    chunk = DocumentChunk(
        collection_id="col1",
        document_id="doc1",
        order=0,
        chunk_id="doc1?c=0",
        text="hello world",
        embedding=[0.0] * VECTOR_DIMENSIONS,
        summary_span=None,
        semantic_chunk_id=0,
        semantic_length=32,
        embedding_length=11,
        metadata_str=json.dumps({"b": 10, "e": 42}),
        embedding_offset=5,
    )

    formatted = result_info([(document, [chunk])])

    assert formatted == [
        {
            "title": "Title 1",
            "keywords": ["alpha"],
            "document_id": "col1:doc1",
            "summary": "Document summary",
            "relevant_embedding_chunks": [
                {
                    "chunk_id": "col1:doc1?c=0",
                    "summary": None,
                    "semantic_length": 32,
                    "embedding_length": 11,
                }
            ],
        }
    ]


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
    text = await chunk.fetch_chunk({"meta1": "val1"})
    assert text == "Hello World"


@pytest.mark.asyncio
async def test_document_source_fetch_chunk_not_found(test_context):
    source = MockDocumentSource("col1", test_context, object(), {})

    with pytest.raises(ValueError):
        await source.fetch_document("doc_none")
