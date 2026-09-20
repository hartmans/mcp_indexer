import asyncio
import logging
import math
from datetime import datetime
from types import SimpleNamespace
from contextlib import asynccontextmanager

import pytest

from mcp_indexer import search as searching
from mcp_indexer.config import CollectionConfig
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.models import Document, DocumentChunk
from mcp_indexer.search import DocumentHit, ChunkHit


def _config(**overrides):
    """A real CollectionConfig so thresholds come from the config model."""
    base = {
        "collection_id": "test",
        "tool_prefix": "test",
    }
    base.update(overrides)
    return CollectionConfig.model_validate(base)


@pytest.mark.parametrize("enabled", [False, True])
async def test_search_selection_with_and_without_reranking(monkeypatch, enabled):
    hits = [DocumentHit("first", 0.9, "First summary"),
            ChunkHit("second", {"o": 0, "s": 10}, 0.8, "second?c=0")]
    events = []

    class Reranker:
        async def setup(self):
            events.append("setup")

        async def score(self, query, documents, top_n):
            assert query == "rerank question"
            assert documents == ["First summary", "Second passage"]
            assert top_n == 1
            events.append("score")
            return [(1, 0.95)]

    async def fetch(document_id, metadata, scope):
        assert scope == "embedding"
        return "Second passage"

    async def native(*args, **kwargs):
        return []

    source = SimpleNamespace(id="test", reranker=Reranker() if enabled else None,
                             fetch_chunk=fetch, native_search=native,
                             config=_config())
    indexer = SimpleNamespace(context=SimpleNamespace(collections={"test": source}))

    async def vector(*args):
        assert events == (["setup"] if enabled else [])
        return hits

    async def assemble(indexer, collection, selected):
        return selected

    monkeypatch.setattr(searching, "vector_search_hits", vector)
    monkeypatch.setattr(searching, "assemble_hits", assemble)
    result = await searching.search(indexer, "test", "question", 1,
                                    rerank_query="rerank question")
    assert [hit.document_id for hit in result] == (["second"] if enabled else ["first", "second"])
    assert events == (["setup", "score"] if enabled else [])


async def test_reranker_failure_retains_prepared_hits(caplog):
    async def fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    source = SimpleNamespace(id="test", reranker=SimpleNamespace(score=fail))
    hits = [DocumentHit("first", 1, "summary")]
    assert await searching.rerank(None, source, hits, "query", 1, 0.5) == hits
    assert "model unavailable" in caplog.text


async def test_rerank_drops_hits_below_min_rerank_score():
    class Reranker:
        async def score(self, query, documents, top_n):
            return [(0, 0.4), (1, 0.9)]

    source = SimpleNamespace(id="test", reranker=Reranker())
    hits = [DocumentHit("low", 1, "summary low"), DocumentHit("high", 1, "summary high")]
    result = await searching.rerank(None, source, hits, "query", 2, min_rerank_score=0.5)
    assert [hit.document_id for hit in result] == ["high"]
    assert result[0].score == 0.9


async def test_rerank_keeps_hits_at_the_exact_min_rerank_score():
    class Reranker:
        async def score(self, query, documents, top_n):
            return [(0, 0.5)]

    source = SimpleNamespace(id="test", reranker=Reranker())
    hits = [DocumentHit("edge", 1, "summary")]
    result = await searching.rerank(None, source, hits, "query", 1, min_rerank_score=0.5)
    assert [hit.document_id for hit in result] == ["edge"]


async def test_rerank_filters_before_the_result_limit():
    class Reranker:
        async def score(self, query, documents, top_n):
            return [(0, 0.1), (1, 0.9), (2, 0.8)]

    source = SimpleNamespace(id="test", reranker=Reranker())
    hits = [DocumentHit(f"doc{i}", 1, f"summary {i}") for i in range(3)]
    # Only one result is requested; the below-threshold first candidate must
    # not consume it.
    result = await searching.rerank(None, source, hits, "query", 1, min_rerank_score=0.5)
    assert [hit.document_id for hit in result] == ["doc1"]


async def test_rerank_logs_debug_scores_for_each_hit(caplog):
    searching.set_debug_search()
    try:
        class Reranker:
            async def score(self, query, documents, top_n):
                return [(0, 0.4), (1, 0.9)]

        source = SimpleNamespace(id="test", reranker=Reranker())
        hits = [DocumentHit("low", 1, "summary low"), DocumentHit("high", 1, "summary high")]
        await searching.rerank(None, source, hits, "query", 2, min_rerank_score=0.5)
    finally:
        searching.set_debug_search(False)
    assert "reranked low: rerank_score=0.4000" in caplog.text
    assert "reranked high: rerank_score=0.9000" in caplog.text


def test_set_debug_search_toggles_the_search_logger_level():
    assert searching.logger.level != logging.DEBUG
    searching.set_debug_search()
    assert searching.logger.level == logging.DEBUG
    searching.set_debug_search(False)
    assert searching.logger.level == logging.INFO


async def test_vector_search_logs_debug_cosine_distances(caplog, monkeypatch):
    rows = [{"document_id": "near", "summary": "summary", "distance": 0.5}]

    class _Session:
        async def execute(self, statement):
            class _M:
                def all(self):
                    return rows

            class _R:
                def mappings(self):
                    return _M()

            return _R()

        async def close(self):
            pass

    @asynccontextmanager
    async def get_session():
        yield _Session()

    class _Embedding:
        async def query(self, text):
            return [0.0] * 1

    collection = SimpleNamespace(id="vec", context=SimpleNamespace(
        get_session=get_session, embedding=_Embedding(),
    ))
    searching.set_debug_search()
    try:
        hits = await searching.vector_search_hits(
            collection, "query", document_limit=1, chunk_limit=1,
            min_cosine_distance=0.7,
        )
    finally:
        searching.set_debug_search(False)
    assert [hit.document_id for hit in hits] == ["near"]
    assert "vector hit for vec: near cosine_distance=0.5000" in caplog.text


async def test_search_logs_debug_native_scores(caplog, monkeypatch):
    native_hits = [DocumentHit("first", 0.9, "summary")]

    async def native(query, *, document_limit, chunk_limit):
        return native_hits

    async def vector(*args, **kwargs):
        return []

    async def assemble(*args):
        return []

    source = SimpleNamespace(id="test", reranker=None, native_search=native,
                             config=_config())
    indexer = SimpleNamespace(context=SimpleNamespace(collections={"test": source}))
    monkeypatch.setattr(searching, "vector_search_hits", vector)
    monkeypatch.setattr(searching, "assemble_hits", assemble)
    searching.set_debug_search()
    try:
        await searching.search(indexer, "test", "question", limit=1)
    finally:
        searching.set_debug_search(False)
    assert "native hit for test: first native_score=0.9000" in caplog.text


async def test_search_passes_configured_thresholds(monkeypatch):
    captured = {}

    async def vector(source, query, document_limit, chunk_limit, min_cosine_distance):
        captured["min_cosine_distance"] = min_cosine_distance
        return []

    async def rerank_call(indexer, collection, hits, query, hit_limit, min_rerank_score):
        captured["min_rerank_score"] = min_rerank_score
        return []

    async def native(query, **kwargs):
        return []

    class Reranker:
        async def setup(self):
            pass

        async def score(self, query, documents, top_n):
            return []

    async def assemble(*args):
        return []

    source = SimpleNamespace(
        id="test", reranker=Reranker(), native_search=native,
        config=_config(min_cosine_distance=0.9, min_rerank_score=0.7),
    )
    indexer = SimpleNamespace(context=SimpleNamespace(collections={"test": source}))
    monkeypatch.setattr(searching, "vector_search_hits", vector)
    monkeypatch.setattr(searching, "rerank", rerank_call)
    monkeypatch.setattr(searching, "assemble_hits", assemble)
    await searching.search(indexer, "test", "question", 1)
    assert captured == {"min_cosine_distance": 0.9, "min_rerank_score": 0.7}


async def test_reranker_prepares_candidates_concurrently(monkeypatch):
    started = set()
    both_started = asyncio.Event()

    async def summary(indexer, source, document_id):
        started.add(document_id)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return f"summary {document_id}"

    class Reranker:
        async def score(self, query, documents, top_n):
            assert documents == ["summary first", "summary second"]
            return [(0, 1.0), (1, 0.5)]

    monkeypatch.setattr(searching, "_document_summary", summary)
    source = SimpleNamespace(id="test", reranker=Reranker())
    result = await searching.rerank(
        object(), source,
        [DocumentHit("first", 1), DocumentHit("second", 0.5)], "query", 2,
        min_rerank_score=0.5,
    )
    assert [hit.document_id for hit in result] == ["first", "second"]


@pytest.mark.parametrize("natural", [None, "", " ", "how stars form"])
async def test_natural_query_routes_to_vector_and_native_keeps_syntax(monkeypatch, natural):
    calls = []
    async def native(query, **kwargs):
        calls.append(("native", query))
        return []
    async def vector(source, query, *args):
        calls.append(("vector", query))
        return []
    async def assemble(*args):
        return []
    source = SimpleNamespace(reranker=None, native_search=native,
                             config=_config(collection_id="wiki"))
    indexer = SimpleNamespace(context=SimpleNamespace(collections={"wiki": source}))
    monkeypatch.setattr(searching, "vector_search_hits", vector)
    monkeypatch.setattr(searching, "assemble_hits", assemble)
    await searching.search(indexer, "wiki", "cat:Astronomy", rerank_query=natural)
    expected = natural if natural and natural.strip() else "cat:Astronomy"
    assert calls == [("vector", expected), ("native", "cat:Astronomy")]


async def test_assembly_hydrates_after_summary_and_merges_chunks(monkeypatch):
    events = []
    chunk = SimpleNamespace(chunk_id="doc?c=0", summary=SimpleNamespace(summary="chunk summary"))
    document = SimpleNamespace(summary="", chunks=[chunk])

    async def ensure(*args):
        events.append("index")

    async def summary(*args):
        events.append("summary")
        document.summary = "fresh summary"
        return document.summary

    async def summarize_results(*args):
        events.append("chunks")

    class Session:
        async def execute(self, statement):
            assert events == ["index", "summary", "chunks"]
            return SimpleNamespace(scalar_one=lambda: document)

    @asynccontextmanager
    async def session():
        yield Session()

    indexer = SimpleNamespace(context=SimpleNamespace(get_session=session),
                              summarize_results=summarize_results)
    monkeypatch.setattr(searching, "_ensure_indexed", ensure)
    monkeypatch.setattr(searching, "_document_summary", summary)
    hit = ChunkHit("doc", {}, 1, "doc?c=0")
    result = await searching.assemble_hits(indexer, SimpleNamespace(id="test"),
                                           [DocumentHit("doc", 1), hit, hit])
    assert result == [(document, [chunk])]
    assert result[0][0].summary == "fresh summary"


async def test_assembly_processes_documents_concurrently(monkeypatch):
    started = set()
    both_started = asyncio.Event()

    async def ensure(indexer, collection, document_id):
        started.add(document_id)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()

    async def summary(*args):
        return "summary"

    class Session:
        async def execute(self, statement):
            document = SimpleNamespace(summary="summary", chunks=[])
            return SimpleNamespace(scalar_one=lambda: document)

    @asynccontextmanager
    async def session():
        yield Session()

    monkeypatch.setattr(searching, "_ensure_indexed", ensure)
    monkeypatch.setattr(searching, "_document_summary", summary)
    indexer = SimpleNamespace(context=SimpleNamespace(get_session=session))
    results = await asyncio.wait_for(searching.assemble_hits(
        indexer, SimpleNamespace(id="test"),
        [DocumentHit("first", 1), DocumentHit("second", 0.5)],
    ), timeout=1)
    assert len(results) == 2


def _unit(component: float) -> list[float]:
    """A unit vector along axis 0 with axis-0 component *component*.

    Against a query vector along axis 0, the cosine distance to this vector
    is exactly ``1 - component``.
    """
    vector = [0.0] * VECTOR_DIMENSIONS
    vector[0] = component
    vector[1] = math.sqrt(max(0.0, 1.0 - component * component))
    return vector


def _document(distance_from_query: float, document_id: str):
    """A Document whose embedding sits at the given cosine distance.

    With the query vector along axis 0, a document embedding with axis-0
    component ``1 - distance`` is at cosine distance ``distance``.
    """
    component = 1.0 - distance_from_query
    return Document(
        collection_id="vec",
        document_id=document_id,
        title=document_id,
        title_strength=1,
        embedding=_unit(component),
        keywords=[],
        summary=f"summary for {document_id}",
        last_modified=datetime(2026, 1, 1),
    )


async def test_min_cosine_distance_excludes_far_hits_in_database(test_context):
    """Far documents/chunks are dropped by the DB query, not fetched and cut."""
    @asynccontextmanager
    async def get_session():
        async with test_context.session_factory() as session:
            yield session

    class _Embedding:
        async def query(self, text):
            return _unit(1.0)

    collection = SimpleNamespace(id="vec", context=SimpleNamespace(
        get_session=get_session, embedding=_Embedding(),
    ))

    # Passing rows (distance <= 0.65): close, mid, mid2.
    # Failing rows (distance > 0.65): far, farthest.
    docs = [
        _document(0.0, "close"),
        _document(0.1, "mid"),
        _document(0.2, "mid2"),
        _document(0.7, "far"),
        _document(0.95, "farthest"),
    ]
    far_chunk = DocumentChunk(
        collection_id="vec", document_id="far", order=0,
        chunk_id="far?c=0", text="chunk", embedding=_unit(0.3),
        summary_span=None, metadata_str='{"o": 0, "s": 1}',
    )
    async with test_context.session_factory() as session:
        for doc in docs:
            session.add(doc)
        session.add(far_chunk)
        await session.commit()

    try:
        hits = await searching.vector_search_hits(
            collection, "query", document_limit=3, chunk_limit=3,
            min_cosine_distance=0.65,
        )
        document_ids = [h.document_id for h in hits
                        if isinstance(h, searching.DocumentHit)]
        chunk_ids = [h.chunk_id for h in hits if isinstance(h, searching.ChunkHit)]
        # The limit (3) applies to the filtered set: close, mid, mid2 returned;
        # far/farthest documents and the far chunk are never fetched.
        assert document_ids == ["close", "mid", "mid2"]
        assert "far" not in document_ids and "farthest" not in document_ids
        assert "far?c=0" not in chunk_ids
    finally:
        async with test_context.session_factory() as session:
            for doc in docs:
                await session.delete(doc)
            await session.delete(far_chunk)
            await session.commit()
