import asyncio
from types import SimpleNamespace
from contextlib import asynccontextmanager

import pytest

from mcp_indexer import search as searching
from mcp_indexer.search import DocumentHit, ChunkHit


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
                             fetch_chunk=fetch, native_search=native)
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
    assert await searching.rerank(None, source, hits, "query", 1) == hits
    assert "model unavailable" in caplog.text


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
    source = SimpleNamespace(reranker=None, native_search=native)
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
