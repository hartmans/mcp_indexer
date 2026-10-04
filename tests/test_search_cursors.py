import asyncio
import copy
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest

from mcp_indexer import search as searching
from mcp_indexer.models import Document, DocumentChunk, ChunkSummary
from mcp_indexer.plugins.base import DocumentSource
from mcp_indexer.search import ChunkHit, DocumentHit, NativeSearchBatch, VectorSearchBatch
from mcp_indexer.search_state import InvalidSearchCursor, SearchAdvanceError, SearchSessionStore


def doc(name, score=0.9):
    return DocumentHit(name, score, name)


class Harness:
    def __init__(self, monkeypatch, *, documents=(), chunks=(), native=(), rerank=True):
        self.documents = list(documents)
        self.chunks = list(chunks)
        self.native_batches = list(native)
        self.calls = []
        self.scores = {}
        self.fail_score = False
        self.fail_assembly = set()
        self.score_gate = None
        self.score_started = asyncio.Event()
        self.clock = 0
        self.store = SearchSessionStore(clock=lambda: self.clock)
        self.source = SimpleNamespace(
            id="test", _search_sessions=self.store,
            config=SimpleNamespace(min_cosine_distance=0.7, min_rerank_score=0.5),
            reranker=self if rerank else None, native_search=self.native_search,
            fetch_chunk=self.fetch_chunk,
            context=SimpleNamespace(embedding=SimpleNamespace(query=self.embed)),
        )
        self.indexer = SimpleNamespace(context=SimpleNamespace(collections={"test": self.source}))
        monkeypatch.setattr(searching, "vector_search_batch", self.vector)
        monkeypatch.setattr(searching, "assemble_hits", self.assemble)

    async def setup(self):
        self.calls.append(("setup",))

    async def embed(self, query):
        self.calls.append(("embed", query))
        return [1.0]

    async def vector(self, source, vector, dl, cl, threshold, *, document_offset, chunk_offset):
        self.calls.append(("vector", dl, cl, document_offset, chunk_offset, threshold))
        return VectorSearchBatch(
            self.documents[document_offset:document_offset + dl],
            self.chunks[chunk_offset:chunk_offset + cl],
        )

    async def native_search(self, query, **kwargs):
        self.calls.append(("native", query, kwargs))
        pos = kwargs["document_offset"]
        if pos >= len(self.native_batches):
            return None
        batch = self.native_batches[pos]
        if isinstance(batch, Exception):
            raise batch
        return NativeSearchBatch(batch, pos + 1, kwargs["chunk_offset"])

    async def fetch_chunk(self, document_id, metadata, scope):
        self.calls.append(("fetch", document_id, dict(metadata), scope))
        return f"{document_id}:{metadata['o']}"

    async def score(self, query, texts, top_n):
        self.calls.append(("score", query, list(texts), top_n))
        self.score_started.set()
        if self.score_gate:
            await self.score_gate.wait()
        if self.fail_score:
            raise RuntimeError("unavailable")
        return sorted([(i, self.scores.get(text, 0.9)) for i, text in enumerate(texts)],
                      key=lambda item: -item[1])

    async def assemble(self, indexer, source, hits):
        grouped = {}
        for hit in hits:
            grouped.setdefault(hit.document_id, []).append(hit)
        self.calls.append(("assemble", list(grouped)))
        return [(SimpleNamespace(document_id=name, summary=name),
                 [hit for hit in candidates if isinstance(hit, ChunkHit)])
                for name, candidates in grouped.items() if name not in self.fail_assembly]

    async def search(self, query=None, **kwargs):
        return await searching.search(self.indexer, "test", query, **kwargs)

    def count(self, operation):
        return sum(call[0] == operation for call in self.calls)


@pytest.fixture
def harness(monkeypatch):
    instances = []
    def make(**kwargs):
        instance = Harness(monkeypatch, **kwargs)
        instances.append(instance)
        return instance
    yield make
    for instance in instances:
        instance.store.close()


def names(page):
    return [document.document_id for document, _ in page.results]


@pytest.mark.parametrize("rerank", [False, True])
async def test_all_streams_resume_and_documents_return_once(harness, rerank):
    h = harness(documents=[doc("a"), doc("b"), doc("c")],
                chunks=[ChunkHit("a", {"o": 0}, 0.9), ChunkHit("d", {"o": 0}, 0.9)],
                native=[[doc("a")], [], [doc("e")], [doc("f")]], rerank=rerank)
    page = await h.search("query", limit=2, document_candidate_limit=1, chunk_candidate_limit=1)
    found = names(page)
    for _ in range(20):
        if page.resume_cursor is None:
            break
        page = await h.search(cursor=page.resume_cursor, limit=2)
        found.extend(names(page))
    assert page.resume_cursor is None
    assert len(found) == len(set(found)) == 6
    assert set(found) == set("abcdef")
    assert h.count("embed") == 1
    assert h.count("setup") == int(rerank)
    if rerank:
        texts = [text for call in h.calls if call[0] == "score" for text in call[2]]
        assert len(texts) == len(set(texts))
        assert all(call[3] is None for call in h.calls if call[0] == "score")


async def test_rejected_summary_later_chunk_and_pending_competition(harness):
    h = harness(documents=[doc("a"), doc("b"), doc("c")],
                native=[[], [ChunkHit("c", {"o": 0}, 1)], [ChunkHit("a", {"o": 0}, 1)]])
    h.scores = {"a": 0.9, "b": 0.8, "c": 0.4, "c:0": 0.95}
    first = await h.search("q", limit=1, document_candidate_limit=3, chunk_candidate_limit=1)
    assert names(first) == ["a"]
    second = await h.search(cursor=first.resume_cursor, limit=1)
    assert names(second) == ["c"]
    third = await h.search(cursor=second.resume_cursor, limit=1)
    assert names(third) == ["b"]
    assert not any(call[0] == "fetch" and call[1] == "a" for call in h.calls)
    assert [text for call in h.calls if call[0] == "score" for text in call[2]].count("c") == 1


async def test_chunk_identity_and_distinct_document_limit(harness):
    h = harness(documents=[doc("a"), doc("b")],
                chunks=[ChunkHit("a", {"o": 0, "s": 1}, 1, "a?c=0"),
                        ChunkHit("a", {"o": 1, "s": 1}, 1, "a?c=1")],
                native=[[doc("a"), ChunkHit("a", {"s": 1, "o": 0}, 1)]])
    first = await h.search("q", limit=2)
    assert names(first) == ["a", "b"]
    assert len(first.results[0][1]) == 2
    assert h.count("fetch") == 2
    assert h.calls[-2][0] == "score"
    assert len(h.calls[-2][2]) == 4


async def test_replay_window_queries_limit_and_mutation(harness):
    h = harness(documents=[doc("a"), doc("b"), doc("c")])
    p0 = await h.search("original", rerank_query="natural", limit=1)
    c0 = p0.resume_cursor
    p1 = await h.search(cursor=c0, limit=1)
    c1 = p1.resume_cursor
    calls = copy.deepcopy(h.calls)
    p1.results[0][0].summary = "tampered"
    replay = await h.search("different", cursor=c0, limit=20, rerank_query="other")
    assert names(replay) == ["b"]
    assert replay.results[0][0].summary == "b"
    assert replay.resume_cursor == c1
    assert len(replay.warnings) == 2
    assert h.calls == calls
    assert not (await h.search(cursor=c0)).warnings
    p2 = await h.search(cursor=c1, limit=1)
    assert names(p2) == ["c"] and p2.resume_cursor is None
    assert names(await h.search(cursor=c1)) == ["c"]
    with pytest.raises(InvalidSearchCursor):
        await h.search(cursor=c0)


async def test_empty_native_batch_and_rejected_page_are_not_exhaustion(harness):
    h = harness(native=[[], [doc("low")], [doc("good")]])
    h.scores["low"] = 0.1
    page = await h.search("q", limit=1)
    assert not page.results and page.resume_cursor
    page = await h.search(cursor=page.resume_cursor)
    assert not page.results and page.resume_cursor
    page = await h.search(cursor=page.resume_cursor)
    assert names(page) == ["good"] and page.resume_cursor
    page = await h.search(cursor=page.resume_cursor)
    assert page.resume_cursor is None


async def test_expiry_refresh_and_collection_isolation(harness):
    h = harness(documents=[doc("a"), doc("b"), doc("c")])
    p0 = await h.search("q", limit=1)
    p1 = await h.search(cursor=p0.resume_cursor)
    other = SimpleNamespace(**vars(h.source))
    other._search_sessions = SearchSessionStore()
    h.indexer.context.collections["other"] = other
    with pytest.raises(InvalidSearchCursor):
        await searching.search(h.indexer, "other", cursor=p0.resume_cursor)
    other._search_sessions.close()
    h.clock = 3599
    await h.search(cursor=p0.resume_cursor)
    h.clock = 3601
    assert (await h.search(cursor=p0.resume_cursor)).resume_cursor == p1.resume_cursor
    h.clock = 7201
    with pytest.raises(InvalidSearchCursor):
        await h.search(cursor=p0.resume_cursor)
    assert not h.store._tokens


async def test_timer_releases_idle_state_without_requests(harness):
    h = harness(documents=[doc("a"), doc("b")])
    h.store = SearchSessionStore(ttl=0.01)
    h.source._search_sessions = h.store
    await h.search("q", limit=1)
    await asyncio.sleep(0.03)
    assert not h.store._tokens


async def test_concurrent_retry_and_cancelled_waiter_share_advancement(harness):
    h = harness(documents=[doc("a"), doc("b")])
    p0 = await h.search("q", limit=1, document_candidate_limit=1, chunk_candidate_limit=0)
    h.score_gate = asyncio.Event()
    h.score_started.clear()
    first = asyncio.create_task(h.search(cursor=p0.resume_cursor, limit=1))
    await h.score_started.wait()
    second = asyncio.create_task(h.search(cursor=p0.resume_cursor, limit=1))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    h.clock = 4000  # Active work must survive the inactivity deadline.
    h.score_gate.set()
    page = await second
    assert names(page) == ["b"]
    assert h.count("score") == 2
    assert names(await h.search(cursor=p0.resume_cursor)) == ["b"]


async def test_reranker_error_reuses_discovery_and_prepared_text(harness):
    h = harness(documents=[doc("a"), doc("b")])
    p0 = await h.search("q", limit=1, document_candidate_limit=1, chunk_candidate_limit=0)
    h.fail_score = True
    with pytest.raises(SearchAdvanceError):
        await h.search(cursor=p0.resume_cursor)
    discoveries = h.count("vector")
    h.fail_score = False
    assert names(await h.search(cursor=p0.resume_cursor)) == ["b"]
    assert h.count("vector") == discoveries


async def test_snapshot_failure_reuses_scores_and_assembly(harness, monkeypatch):
    h = harness(documents=[doc("a"), doc("b")])
    p0 = await h.search("q", limit=1, document_candidate_limit=1, chunk_candidate_limit=0)
    original = searching._copy_page
    failed = False
    def snapshot(value):
        nonlocal failed
        if isinstance(value, searching.SearchResult) and not failed:
            failed = True
            raise RuntimeError("snapshot failed")
        return original(value)
    monkeypatch.setattr(searching, "_copy_page", snapshot)
    with pytest.raises(RuntimeError, match="snapshot failed"):
        await h.search(cursor=p0.resume_cursor)
    scored, assembled = h.count("score"), h.count("assemble")
    assert names(await h.search(cursor=p0.resume_cursor)) == ["b"]
    assert (h.count("score"), h.count("assemble")) == (scored, assembled)


async def test_discovery_failure_keeps_native_position_and_warns(harness):
    h = harness(documents=[doc("a")], native=[RuntimeError("offline"), [doc("b")]])
    p0 = await h.search("q", limit=1)
    assert names(p0) == ["a"] and p0.warnings
    h.native_batches[0] = []
    p1 = await h.search(cursor=p0.resume_cursor)
    assert not p1.results and p1.resume_cursor
    assert h.calls[-1][0] == "native"
    assert h.calls[-1][2]["document_offset"] == 0
    assert names(await h.search(cursor=p1.resume_cursor)) == ["b"]


async def test_all_discovery_failures_do_not_rotate_cursor(harness, monkeypatch):
    h = harness(documents=[doc("a")])
    p0 = await h.search("q", limit=1, document_candidate_limit=1, chunk_candidate_limit=0)
    async def fail_vector(*args, **kwargs):
        return VectorSearchBatch(RuntimeError("offline"), [])
    monkeypatch.setattr(searching, "vector_search_batch", fail_vector)
    h.native_batches = [RuntimeError("offline")]
    # Native was already exhausted on the first page; only vector is attempted.
    with pytest.raises(SearchAdvanceError):
        await h.search(cursor=p0.resume_cursor)
    assert h.store.lookup(p0.resume_cursor).current_cursor == p0.resume_cursor


async def test_preparation_and_assembly_have_bounded_retries(harness):
    h = harness(documents=[doc("bad"), doc("good")], chunks=[ChunkHit("missing", {"o": 0}, 1)])
    async def missing(*args, **kwargs):
        raise OSError("gone")
    h.source.fetch_chunk = missing
    h.fail_assembly.add("bad")
    page = await h.search("q", limit=3)
    assert names(page) == ["good"] and page.warnings
    page = await h.search(cursor=page.resume_cursor, limit=3)
    assert page.resume_cursor is None and page.warnings
    assert h.count("score") == 1
    assert [call for call in h.calls if call[0] == "assemble"] == [
        ("assemble", ["bad", "good"]), ("assemble", ["bad"]),
    ]


@pytest.mark.parametrize("override", [None, "", " ", "natural"])
async def test_query_routing_and_normalization(harness, override):
    h = harness(documents=[doc("a"), doc("b")])
    p0 = await h.search("cat:Physics", rerank_query=override, limit=1)
    assert ("embed", override if override and override.strip() else "cat:Physics") in h.calls
    assert next(call for call in h.calls if call[0] == "native")[1] == "cat:Physics"
    p1 = await h.search(cursor=p0.resume_cursor, rerank_query=override)
    assert not p1.warnings


async def test_candidate_limits_and_invalid_native_progress(harness):
    h = harness(documents=[doc("a")])
    with pytest.raises(ValueError, match="query"):
        await h.search()
    with pytest.raises(ValueError, match="positive"):
        await h.search("q", limit=0)
    with pytest.raises(ValueError, match="Candidate"):
        await h.search("q", document_candidate_limit=0, chunk_candidate_limit=0)
    p0 = await h.search("q", document_candidate_limit=1, chunk_candidate_limit=0)
    with pytest.raises(ValueError, match="cannot change"):
        await h.search(cursor=p0.resume_cursor, document_candidate_limit=2)
    async def stalled(query, **kwargs):
        return NativeSearchBatch([], kwargs["document_offset"], kwargs["chunk_offset"])
    h.source.native_search = stalled
    h.store.lookup(p0.resume_cursor).native_done = False
    page = await h.search(cursor=p0.resume_cursor)
    assert page.resume_cursor and page.warnings


async def test_source_stores_are_instance_owned():
    source = DocumentSource("a", context=object(), collection_config=object())
    other = DocumentSource("b", context=object(), collection_config=object())
    assert source._search_sessions is not other._search_sessions
    assert await source.native_search("q", document_limit=1, chunk_limit=1) is None


async def test_saved_thresholds_and_reranker_do_not_change(harness):
    h = harness(documents=[doc("a"), doc("b")])
    h.scores = {"a": 0.9, "b": 0.6}
    p0 = await h.search("q", limit=1, document_candidate_limit=1, chunk_candidate_limit=0)
    h.source.config.min_cosine_distance = 0.1
    h.source.config.min_rerank_score = 0.8
    h.source.reranker = None
    assert names(await h.search(cursor=p0.resume_cursor)) == ["b"]
    assert h.count("score") == 2
    assert all(call[-1] == 0.7 for call in h.calls if call[0] == "vector")


async def test_uneven_native_offsets_are_source_owned(harness):
    h = harness()
    positions = []
    async def native(query, **kwargs):
        positions.append((kwargs["document_offset"], kwargs["chunk_offset"]))
        if len(positions) == 1:
            return NativeSearchBatch([], 17, 0)
        if len(positions) == 2:
            return NativeSearchBatch([ChunkHit("a", {"o": 0}, 1)], 17, 33)
        return None
    h.source.native_search = native
    page = await h.search("q")
    assert not page.results and page.resume_cursor
    page = await h.search(cursor=page.resume_cursor)
    assert names(page) == ["a"]
    page = await h.search(cursor=page.resume_cursor)
    assert page.resume_cursor is None
    assert positions == [(0, 0), (17, 0), (17, 33)]


@pytest.mark.parametrize("ranking", [[(0, 0.8), (0, 0.7)], [], [(0, float("nan"))], [(1, 0.9)]])
async def test_invalid_rerank_response_is_retryable(harness, ranking):
    h = harness(documents=[doc("a")])
    async def score(*args, **kwargs):
        return ranking
    h.score = score
    with pytest.raises(SearchAdvanceError):
        await h.search("q")
    assert not h.store._tokens  # Failed initial requests expose no continuation.


async def test_reranker_setup_failure_precedes_discovery(harness):
    h = harness()
    async def fail():
        raise RuntimeError("no model")
    h.setup = fail
    with pytest.raises(SearchAdvanceError, match="initialize"):
        await h.search("q")
    assert h.calls == [] and not h.store._tokens


async def test_successful_initial_terminal_response_releases_session(harness):
    h = harness()
    page = await h.search("q")
    assert page.resume_cursor is None and not h.store._tokens


async def test_native_score_diagnostics(harness, caplog):
    h = harness(native=[[doc("a", 0.75)]])
    searching.set_debug_search()
    try:
        await h.search("q")
    finally:
        searching.set_debug_search(False)
    assert "native hit for test: a native_score=0.7500" in caplog.text
    assert "reranked a: rerank_score=0.9000" in caplog.text


async def test_vector_offsets_and_errors_remain_independent(monkeypatch):
    statements = []
    calls = 0
    class Session:
        async def execute(self, statement):
            nonlocal calls
            calls += 1
            statements.append(statement)
            if calls == 2:
                raise OSError("chunk stream unavailable")
            return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: [
                {"document_id": "a", "summary": "summary", "distance": 0.2},
            ]))
    @asynccontextmanager
    async def db():
        yield Session()
    source = SimpleNamespace(id="test", context=SimpleNamespace(get_session=db))
    batch = await searching.vector_search_batch(
        source, [1.0], 3, 4, 0.7, document_offset=17, chunk_offset=33,
    )
    assert [hit.document_id for hit in batch.documents] == ["a"]
    assert isinstance(batch.chunks, OSError)
    params = [statement.compile().params for statement in statements]
    assert params[0]["param_2"] == 3 and params[0]["param_3"] == 17
    assert params[1]["param_2"] == 4 and params[1]["param_3"] == 33
    assert all(param["collection_id_1"] == "test" for param in params)
    assert all(param["param_1"] == 0.7 for param in params)


async def test_replay_copies_real_detached_orm_graph(harness, monkeypatch):
    h = harness(documents=[doc("a"), doc("b")])
    async def assemble(indexer, source, hits):
        results = []
        for hit in hits:
            document = Document(collection_id="test", document_id=hit.document_id,
                                summary="original", title="Title", keywords=[],
                                title_strength=1, embedding=[1.0], last_modified=datetime.now())
            summary = ChunkSummary(collection_id="test", document_id=hit.document_id,
                                   summary_span=0, summary="passage")
            chunk = DocumentChunk(collection_id="test", document_id=hit.document_id,
                                  chunk_id=hit.document_id + "?c=0", order=0,
                                  metadata_str='{"o":0}', embedding=[1.0], summary=summary)
            document.chunks = [chunk]
            results.append((document, [chunk]))
        return results
    monkeypatch.setattr(searching, "assemble_hits", assemble)
    p0 = await h.search("q", limit=1)
    p1 = await h.search(cursor=p0.resume_cursor)
    p1.results[0][0].summary = "changed"
    p1.results[0][1][0].summary.summary = "changed"
    replay = await h.search(cursor=p0.resume_cursor)
    assert replay.results[0][0].summary == "original"
    assert replay.results[0][1][0].summary.summary == "passage"
    assert replay.results[0][0].chunks[0] is replay.results[0][1][0]
