from contextlib import asynccontextmanager
from types import SimpleNamespace

from mcp_indexer.indexer import Indexer
from mcp_indexer.models import Document, DocumentChunk


class FakeMappingsResult:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


class FakeScalarsResult:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def unique(self):
        return self

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self, results):
        self.results = iter(results)

    async def execute(self, statement):
        del statement
        return next(self.results)


class FakeEmbedding:
    async def query(self, query):
        assert query == "find relevant text"
        return [0.0] * 768


class FakeReranker:
    def __init__(self):
        self.calls = []

    async def score(self, query, documents, top_n=None):
        self.calls.append((query, documents, top_n))
        return [(2, 0.9), (1, 0.8)]


class FakeSource:
    def __init__(self, reranker):
        self.reranker = reranker
        self.fetches = []

    async def fetch_chunk(self, document_id, chunk_metadata, scope):
        self.fetches.append((document_id, chunk_metadata, scope))
        return {
            "one?c=0": "raw first chunk",
            "two?c=0": "raw second chunk",
        }[f"{document_id}?c=0"]


def make_document(document_id, title, summary):
    document = Document(
        collection_id="collection",
        document_id=document_id,
        title=title,
        title_strength=0,
        embedding=[0.0] * 768,
        keywords=[],
        summary=summary,
    )
    chunk = DocumentChunk(
        collection_id="collection",
        document_id=document_id,
        order=0,
        chunk_id=f"{document_id}?c=0",
        embedding=[0.0] * 768,
        metadata_str='{"o":0,"s":20}',
    )
    document.chunks = [chunk]
    return document


async def test_search_reranks_candidates_before_python_grouping(monkeypatch):
    document_one = make_document("one", "One", "summary one")
    document_two = make_document("two", "Two", "summary two")
    candidate_rows = [
        {
            "collection_id": "collection",
            "document_id": "one",
            "chunk_id": None,
            "distance": 0.1,
        },
        {
            "collection_id": "collection",
            "document_id": "one",
            "chunk_id": "one?c=0",
            "distance": 0.2,
        },
        {
            "collection_id": "collection",
            "document_id": "two",
            "chunk_id": "two?c=0",
            "distance": 0.3,
        },
    ]
    session = FakeSession(
        [
            FakeMappingsResult(candidate_rows),
            FakeScalarsResult([document_one, document_two]),
            FakeScalarsResult([document_one, document_two]),
        ]
    )
    reranker = FakeReranker()
    source = FakeSource(reranker)

    @asynccontextmanager
    async def get_session():
        yield session

    context = SimpleNamespace(
        embedding=FakeEmbedding(),
        collections={"collection": source},
        get_session=get_session,
        config=SimpleNamespace(get_server_config=lambda: SimpleNamespace()),
    )
    indexer = Indexer(context)
    summarized = []

    async def summarize_results(collection_id, document_ids):
        summarized.append((collection_id, set(document_ids)))

    monkeypatch.setattr(indexer, "summarize_results", summarize_results)

    results = await indexer.search(
        "collection",
        "find relevant text",
        limit=2,
    )

    assert reranker.calls == [
        (
            "find relevant text",
            [
                "Title: One\n\nsummary one",
                "Title: One\n\nraw first chunk",
                "Title: Two\n\nraw second chunk",
            ],
            2,
        )
    ]
    assert source.fetches == [
        ("one", {"o": 0, "s": 20}, "embedding"),
        ("two", {"o": 0, "s": 20}, "embedding"),
    ]
    assert summarized == [("collection", {"one", "two"})]
    assert [(document.document_id, [chunk.chunk_id for chunk in chunks]) for document, chunks in results] == [
        ("two", ["two?c=0"]),
        ("one", ["one?c=0"]),
    ]


async def test_search_without_reranker_keeps_sql_ranked_hits(monkeypatch):
    document = make_document("one", "One", "summary one")
    ranked_rows = [
        {
            "collection_id": "collection",
            "document_id": "one",
            "chunk_id": "one?c=0",
        }
    ]
    session = FakeSession(
        [
            FakeMappingsResult(ranked_rows),
            FakeScalarsResult([document]),
        ]
    )

    @asynccontextmanager
    async def get_session():
        yield session

    context = SimpleNamespace(
        embedding=FakeEmbedding(),
        collections={"collection": SimpleNamespace()},
        get_session=get_session,
        config=SimpleNamespace(get_server_config=lambda: SimpleNamespace()),
    )
    indexer = Indexer(context)
    monkeypatch.setattr(indexer, "summarize_results", _do_nothing)

    results = await indexer.search("collection", "find relevant text", limit=1)

    assert [(item.document_id, [chunk.chunk_id for chunk in chunks]) for item, chunks in results] == [
        ("one", ["one?c=0"]),
    ]


async def _do_nothing(*args, **kwargs):
    del args, kwargs
