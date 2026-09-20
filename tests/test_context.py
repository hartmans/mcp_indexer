import pytest
from types import SimpleNamespace
from mcp_indexer.context import Context
from mcp_indexer.llm import LlmCall, EmbeddingCall
from mcp_indexer.config import ConfigManager


class MockDb:
    def __init__(self):
        self.opened_tables = []

    def open_table(self, name):
        self.opened_tables.append(name)
        return f"table_{name}"


def test_context_get_table(test_context):
    # Simple mocks for other fields
    mock_llm = None
    mock_emb = None
    mock_cfg = None

    ctx = Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=mock_emb, llm=mock_llm, config=mock_cfg)

    # Test that Context works with the new async API
    assert ctx.engine is not None
    assert ctx.session_factory is not None


async def test_build_collections_assigns_configured_reranker(monkeypatch):
    from mcp_indexer.config import CollectionConfig
    from mcp_indexer.context import SOURCE_REGISTRY
    from mcp_indexer.plugins.base import DocumentSource
    import mcp_indexer.rerank

    class TestSource(DocumentSource):
        source_prefix = "context-rerank-test"

    class FakeReranker:
        instances = []

        def __init__(self):
            self.setup_calls = 0
            self.instances.append(self)

        async def setup(self):
            self.setup_calls += 1

    configs = {
        "enabled-one": CollectionConfig.model_validate(
            {
                "collection_id": "enabled-one",
                "tool_prefix": "one",
                "rerank": True,
                "source_blob": {"type": "context-rerank-test"},
            }
        ),
        "disabled": CollectionConfig.model_validate(
            {
                "collection_id": "disabled",
                "tool_prefix": "disabled",
                "rerank": False,
                "source_blob": {"type": "context-rerank-test"},
            }
        ),
        "enabled-two": CollectionConfig.model_validate(
            {
                "collection_id": "enabled-two",
                "tool_prefix": "two",
                "rerank": True,
                "source_blob": {"type": "context-rerank-test"},
            }
        ),
    }
    config = SimpleNamespace(
        list_collections=lambda: list(configs),
        get_collection_config=configs.__getitem__,
    )
    context = Context(
        engine=None,
        session_factory=None,
        embedding=None,
        llm=None,
        config=config,
    )
    monkeypatch.setattr(mcp_indexer.rerank, "QwenReranker", FakeReranker)

    try:
        await context.build_collections()
    finally:
        SOURCE_REGISTRY.pop("context-rerank-test", None)

    assert len(FakeReranker.instances) == 1
    reranker = FakeReranker.instances[0]
    assert reranker.setup_calls == 0
    assert context.collections["enabled-one"].reranker is reranker
    assert context.collections["enabled-two"].reranker is reranker
    assert context.collections["disabled"].reranker is None


async def test_build_collections_selects_llama_cpp_reranker(monkeypatch):
    from mcp_indexer.config import CollectionConfig, RerankConfig
    from mcp_indexer.context import SOURCE_REGISTRY
    from mcp_indexer.plugins.base import DocumentSource
    import mcp_indexer.rerank

    class TestSource(DocumentSource):
        source_prefix = "context-llama-rerank-test"

    class FakeLlamaCppReranker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    collection = CollectionConfig.model_validate(
        {
            "collection_id": "docs",
            "tool_prefix": "docs",
            "rerank": True,
            "source_blob": {"type": "context-llama-rerank-test"},
        }
    )
    rerank_config = RerankConfig(
        plugin="llama_cpp",
        url="http://reranker:8080",
        model="reranker-model",
        request_timeout=12,
    )
    config = SimpleNamespace(
        list_collections=lambda: ["docs"],
        get_collection_config=lambda collection_id: collection,
        get_rerank_config=lambda: rerank_config,
    )
    context = Context(
        engine=None,
        session_factory=None,
        embedding=None,
        llm=None,
        config=config,
    )
    monkeypatch.setattr(
        mcp_indexer.rerank, "LlamaCppReranker", FakeLlamaCppReranker
    )

    try:
        await context.build_collections()
    finally:
        SOURCE_REGISTRY.pop("context-llama-rerank-test", None)

    reranker = context.collections["docs"].reranker
    assert isinstance(reranker, FakeLlamaCppReranker)
    assert reranker.kwargs["url"] == "http://reranker:8080"
    assert reranker.kwargs["model"] == "reranker-model"
    assert reranker.kwargs["request_timeout"] == 12
