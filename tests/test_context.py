import pytest
from mcp_indexer.context import Context
from mcp_indexer.llm import LlmCall, EmbeddingCall
from mcp_indexer.config import ConfigManager

class MockDb:
    def __init__(self):
        self.opened_tables = []
    def open_table(self, name):
        self.opened_tables.append(name)
        return f"table_{name}"

def test_build_context(monkeypatch):
    # Mock lancedb.connect to avoid actual DB creation
    monkeypatch.setattr("lancedb.connect", lambda uri: "mock_db")
    
    # Mock LlmCall and EmbeddingCall to avoid actual LLM/Embedding calls
    class MockLlmCall(LlmCall):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = "mock_llm_model"

    class MockEmbeddingCall(EmbeddingCall):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = "mock_embedding_model"

    monkeypatch.setattr("mcp_indexer.context.LlmCall", MockLlmCall)
    monkeypatch.setattr("mcp_indexer.context.EmbeddingCall", MockEmbeddingCall)

    config_path = "tests/test_config.toml"
    ctx = Context.build_context(config_path)

    assert ctx.db == "mock_db"
    assert ctx.config is not None
    assert isinstance(ctx.llm, MockLlmCall)
    assert isinstance(ctx.embedding, MockEmbeddingCall)
    
    # Verify the config was passed correctly to the calls
    assert ctx.llm.kwargs["model"] == "gemma4:31b"
    assert ctx.llm.kwargs["model_provider"] == "ollama"
    assert ctx.llm.kwargs["batch_size"] == 7
    assert ctx.llm.kwargs["request_timeout"] == 400.0
    assert ctx.llm.kwargs["timeout_retries"] == 20
    assert ctx.embedding.kwargs["model"] == "qwen3-embedding:4b"
    assert ctx.embedding.kwargs["provider"] == "ollama"
    assert "model_provider" not in ctx.embedding.kwargs
    assert ctx.embedding.kwargs["batch_size"] == 13
    assert ctx.embedding.kwargs["request_timeout"] == 400.0
    assert ctx.embedding.kwargs["timeout_retries"] == 20

def test_context_get_table():
    mock_db = MockDb()
    # Simple mocks for other fields
    mock_llm = None
    mock_emb = None
    mock_cfg = None
    
    ctx = Context(db=mock_db, embedding=mock_emb, llm=mock_llm, config=mock_cfg)
    
    table = ctx.get_table("my_table")
    assert table == "table_my_table"
    assert "my_table" in mock_db.opened_tables
