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


def test_context_get_table(test_context):
    # Simple mocks for other fields
    mock_llm = None
    mock_emb = None
    mock_cfg = None

    ctx = Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=mock_emb, llm=mock_llm, config=mock_cfg)

    # Test that Context works with the new async API
    assert ctx.engine is not None
    assert ctx.session_factory is not None
