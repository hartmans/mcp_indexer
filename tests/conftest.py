import pytest
from mcp_indexer.context import Context
from mcp_indexer.config import ConfigManager


@pytest.fixture
def test_context():
    """Build a Context with the configured database."""
    config_path = "tests/test_config.toml"
    ctx = Context.build_context(config_path)
    yield ctx