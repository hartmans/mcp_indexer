import asyncio
import os
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select

from mcp_indexer.config import ConfigManager, ServerConfig, CollectionConfig
from mcp_indexer.models import Base


def _load_test_env() -> None:
    """Load simple KEY=VALUE entries from the repository's .env without overriding CI."""
    env_path = Path(__file__).parents[1] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


_load_test_env()


class FakeConfigManager(ConfigManager):
    """Drop-in ConfigManager that accepts a raw dict instead of file paths."""

    def __init__(self, config: dict):
        self._raw_config = config
    
    def get_server_config(self) -> ServerConfig:
        pass  # type: ignore[return-value]


@pytest.fixture(scope="session")
def pg_url():
    """Get the PostgreSQL connection URL for tests from .env or the environment."""
    return os.environ["MCP_INDEXER_TEST_DB_URL"]


@pytest.fixture
async def test_context(pg_url):
    """Build a Context with a test database."""
    # Create engine and session factory
    engine = create_async_engine(pg_url)

    # Initialize the database
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    session_factory = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create a minimal server config
    server_config = ServerConfig.model_validate({
        "db_uri": pg_url,
        "min_size": 10,
        "max_size": 1000,
    })

    # Create a minimal Context-like object
    class TestContext:
        def __init__(self):
            self.engine = engine
            self.session_factory = session_factory
            self.config = type('Config', (), {'server_config': server_config})()

    ctx = TestContext()
    yield ctx

    # Cleanup - drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
def fake_config():
    """Provide a FakeConfigManager for testing without file I/O."""
    yield FakeConfigManager({"server": {}, "collections": {}})
