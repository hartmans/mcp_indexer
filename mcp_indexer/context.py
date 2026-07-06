from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from mcp_indexer.config import ConfigManager
from mcp_indexer.llm import LlmCall, EmbeddingCall
from mcp_indexer.models import Base

# Global registry for DocumentSource plugins
SOURCE_REGISTRY: dict[str, type] = {}

if TYPE_CHECKING:
    from .plugins.base import DocumentSource


@dataclass
class Context:
    """
    Provides shared resources for DocumentSource plugins and the indexer.
    """

    engine: AsyncEngine
    session_factory: sessionmaker[AsyncSession]
    embedding: EmbeddingCall
    llm: LlmCall
    config: ConfigManager
    collections: dict[str, "DocumentSource"] = field(default_factory=dict)

    @staticmethod
    def build_context(config_path: str | list[str]) -> "Context":
        """
        Builds a Context object from a configuration file.
        """
        config_manager = ConfigManager(config_path)
        server_config = config_manager.get_server_config()

        llm_call = LlmCall(
            batch_size=server_config.llm_batch_size,
            request_timeout=server_config.llm_request_timeout,
            timeout_retries=server_config.llm_timeout_retries,
            **server_config.llm,
        )
        embedding_config = server_config.embedding.copy()
        if "model_provider" in embedding_config and "provider" not in embedding_config:
            embedding_config["provider"] = embedding_config.pop("model_provider")
        embedding_call = EmbeddingCall(
            batch_size=server_config.embedding_batch_size,
            request_timeout=server_config.embedding_request_timeout,
            timeout_retries=server_config.embedding_timeout_retries,
            **embedding_config,
        )

        # Parse db_uri - supports PostgreSQL connection strings
        db_uri = server_config.db_uri

        # Convert file path to PostgreSQL if it looks like a file path
        if db_uri.startswith("~") or db_uri.startswith("/") or db_uri.startswith("."):
            # This is a file path - assume it's a PostgreSQL database file
            # Expand tilde and convert to PostgreSQL format
            import os
            db_uri = os.path.expanduser(db_uri)
            # Assume it's a PostgreSQL database with pgvector extension
            # Default to localhost:5432 with 'indexer' database
            db_uri = f"postgresql+asyncpg://postgres:postgres@localhost:5432/indexer"

        # Create async engine
        engine = create_async_engine(
            db_uri,
            echo=False,
            pool_pre_ping=True,
        )

        # Create session factory
        session_factory = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        ctx = Context(
            engine=engine,
            session_factory=session_factory,
            embedding=embedding_call,
            llm=llm_call,
            config=config_manager,
        )

        return ctx

    async def build_collections(self) -> None:
        """
        Builds the collections mapping based on the configuration.
        Raises ValueError if a collection cannot be initialized.
        """
        import mcp_indexer.plugins  # noqa: F401

        for collection_id in self.config.list_collections():
            col_config = self.config.get_collection_config(collection_id)

            prefix = col_config.source_blob.get("type")
            if not prefix:
                raise ValueError(
                    f"Collection '{collection_id}' is missing 'type' in source_config."
                )

            cls = SOURCE_REGISTRY.get(prefix)
            if not cls:
                raise ValueError(
                    f"No DocumentSource plugin registered for prefix '{prefix}' (collection '{collection_id}')."
                )

            self.collections[collection_id] = cls(
                collection_id=collection_id,
                context=self,
                collection_config=col_config,
            )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        async with self.session_factory() as session:
            try:
                yield session
            finally:
                await session.close()

    async def init_db(self) -> None:
        """Initialize the database schema."""
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        async with self.engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)


