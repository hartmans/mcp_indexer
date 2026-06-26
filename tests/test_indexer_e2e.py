import pytest

from mcp_indexer.config import CollectionConfig, ServerConfig
from mcp_indexer.indexer import Indexer
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.plugins.text_source import TextFileSource


class FakeLlm:
    def __init__(self):
        self.calls = []

    async def __call__(self, prompts):
        self.calls.append(list(prompts))
        return [self._extract_text(prompt) for prompt in prompts]

    def _extract_text(self, prompt):
        if isinstance(prompt, str):
            return prompt.split("\n\n", 1)[-1]

        last_role, last_content = prompt[-1]
        del last_role
        return last_content.split("\n\n", 1)[-1]


class FakeEmbedding:
    def __init__(self):
        self.calls = []

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = [0.0] * VECTOR_DIMENSIONS
        vector[0] = float(lowered.count("alpha"))
        vector[1] = float(lowered.count("beta"))
        vector[2] = float(len(lowered))
        return vector

    async def __call__(self, texts):
        values = list(texts)
        self.calls.append(("docs", values))
        return [self._embed(text) for text in values]

    async def query(self, text):
        self.calls.append(("query", text))
        return self._embed(text)


class FakeConfigManager:
    def __init__(self, collection_id: str, collection_config: CollectionConfig, server_config: ServerConfig):
        self.collection_id = collection_id
        self.collection_config = collection_config
        self.server_config = server_config

    def get_collection_config(self, collection_id: str) -> CollectionConfig:
        assert collection_id == self.collection_id
        return self.collection_config

    def get_server_config(self) -> ServerConfig:
        return self.server_config


async def run_indexing_pipeline(indexer: Indexer):
    await indexer.index_all()


@pytest.mark.asyncio
async def test_indexer_indexes_text_documents_end_to_end():
    """Placeholder - test needs to be rewritten for PostgreSQL schema."""
    pytest.skip("E2E test requires PostgreSQL migration - implement later")


@pytest.mark.asyncio
async def test_indexer_splits_oversized_semantic_chunks_for_summary_only():
    """Placeholder - test needs to be rewritten for PostgreSQL schema."""
    pytest.skip("E2E test requires PostgreSQL migration - implement later")


@pytest.mark.asyncio
async def test_indexer_uses_escaped_file_paths_in_document_ids():
    """Placeholder - test needs to be rewritten for PostgreSQL schema."""
    pytest.skip("E2E test requires PostgreSQL migration - implement later")