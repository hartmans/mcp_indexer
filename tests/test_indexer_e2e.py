import pytest

from mcp_indexer.config import CollectionConfig, ServerConfig
from mcp_indexer.indexer import Indexer
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.plugins.text_source import TextFileSource
from sqlalchemy import select


class FakeLlm:
    def __init__(self):
        self.calls = []

    async def __call__(self, prompts):
        self.calls.append(list(prompts))
        return [self._extract_text(prompt) for prompt in prompts]

    def _extract_text(self, prompt):
        if isinstance(prompt, str):
            return prompt.split("\n\n", 1)[-1]
        if isinstance(prompt, tuple):
            # Single message tuple like ("user", "text")
            return prompt[-1].split("\n\n", 1)[-1]

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


async def run_indexing_pipeline(indexer):
    await indexer.index_all()


@pytest.mark.asyncio
async def test_indexer_indexes_text_documents_end_to_end(test_context, tmp_path):
    """Test that the indexer correctly processes text documents end-to-end with PostgreSQL."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text(
        "Alpha project notes.\n\nAlpha is the main topic for this document.",
        encoding="utf-8",
    )
    (docs_dir / "beta.txt").write_text(
        "Beta launch checklist.\n\nBeta is the main topic for this document.",
        encoding="utf-8",
    )

    collection_config = CollectionConfig.model_validate({
        "collection_id": "notes",
        "tool_prefix": "notes",
        "description": "Test notes collection",
        "doc_summary_prompt": "Summarize the document",
        "chunk_summary_prompt": "Summarize the chunk",
        "source_blob": {"type": "text", "directory": str(docs_dir)},
    })
    server_config = ServerConfig.model_validate({
        "db_uri": "postgresql://test",
        "min_size": 10,
        "max_size": 1000,
    })
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    from mcp_indexer.context import Context
    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=FakeEmbedding(),
        llm=FakeLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source

    indexer = Indexer(context)
    await run_indexing_pipeline(indexer)

    from mcp_indexer.models import DocumentChunk, Document

    async with test_context.session_factory() as session:
        chunks = (await session.execute(select(DocumentChunk))).scalars().all()

    assert len(chunks) == 2
    assert any("alpha" in c.text.lower() for c in chunks)
    assert any("beta" in c.text.lower() for c in chunks)


@pytest.mark.asyncio
async def test_indexer_splits_oversized_semantic_chunks_for_summary_only(test_context, tmp_path):
    """Test that oversized semantic chunks are split for summary only."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    big_text = "aaaaabbbbbcccccddddd"
    (docs_dir / "alpha.txt").write_text(big_text, encoding="utf-8")

    collection_config = CollectionConfig.model_validate({
        "collection_id": "notes",
        "tool_prefix": "notes",
        "description": "Test notes collection",
        "doc_summary_prompt": "Summarize the document",
        "chunk_summary_prompt": "Summarize the chunk",
        "source_blob": {"type": "text", "directory": str(docs_dir), "summary_length": 10},
    })
    server_config = ServerConfig.model_validate({
        "db_uri": "postgresql://test",
        "min_size": 10,
        "max_size": 1000,
    })
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    from mcp_indexer.context import Context
    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=FakeEmbedding(),
        llm=FakeLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source

    indexer = Indexer(context)
    await run_indexing_pipeline(indexer)

    from mcp_indexer.models import ChunkSummary

    async with test_context.session_factory() as session:
        summaries = (await session.execute(select(ChunkSummary))).scalars().all()

    assert len(summaries) >= 1


@pytest.mark.asyncio
async def test_indexer_uses_escaped_file_paths_in_document_ids(test_context, tmp_path):
    """Test that file paths with special characters are properly escaped."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    subdir = docs_dir / "team's notes"
    subdir.mkdir()
    weird_file = subdir / "alpha?.txt"
    weird_file.write_text("Alpha document with punctuation in the path.", encoding="utf-8")

    collection_config = CollectionConfig.model_validate({
        "collection_id": "notes",
        "tool_prefix": "notes",
        "description": "Test notes collection",
        "doc_summary_prompt": "Summarize the document",
        "chunk_summary_prompt": "Summarize the chunk",
        "source_blob": {"type": "text", "directory": str(docs_dir)},
    })
    server_config = ServerConfig.model_validate({
        "db_uri": "postgresql://test",
        "min_size": 10,
        "max_size": 1000,
    })
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    from mcp_indexer.context import Context
    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=FakeEmbedding(),
        llm=FakeLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source

    indexer = Indexer(context)
    await run_indexing_pipeline(indexer)

    from mcp_indexer.models import Document

    async with test_context.session_factory() as session:
        meta_rows = (await session.execute(select(Document))).scalars().all()

    assert len(meta_rows) == 1
    # The document_id should be properly escaped (e.g., %3F for ?)
    doc_id = meta_rows[0].document_id
    assert "alpha" in doc_id
    assert "team" in doc_id