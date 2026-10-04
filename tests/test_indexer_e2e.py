import asyncio
import pytest

from mcp_indexer.config import CollectionConfig, ServerConfig
from mcp_indexer.indexer import Indexer
from mcp_indexer.search import search
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

    def list_collections(self) -> list[str]:
        return [self.collection_id]


async def run_indexing_pipeline(indexer, passes: int = 1):
    """Run indexing pipeline for specified number of passes.
    
    Multiple passes are needed because chunk creation and chunk summarization
    run in parallel within a single pass, so summaries won't be created until
    chunks are committed in a previous pass.
    """
    for _ in range(passes):
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

    # Chunks were created in Postgres during indexing.
    assert len(chunks) == 2

    # After passage testing, Document records are populated with title and embedding from the LLM pass.
    async with test_context.session_factory() as session:
        documents = (await session.execute(select(Document))).scalars().all()
    doc_ids = {d.document_id for d in documents}
    assert "alpha.txt" in doc_ids
    assert "beta.txt" in doc_ids

    # Search must prepare summaries before hydrating its response, including
    # when indexing left documents without summaries.
    results = await search(indexer, "notes", "alpha", limit=2)
    assert results.results
    assert any(chunks for _, chunks in results.results)
    for document, chunks in results.results:
        assert document.summary
        assert all(chunk.summary and chunk.summary.summary for chunk in chunks)


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
    # First pass creates chunks, second pass summarizes them
    await run_indexing_pipeline(indexer, passes=2)

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


@pytest.mark.asyncio
async def test_failed_document_tracking(test_context, tmp_path):
    """Test that failed documents are tracked and skipped in subsequent passes."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text("Alpha document content", encoding="utf-8")
    (docs_dir / "beta.txt").write_text("Beta document content", encoding="utf-8")

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
    from mcp_indexer.models import FailedDocument, Document

    class FailingEmbedding(FakeEmbedding):
        """Embedding that fails on specific documents."""
        def __init__(self, fail_on: list[str]):
            super().__init__()
            self.fail_on = fail_on

        async def __call__(self, texts):
            # Check if any text contains a failing document
            for text in texts:
                for fail_doc in self.fail_on:
                    if fail_doc in text.lower():
                        raise ValueError(f"Simulated failure for document containing: {fail_doc}")
            return await super().__call__(texts)

    # First pass: simulate failure for beta.txt during indexing
    failing_embedding = FailingEmbedding(fail_on=["beta"])
    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=failing_embedding,
        llm=FakeLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source

    indexer = Indexer(context)
    # First pass will log failure for beta.txt but not raise (failures are recorded)
    # Wait a bit for async failure recording to complete
    await run_indexing_pipeline(indexer, passes=1)
    await asyncio.sleep(0.1)

    # Verify beta.txt is in FailedDocument
    async with test_context.session_factory() as session:
        failed_rows = (await session.execute(select(FailedDocument))).scalars().all()
        failed_ids = {row.document_id for row in failed_rows}

    assert "beta.txt" in failed_ids

    # Second pass: create new context with working embedding
    # beta.txt should be skipped due to failure record
    working_context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=FakeEmbedding(),
        llm=FakeLlm(),
        config=config_manager,
    )
    working_source = TextFileSource("notes", context=working_context, collection_config=collection_config)
    working_context.collections["notes"] = working_source

    working_indexer = Indexer(working_context)
    await run_indexing_pipeline(working_indexer, passes=2)

    # Verify alpha.txt was processed but beta.txt was skipped
    async with test_context.session_factory() as session:
        documents = (await session.execute(select(Document))).scalars().all()
        failed_after = (await session.execute(select(FailedDocument))).scalars().all()

    # Only alpha.txt should be in documents
    assert len(documents) == 1
    assert documents[0].document_id == "alpha.txt"
    # beta.txt should still be in FailedDocument (failure records are not cleared)
    assert any(f.document_id == "beta.txt" for f in failed_after)
