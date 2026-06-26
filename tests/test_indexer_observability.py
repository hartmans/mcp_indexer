import asyncio
import json
import logging
from datetime import datetime

import pytest

import mcp_indexer.indexer as indexer_module
from mcp_indexer.config import CollectionConfig, ServerConfig
from mcp_indexer.context import Context
from mcp_indexer.indexer import DEBUG_STATS_FILENAME, Indexer
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.plugins.base import ChunkSummary, Document, DocumentChunk
from mcp_indexer.plugins.text_source import TextFileSource


class FastLlm:
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


class BlockingLlm(FastLlm):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, prompts):
        self.calls.append(list(prompts))
        if not self.started.is_set():
            self.started.set()
            await self.release.wait()
        return [self._extract_text(prompt) for prompt in prompts]


class FakeEmbedding:
    def __init__(self):
        self.calls = []

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * VECTOR_DIMENSIONS
        vector[0] = float(len(text))
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


def build_collection_config(docs_dir, summary_length: int = 65536) -> CollectionConfig:
    return CollectionConfig.model_validate({
        "collection_id": "notes",
        "tool_prefix": "notes",
        "description": "Test notes collection",
        "doc_summary_prompt": "Summarize the document",
        "chunk_summary_prompt": "Summarize the chunk",
        "source_blob": {
            "type": "text",
            "directory": str(docs_dir),
            "summary_length": summary_length,
        },
    })


def build_server_config(tmp_path) -> ServerConfig:
    return ServerConfig.model_validate({
        "db_uri": str(tmp_path / "db"),
        "min_size": 5,
        "max_size": 1000,
        "max_to_summarize": 1000,
    })


@pytest.mark.asyncio
async def test_indexer_debug_writes_document_stats_jsonl(test_context, tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text("Alpha project notes live here.", encoding="utf-8")

    collection_config = build_collection_config(docs_dir, summary_length=10)
    server_config = build_server_config(tmp_path)
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=FakeEmbedding(),
        llm=FastLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source

    monkeypatch.chdir(tmp_path)

    indexer = Indexer(context, debug=True)
    await run_indexing_pipeline(indexer)

    stats_path = tmp_path / DEBUG_STATS_FILENAME
    lines = stats_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    stats = [json.loads(line) for line in lines]
    index_stats = next(item for item in stats if item["operation"] == "index")
    chunk_summary_stats = next(item for item in stats if item["operation"] == "chunk_summary")
    document_summary_stats = next(item for item in stats if item["operation"] == "document_summary")

    assert index_stats["document_id"].endswith("alpha.txt")
    assert index_stats["semantic_chunks"] == 1
    assert index_stats["embedding_chunks"] == 1
    assert index_stats["summary_spans"] == 0
    assert chunk_summary_stats["document_id"].endswith("alpha.txt")
    assert document_summary_stats["document_id"].endswith("alpha.txt")


@pytest.mark.asyncio
async def test_indexer_logs_running_document_stats_and_stalled_monitor(test_context, tmp_path, monkeypatch, caplog):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text("Alpha project notes live here.", encoding="utf-8")

    collection_config = build_collection_config(docs_dir, summary_length=10)
    server_config = build_server_config(tmp_path)
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    llm = BlockingLlm()
    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=FakeEmbedding(),
        llm=llm,
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source

    monkeypatch.setattr(indexer_module, "MONITOR_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(indexer_module, "MONITOR_STALL_THRESHOLD_SECONDS", 0.02)

    values = iter([0.0, 0.05, 0.05, 0.05])

    def fake_monotonic():
        try:
            return next(values)
        except StopIteration:
            return 0.05

    monkeypatch.setattr(indexer_module, "monitor_time", fake_monotonic)
    caplog.set_level(logging.INFO, logger="mcp_indexer.indexer")

    indexer = Indexer(context)
    task = asyncio.create_task(indexer.index_all())
    await llm.started.wait()
    await asyncio.sleep(0.03)
    llm.release.set()
    await task
    await indexer.wait_for_idle()

    messages = [record.message for record in caplog.records]
    assert any("Still running chunk_summary document_id=" in message for message in messages)


@pytest.mark.asyncio
async def test_indexer_does_not_log_short_lived_document_on_first_monitor_cycle(test_context, tmp_path, monkeypatch, caplog):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text("Alpha project notes live here.", encoding="utf-8")

    collection_config = build_collection_config(docs_dir, summary_length=10)
    server_config = build_server_config(tmp_path)
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    llm = BlockingLlm()
    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=FakeEmbedding(),
        llm=llm,
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source

    monkeypatch.setattr(indexer_module, "MONITOR_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(indexer_module, "MONITOR_STALL_THRESHOLD_SECONDS", 1.0)
    caplog.set_level(logging.INFO, logger="mcp_indexer.indexer")

    indexer = Indexer(context)
    task = asyncio.create_task(indexer.index_all())
    await llm.started.wait()
    llm.release.set()
    await task
    await indexer.wait_for_idle()

    messages = [record.message for record in caplog.records]
    assert not any("Still running" in message for message in messages)


@pytest.mark.asyncio
async def test_run_upsirts_batches_units_by_table(test_context):
    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=FakeEmbedding(),
        llm=FastLlm(),
        config=None,
    )
    source = TextFileSource("notes", context=context, collection_config=CollectionConfig.model_validate({
        "collection_id": "notes",
        "tool_prefix": "notes",
        "description": "Test",
        "doc_summary_prompt": "Summarize",
        "chunk_summary_prompt": "Summarize",
        "source_blob": {"type": "text", "directory": "."},
    }))

    now = datetime.now()

    indexer = Indexer(context)
    indexer.chunk_upsirt.extend([
        (source, [
            DocumentChunk(
                collection_id="notes",
                document_id="alpha.txt",
                order=0,
                chunk_id="alpha.txt?c=0",
                embedding=[1.0] * VECTOR_DIMENSIONS,
                summary_span=0,
                metadata_str=json.dumps({"b": 0, "e": 5}),
            ),
        ]),
    ])
    indexer.summary_upsirt.extend([
        (source, [ChunkSummary(collection_id="notes", document_id="alpha.txt", summary_span=0, summary="alpha")]),
    ])
    indexer.document_upsirt.extend([
        (source, Document(
            collection_id="notes",
            document_id="alpha.txt",
            title="alpha",
            title_strength=0,
            last_modified=now,
            summary="alpha summary",
            embedding=[3.0] * VECTOR_DIMENSIONS,
            keywords=[],
            metadata_str="{}",
        )),
    ])

    task = await indexer.run_upsirts()
    assert task is not None
    await task

    # Verify the data was inserted correctly
    async with test_context.session_factory() as session:
        chunks = (await session.execute(
            indexer_module.select(DocumentChunk)
            .where(DocumentChunk.collection_id == "notes")
        )).scalars().all()

    assert len(chunks) == 1
    assert chunks[0].document_id == "alpha.txt"


@pytest.mark.asyncio
async def test_run_upsirts_reuses_single_running_task(test_context, monkeypatch):
    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=FakeEmbedding(),
        llm=FastLlm(),
        config=None,
    )
    source = TextFileSource("notes", context=context, collection_config=CollectionConfig.model_validate({
        "collection_id": "notes",
        "tool_prefix": "notes",
        "description": "Test",
        "doc_summary_prompt": "Summarize",
        "chunk_summary_prompt": "Summarize",
        "source_blob": {"type": "text", "directory": "."},
    }))

    indexer = Indexer(context)
    release = asyncio.Event()
    started = asyncio.Event()
    original_execute_chunks = indexer._execute_chunk_upsirts

    async def blocking_execute_chunks(work):
        started.set()
        await release.wait()
        await original_execute_chunks(work)

    monkeypatch.setattr(indexer, "_execute_chunk_upsirts", blocking_execute_chunks)

    indexer.chunk_upsirt.append((source, [
        DocumentChunk(
            collection_id="notes",
            document_id="alpha.txt",
            order=0,
            chunk_id="alpha.txt?c=0",
            embedding=[1.0] * VECTOR_DIMENSIONS,
            summary_span=None,
            metadata_str=json.dumps({"b": 0, "e": 5}),
        ),
    ]))

    first_task = await indexer.run_upsirts()
    assert first_task is not None
    await started.wait()

    indexer.document_upsirt.append((source, Document(
        collection_id="notes",
        document_id="alpha.txt",
        title="alpha",
        title_strength=0,
        last_modified=datetime.now(),
        summary="alpha summary",
        embedding=[1.0] * VECTOR_DIMENSIONS,
        keywords=[],
        metadata_str="{}",
    )))

    second_task = await indexer.run_upsirts()
    assert second_task is first_task

    release.set()
    await first_task
    assert indexer._upsirt_task is None