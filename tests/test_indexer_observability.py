import asyncio
import json
import logging
from datetime import datetime

import lancedb
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


class CountingMergeInsert:
    def __init__(self, builder, counter):
        self._builder = builder
        self._counter = counter

    def when_matched_update_all(self):
        self._builder = self._builder.when_matched_update_all()
        return self

    def when_not_matched_insert_all(self):
        self._builder = self._builder.when_not_matched_insert_all()
        return self

    def when_not_matched_by_source_delete(self, condition):
        self._builder = self._builder.when_not_matched_by_source_delete(condition)
        return self

    def execute(self, rows):
        self._counter["execute_calls"] += 1
        self._counter["row_counts"].append(len(rows))
        return self._builder.execute(rows)


class CountingTable:
    def __init__(self, table):
        self._table = table
        self.merge_insert_counter = {"execute_calls": 0, "row_counts": []}
        self.add_calls: list[int] = []
        self.delete_calls: list[str] = []

    def merge_insert(self, keys):
        return CountingMergeInsert(self._table.merge_insert(keys), self.merge_insert_counter)

    def add(self, rows):
        self.add_calls.append(len(rows))
        return self._table.add(rows)

    def delete(self, condition):
        self.delete_calls.append(condition)
        return self._table.delete(condition)

    def __getattr__(self, name):
        return getattr(self._table, name)


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
async def test_indexer_debug_writes_document_stats_jsonl(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text("Alpha project notes live here.", encoding="utf-8")

    collection_config = build_collection_config(docs_dir, summary_length=10)
    server_config = build_server_config(tmp_path)
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    context = Context(
        db=lancedb.connect(str(tmp_path / "db")),
        embedding=FakeEmbedding(),
        llm=FastLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source
    source.build_tables()

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
async def test_indexer_logs_running_document_stats_and_stalled_monitor(tmp_path, monkeypatch, caplog):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text("Alpha project notes live here.", encoding="utf-8")

    collection_config = build_collection_config(docs_dir, summary_length=10)
    server_config = build_server_config(tmp_path)
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    llm = BlockingLlm()
    context = Context(
        db=lancedb.connect(str(tmp_path / "db")),
        embedding=FakeEmbedding(),
        llm=llm,
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source
    source.build_tables()

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
async def test_indexer_does_not_log_short_lived_document_on_first_monitor_cycle(tmp_path, monkeypatch, caplog):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text("Alpha project notes live here.", encoding="utf-8")

    collection_config = build_collection_config(docs_dir, summary_length=10)
    server_config = build_server_config(tmp_path)
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    llm = BlockingLlm()
    context = Context(
        db=lancedb.connect(str(tmp_path / "db")),
        embedding=FakeEmbedding(),
        llm=llm,
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source
    source.build_tables()

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
async def test_run_upsirts_batches_units_by_table(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text("Alpha project notes live here.", encoding="utf-8")

    collection_config = build_collection_config(docs_dir, summary_length=10)
    server_config = build_server_config(tmp_path)
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    context = Context(
        db=lancedb.connect(str(tmp_path / "db")),
        embedding=FakeEmbedding(),
        llm=FastLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source
    source.build_tables()

    source.chunk_table = CountingTable(source.chunk_table)
    source.meta_table = CountingTable(source.meta_table)
    source.summary_table = CountingTable(source.summary_table)

    indexer = Indexer(context)
    now = datetime.now()
    source_ref = context.collections["notes"]

    indexer.chunk_upsirt.extend([
        (source_ref, [
            DocumentChunk(
                document_id="alpha.txt",
                order=0,
                chunk_id="alpha.txt?c=0",
                embedding=[1.0] * VECTOR_DIMENSIONS,
                summary_span=0,
                metadata={"b": 0, "e": 5},
            ),
        ]),
        (source_ref, [
            DocumentChunk(
                document_id="beta.txt",
                order=0,
                chunk_id="beta.txt?c=0",
                embedding=[2.0] * VECTOR_DIMENSIONS,
                summary_span=0,
                metadata={"b": 0, "e": 4},
            ),
        ]),
    ])
    indexer.summary_upsirt.extend([
        (source_ref, [ChunkSummary(document_id="alpha.txt", summary_span=0, summary="alpha")]),
        (source_ref, [ChunkSummary(document_id="beta.txt", summary_span=0, summary="beta")]),
    ])
    indexer.document_upsirt.extend([
        (source_ref, Document(
            document_id="alpha.txt",
            title="alpha",
            title_strength=0,
            last_modified=now,
            summary="alpha summary",
            embedding=[3.0] * VECTOR_DIMENSIONS,
            keywords=[],
        )),
        (source_ref, Document(
            document_id="beta.txt",
            title="beta",
            title_strength=0,
            last_modified=now,
            summary="beta summary",
            embedding=[4.0] * VECTOR_DIMENSIONS,
            keywords=[],
        )),
    ])

    task = await indexer.run_upsirts()
    assert task is not None
    await task

    assert source.chunk_table.merge_insert_counter["execute_calls"] == 1
    assert source.chunk_table.merge_insert_counter["row_counts"] == [2]
    assert len(source.summary_table.delete_calls) == 2
    assert source.summary_table.add_calls == [2]
    assert source.meta_table.merge_insert_counter["execute_calls"] == 1
    assert source.meta_table.merge_insert_counter["row_counts"] == [2]
    assert indexer.chunk_upsirt == []
    assert indexer.summary_upsirt == []
    assert indexer.document_upsirt == []


@pytest.mark.asyncio
async def test_run_upsirts_reuses_single_running_task(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "alpha.txt").write_text("Alpha project notes live here.", encoding="utf-8")

    collection_config = build_collection_config(docs_dir, summary_length=10)
    server_config = build_server_config(tmp_path)
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    context = Context(
        db=lancedb.connect(str(tmp_path / "db")),
        embedding=FakeEmbedding(),
        llm=FastLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source
    source.build_tables()

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
            document_id="alpha.txt",
            order=0,
            chunk_id="alpha.txt?c=0",
            embedding=[1.0] * VECTOR_DIMENSIONS,
            summary_span=None,
            metadata={"b": 0, "e": 5},
        ),
    ]))

    first_task = await indexer.run_upsirts()
    assert first_task is not None
    await started.wait()

    indexer.document_upsirt.append((source, Document(
        document_id="alpha.txt",
        title="alpha",
        title_strength=0,
        last_modified=datetime.now(),
        summary="alpha summary",
        embedding=[1.0] * VECTOR_DIMENSIONS,
        keywords=[],
    )))

    second_task = await indexer.run_upsirts()
    assert second_task is first_task

    release.set()
    await first_task
    assert indexer._upsirt_task is None
