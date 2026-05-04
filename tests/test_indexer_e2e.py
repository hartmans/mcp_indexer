import lancedb
import pytest

from mcp_indexer.config import CollectionConfig, ServerConfig
from mcp_indexer.context import Context
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
    await indexer.index_all(index=True, summarize_chunks=False, summarize_documents=False)
    await indexer.index_all(index=False, summarize_chunks=True, summarize_documents=False)
    await indexer.index_all(index=False, summarize_chunks=False, summarize_documents=True)


@pytest.mark.asyncio
async def test_indexer_indexes_text_documents_end_to_end(tmp_path):
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
        "db_uri": str(tmp_path / "db"),
        "min_size": 10,
        "max_size": 1000,
    })
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    context = Context(
        db=lancedb.connect(str(tmp_path / "db")),
        embedding=FakeEmbedding(),
        llm=FakeLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source
    source.build_tables()

    indexer = Indexer(context)
    await run_indexing_pipeline(indexer)

    chunk_table = context.db.open_table("notes")
    meta_table = context.db.open_table("notes_meta")

    chunk_rows = chunk_table.to_pandas().to_dict("records")
    meta_rows = meta_table.to_pandas().to_dict("records")

    assert len(chunk_rows) == 2
    assert len(meta_rows) == 2
    assert {row["document_id"] for row in chunk_rows} == {
        f"{source.id_prefix}alpha.txt",
        f"{source.id_prefix}beta.txt",
    }
    assert {row["title"] for row in meta_rows} == {"alpha.txt", "beta.txt"}
    assert any("alpha" in row["summary"].lower() for row in meta_rows)
    assert any("beta" in row["summary"].lower() for row in meta_rows)
    assert not any(call[0] == "query" for call in context.embedding.calls)

    search_results = await indexer.search("notes", "alpha", limit=2)
    assert search_results
    assert search_results[0].document_id.endswith("alpha.txt")
    assert dict(search_results[0].metadata)["b"] == 0


@pytest.mark.asyncio
async def test_indexer_splits_oversized_semantic_chunks_for_summary_only(tmp_path):
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
        "db_uri": str(tmp_path / "db"),
        "min_size": 1,
        "max_size": 5,
        "max_to_summarize": 10,
    })
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    llm = FakeLlm()
    context = Context(
        db=lancedb.connect(str(tmp_path / "db")),
        embedding=FakeEmbedding(),
        llm=llm,
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source
    source.build_tables()

    indexer = Indexer(context)
    await run_indexing_pipeline(indexer)

    chunk_rows = context.db.open_table("notes").to_pandas().to_dict("records")
    meta_rows = context.db.open_table("notes_meta").to_pandas().to_dict("records")
    summary_rows = context.db.open_table("notes_summary").to_pandas().to_dict("records")
    chunk_rows.sort(key=lambda row: row["chunk_id"])
    summary_rows.sort(key=lambda row: row["summary_span"])

    assert len(chunk_rows) == 4
    assert [row["summary_span"] for row in chunk_rows] == [0, 1, 2, 3]
    assert [row["summary"] for row in summary_rows] == ["aaaaa", "bbbbb", "ccccc", "ddddd"]
    assert len(llm.calls) == 5
    assert llm.calls[-1] == [[
        ("system", "Summarize the document"),
        ("user", "Write no more than two paragraphs to summarize the following document:\n\naaaaa\n\nbbbbb\n\nccccc\n\nddddd"),
    ]]
    assert meta_rows[0]["summary"] == "aaaaa\n\nbbbbb\n\nccccc\n\nddddd"
    assert [call[0] for call in context.embedding.calls] == ["docs", "docs"]


@pytest.mark.asyncio
async def test_indexer_uses_escaped_file_paths_in_document_ids(tmp_path):
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
        "db_uri": str(tmp_path / "db"),
        "min_size": 10,
        "max_size": 1000,
    })
    config_manager = FakeConfigManager("notes", collection_config, server_config)

    context = Context(
        db=lancedb.connect(str(tmp_path / "db")),
        embedding=FakeEmbedding(),
        llm=FakeLlm(),
        config=config_manager,
    )
    source = TextFileSource("notes", context=context, collection_config=collection_config)
    context.collections["notes"] = source
    source.build_tables()

    indexer = Indexer(context)
    await run_indexing_pipeline(indexer)

    chunk_rows = context.db.open_table("notes").to_pandas().to_dict("records")

    assert len(chunk_rows) == 1
    assert chunk_rows[0]["document_id"].endswith("team%27s%20notes/alpha%3F.txt")
