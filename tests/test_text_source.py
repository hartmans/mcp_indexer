from pathlib import Path

import pytest

from mcp_indexer.context import Context
from mcp_indexer.plugins.text_source import TextFilePointer, TextFileSource, TextFileSourceConfig


class SimpleCfg:
    def __init__(self, directory, **kwargs):
        self.source_blob = {"directory": directory, **kwargs}
        self.doc_summary_prompt = "Summarize the document"

    def resolve_source_config(self, model):
        return model.model_validate(self.source_blob)


class FakeLlm:
    def __init__(self):
        self.calls = []

    async def __call__(self, prompts):
        self.calls.append(list(prompts))
        return [f"summary:{self._extract_text(prompt)}" for prompt in prompts]

    def _extract_text(self, prompt):
        if isinstance(prompt, str):
            return prompt.splitlines()[-1]

        last_role, last_content = prompt[-1]
        del last_role
        return last_content.split("\n\n", 1)[-1]


@pytest.fixture
def temp_text_env(tmp_path):
    (tmp_path / "test.md").write_text(
        "# Main Title\n\n## Subtitle\n\nThis is a paragraph.\n\nAnother paragraph.",
        encoding="utf-8",
    )
    (tmp_path / "test.rst").write_text(
        "Main Title\n===========\n\nSubtitle\n----------\n\nThis is a paragraph.",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def mock_context(test_context):
    return Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=None, llm=None, config=None)


@pytest.mark.asyncio
async def test_text_source_config_resolution(temp_text_env, mock_context):
    source = TextFileSource("col1", context=mock_context, collection_config=SimpleCfg(temp_text_env))

    assert isinstance(source.source_config, TextFileSourceConfig)
    assert source.source_config.directory == temp_text_env
    assert source.source_config.encoding == "utf-8"
    assert source.source_config.summary_length == 65536


@pytest.mark.asyncio
async def test_markdown_semantic_boundaries(mock_context):
    text = b"Intro text\n\n# Header 1\nContent 1\n\n# Header 2\nContent 2"

    source = TextFileSource("col", context=mock_context, collection_config=SimpleCfg(Path(".")))

    results = []
    async for chunk in source.split_text({}, text, min_size=1, max_size=1000):
        results.append(chunk)

    assert len(results) == 3
    assert b"# Header 1" in "".join(results[1][1]).encode("utf-8")


@pytest.mark.asyncio
async def test_rst_semantic_boundaries(mock_context):
    text = b"Title\n=======\n\nSection 1\n----------\n\nContent"

    source = TextFileSource("col", context=mock_context, collection_config=SimpleCfg(Path(".")))

    results = []
    async for chunk in source.split_text({}, text, min_size=1, max_size=1000):
        results.append(chunk)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_text_file_pointer_retrieval(temp_text_env, mock_context):
    path = temp_text_env / "test.md"
    path.write_text("# Title\n\nContent", encoding="utf-8")

    source = TextFileSource("col", context=mock_context, collection_config=SimpleCfg(temp_text_env))
    source.source_config = TextFileSourceConfig(directory=temp_text_env)

    pointer = source.fetch_document("test.md")
    assert isinstance(pointer, TextFilePointer)

    meta = await pointer.get_metadata()
    assert meta["title"] == "test.md"

    chunks = []
    async for chunk in pointer.get_chunks(min_size=1, max_size=100):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join("".join(c[1]) for c in chunks)
    assert "# Title" in full_text


@pytest.mark.asyncio
async def test_text_file_source_get_document_summary_summarizes_small_documents(tmp_path, test_context):
    path = tmp_path / "small.txt"
    path.write_text("Short document body", encoding="utf-8")

    llm = FakeLlm()
    context = Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=None, llm=llm, config=None)
    source = TextFileSource("col", context=context, collection_config=SimpleCfg(tmp_path, summary_length=100))

    pointer = source.fetch_document("small.txt")
    meta = await pointer.get_metadata()
    summary = await source.get_document_summary(pointer.document_id)

    assert meta["title"] == "small.txt"
    assert "summary" not in meta
    assert summary == "summary:Short document body"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_text_file_pointer_get_metadata_skips_summary_for_large_documents(tmp_path, test_context):
    path = tmp_path / "large.txt"
    path.write_text("X" * 20, encoding="utf-8")

    llm = FakeLlm()
    context = Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=None, llm=llm, config=None)
    source = TextFileSource("col", context=context, collection_config=SimpleCfg(tmp_path, summary_length=10))

    meta = await source.fetch_document("large.txt").get_metadata()

    assert "summary" not in meta
    assert llm.calls == []