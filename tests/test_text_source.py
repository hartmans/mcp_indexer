from datetime import datetime
from pathlib import Path

import pytest

from mcp_indexer.context import Context
from mcp_indexer.plugins.base import ContentBridge
from mcp_indexer.plugins.file_source import FileSource, FileBridge, FileSourceConfig, MARKDOWN_POLICY, TEXT_POLICY
from mcp_indexer.plugins.text_document import TextDocument


class SimpleCfg:
    def __init__(self, directory, **kwargs):
        self.source_blob = {"directory": directory, **kwargs}
        self.doc_summary_prompt = "Summarize the document"
        self.plugin = {}

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


class FakeBridge(ContentBridge):
    """Test-only non-file bridge: serves held bytes, no filesystem access."""

    def __init__(self, content: bytes):
        self._content = content

    async def read_bytes(self) -> bytes:
        return self._content

    async def read_byte_range(self, start: int, end: int) -> bytes:
        return self._content[start:end]

    def path(self):
        raise NotImplementedError("FakeBridge has no path")


@pytest.fixture
def mock_context(test_context):
    return Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=None, llm=None, config=None)


@pytest.fixture
def fake_source(mock_context, tmp_path):
    return FileSource("col", context=mock_context, collection_config=SimpleCfg(tmp_path))


def make_text_document(source, content: bytes, policy=None, metadata=None) -> TextDocument:
    return TextDocument(
        "doc",
        source,
        FakeBridge(content),
        policy if policy is not None else dict(MARKDOWN_POLICY),
        datetime.now(),
        metadata=metadata if metadata is not None else {"title": "doc"},
    )


@pytest.mark.asyncio
async def test_text_document_with_fake_bridge_and_policy(fake_source, monkeypatch):
    """Text interpretation works from a supplied policy and non-file
    content: no filesystem access is required."""
    def no_path_opened(*args, **kwargs):
        raise AssertionError("Text interpretation must not touch the filesystem")
    monkeypatch.setattr(Path, "read_bytes", no_path_opened)
    monkeypatch.setattr(FileBridge, "read_bytes", no_path_opened)

    text = "Intro\n\n# A\nalpha body\n\n# B\nbeta body"
    doc = make_text_document(fake_source, text.encode("utf-8"), policy=MARKDOWN_POLICY)

    meta = await doc.get_metadata()
    assert meta["title"] == "doc"
    assert meta["last_modified"] == str(doc.last_modified)

    chunks = []
    async for chunk in doc.get_chunks(min_size=1, max_size=1000):
        chunks.append(chunk)
    assert len(chunks) == 3
    assert "".join("".join(c[1]) for c in chunks) == text
    for chunk_metadata, candidates in chunks:
        assert await doc.fetch_chunk(chunk_metadata) == "".join(candidates)


@pytest.mark.asyncio
async def test_plain_text_fallback_cuts_on_separator_lines(fake_source):
    """A line of common separator punctuation is a semantic boundary in the
    plain-text fallback."""
    doc = make_text_document(
        fake_source, "Before\n\n===\n\nAfter".encode("utf-8"), policy=TEXT_POLICY,
    )
    chunks = [c async for c in doc.get_chunks(1, 1000)]
    assert len(chunks) == 2
    assert "Before" in "".join(chunks[0][1])
    assert "After" in "".join(chunks[1][1])


@pytest.mark.asyncio
async def test_default_policy_applies_when_policy_omits_boundaries(fake_source):
    doc = make_text_document(fake_source, "A\n\nB\n\nC".encode(), policy={"handler": "text"})
    assert doc.semantic_boundary_regexps == ()
    assert doc.embedding_boundary_regexps == (r"\n\s*\n",)
    chunks = [c async for c in doc.get_chunks(1, 1000)]
    # Paragraph breaks still cut embedding candidates.
    assert len(chunks[0][1]) == 3


@pytest.mark.asyncio
async def test_multibyte_byte_range_round_trip(fake_source):
    """Byte-range reads address bytes, not characters, for multibyte text."""
    text = "# Café\n\nÉlan and 中文 content.\n\nMore text."
    raw = text.encode("utf-8")
    assert len(raw) != len(text)  # guard: the content is actually multibyte
    doc = make_text_document(fake_source, raw, policy=MARKDOWN_POLICY)

    chunks = [c async for c in doc.get_chunks(1, 1000)]
    assert "".join("".join(cands) for _m, cands in chunks) == text
    for meta, cands in chunks:
        # The stored b/e byte span selects exactly this chunk from the raw
        # bytes.
        assert raw[meta["b"]:meta["e"]] == "".join(cands).encode("utf-8")
        assert await doc.fetch_chunk(meta) == "".join(cands)


@pytest.mark.asyncio
@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n", b"\r"])
async def test_byte_offsets_hold_across_newline_conventions(fake_source, line_ending):
    """b/e must hold regardless of the content's newline convention:
    content is read as bytes and decoded, never with newline translation."""
    lines = [b"# Title", b"Body one", b"## Section", b"Body two"]
    content = line_ending.join(lines)
    doc = make_text_document(fake_source, content, policy=MARKDOWN_POLICY)

    chunks = [c async for c in doc.get_chunks(1, 10_000)]
    assert chunks
    assert "".join("".join(cands) for _m, cands in chunks) == content.decode()
    for meta, cands in chunks:
        assert content[meta["b"]:meta["e"]] == "".join(cands).encode("utf-8")
        assert await doc.fetch_chunk(dict(meta)) == "".join(cands)


@pytest.mark.asyncio
async def test_text_file_pointer_retrieval(tmp_path, mock_context):
    (tmp_path / "test.md").write_text("# Title\n\nContent", encoding="utf-8")

    source = FileSource("col", context=mock_context, collection_config=SimpleCfg(tmp_path))
    source.source_config = FileSourceConfig(directory=tmp_path)

    pointer = await source.fetch_document("test.md")
    assert isinstance(pointer, TextDocument)

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
    (tmp_path / "small.txt").write_text("Short document body", encoding="utf-8")

    llm = FakeLlm()
    context = Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=None, llm=llm, config=None)
    source = FileSource("col", context=context, collection_config=SimpleCfg(tmp_path, summary_length=100))

    pointer = await source.fetch_document("small.txt")
    meta = await pointer.get_metadata()
    summary = await pointer.get_document_summary()

    assert meta["title"] == "small.txt"
    assert "summary" not in meta
    assert summary == "summary:Short document body"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_text_file_pointer_get_metadata_skips_summary_for_large_documents(tmp_path, test_context):
    (tmp_path / "large.txt").write_text("X" * 20, encoding="utf-8")

    llm = FakeLlm()
    context = Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=None, llm=llm, config=None)
    source = FileSource("col", context=context, collection_config=SimpleCfg(tmp_path, summary_length=10))

    pointer = await source.fetch_document("large.txt")
    meta = await pointer.get_metadata()

    assert "summary" not in meta
    assert llm.calls == []
