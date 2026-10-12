from pathlib import Path

import pytest

from mcp_indexer.context import Context
from mcp_indexer.plugins.file_source import FileSource, FileSourceConfig, FileSourcePointer


class ConcreteFileSource(FileSource):
    semantic_boundary_regexps = (r"\n={3,}\n",)
    embedding_boundary_regexps = (r"\n\s*\n",)

    def fetch_document(self, document_id: str) -> FileSourcePointer:
        relative_path_str = self.decode_document_path(document_id)
        absolute_path = self.source_config.directory / relative_path_str
        return FileSourcePointer(self, document_id, absolute_path)


class SimpleCfg:
    def __init__(self, directory):
        self.source_blob = {"directory": directory}

    def resolve_source_config(self, model):
        return model.model_validate(self.source_blob)


@pytest.fixture
def temp_file_env(tmp_path):
    (tmp_path / "doc1.txt").write_text("Title 1\n=======\n\nParagraph 1\n\nParagraph 2", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Title 2\n-------\n\nSubtitle\n-----\n\nParagraph 1", encoding="utf-8")
    return tmp_path


@pytest.fixture
def mock_context(test_context):
    return Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=None, llm=None, config=None)


@pytest.mark.asyncio
async def test_file_source_discovery(temp_file_env, mock_context):
    source = ConcreteFileSource("test_col", context=mock_context, collection_config=SimpleCfg(temp_file_env))

    docs = []
    async for doc in source.get_documents():
        docs.append(doc)

    assert len(docs) == 2
    assert any("doc1.txt" in doc.document_id for doc in docs)
    assert any("doc2.txt" in doc.document_id for doc in docs)


@pytest.mark.asyncio
async def test_split_text_rst_headers(mock_context):
    text = "Title\n=======\nSubtitle\n--------\nParagraph"

    source = ConcreteFileSource("col", context=mock_context, collection_config=SimpleCfg(Path(".")))

    results = []
    async for chunk in source.split_text({}, text, min_size=1, max_size=1000):
        results.append(chunk)

    assert len(results) >= 2

    results_large = []
    async for chunk in source.split_text({}, text, min_size=50, max_size=1000):
        results_large.append(chunk)

    assert len(results_large) < len(results)


@pytest.mark.asyncio
async def test_file_source_pointer_retrieval(temp_file_env, mock_context):
    path = temp_file_env / "doc1.txt"
    path.write_text("Semantic 1\n\nSemantic 2", encoding="utf-8")

    source = ConcreteFileSource("col", context=mock_context, collection_config=SimpleCfg(temp_file_env))
    source.source_config = FileSourceConfig(directory=temp_file_env)

    pointer = source.fetch_document("doc1.txt")

    chunks = []
    async for chunk_info in pointer.get_chunks(min_size=1, max_size=100):
        chunks.append(chunk_info)

    assert len(chunks) == 1
    meta, _ = chunks[0]
    text = await pointer.fetch_chunk(meta)
    assert "Semantic 1" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n", b"\r"])
async def test_byte_offsets_hold_across_newline_conventions(tmp_path, mock_context, line_ending):
    """b/e are byte offsets into the file, so they must hold regardless of
    the file's newline convention. This is the guard for reading via
    read_bytes().decode() rather than read_text(): universal-newline
    translation (CRLF/CR -> LF) would make the decoded text -- and hence
    the offsets computed from it -- disagree with the raw file bytes.
    """
    lines = [b"A one", b"===", b"B two", b"===", b"C three"]
    content = line_ending.join(lines)
    (tmp_path / "nl.txt").write_bytes(content)

    source = ConcreteFileSource(
        "col", context=mock_context, collection_config=SimpleCfg(tmp_path),
    )
    pointer = source.fetch_document("nl.txt")

    chunks = []
    async for meta, cands in pointer.get_chunks(min_size=1, max_size=10_000):
        chunks.append((meta, cands))
    assert chunks, "expected at least one semantic chunk"

    # Whole-document round-trip: the decoded candidates must reproduce the
    # file byte-for-byte. A read_text() with newline translation would
    # shorten CRLF/CR content and fail this.
    assert "".join("".join(cands) for _m, cands in chunks) == content.decode()

    # Per chunk: the stored b/e must select exactly this chunk's text from
    # the raw file bytes.
    for meta, cands in chunks:
        assert content[meta["b"]:meta["e"]].decode() == "".join(cands)

    # Contiguity: the chunks partition the file with no gaps or overlaps.
    ordered = sorted(chunks, key=lambda c: c[0]["b"])
    assert ordered[0][0]["b"] == 0
    for prev, cur in zip(ordered, ordered[1:]):
        assert prev[0]["e"] == cur[0]["b"]
    assert ordered[-1][0]["e"] == len(content)

    # fetch_chunk reads the stored byte span and must return the same text.
    for meta, cands in chunks:
        assert await pointer.fetch_chunk(dict(meta)) == "".join(cands)


@pytest.mark.asyncio
async def test_file_source_document_id_escapes_unsafe_path_characters(tmp_path, mock_context):
    nested_dir = tmp_path / "team's notes"
    nested_dir.mkdir()
    weird_path = nested_dir / "alpha?.txt"
    weird_path.write_text("Encoded path doc", encoding="utf-8")

    source = ConcreteFileSource("col", context=mock_context, collection_config=SimpleCfg(tmp_path))

    docs = [doc async for doc in source.get_documents()]

    assert len(docs) == 1
    document_id = docs[0].document_id
    assert "%27" in document_id
    assert "%3F" in document_id
    assert "team%27s%20notes/alpha%3F.txt" in document_id

    pointer = source.fetch_document(document_id)
    assert pointer.path == weird_path