from pathlib import Path

import pytest

from mcp_indexer.context import Context
from mcp_indexer.plugins.file_source import FileSource, FileSourceConfig, FileSourcePointer


class ConcreteFileSource(FileSource):
    semantic_boundary_regexps = (rb"\n={3,}\n",)
    embedding_boundary_regexps = (rb"\n\s*\n",)

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
def mock_context():
    return Context(db=None, embedding=None, llm=None, config=None)


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
    text = "Title\n=======\nSubtitle\n--------\nParagraph".encode("utf-8")

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
    text_list = await pointer.fetch_chunk(meta)
    assert "Semantic 1" in "".join(text_list)


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
