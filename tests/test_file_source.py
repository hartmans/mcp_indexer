import pytest
import asyncio
from pathlib import Path
import shutil
import tempfile
from typing import AsyncGenerator, Any
from mcp_indexer.plugins.file_source import FileSource, FileSourcePointer, FileSourceConfig
from mcp_indexer.context import Context
from mcp_indexer.config import CollectionConfig

class ConcreteFileSource(FileSource):
    """Concrete implementation of FileSource for testing."""
    # Custom boundaries for testing
    semantic_boundary_regexps = (rb"\n={3,}\n",) # rst style headers
    embedding_boundary_regexps = (rb"\n\s*\n",)

    def fetch_document(self, document_id: str) -> FileSourcePointer:
        relative_path_str = self.strip_id_prefix(document_id)
        absolute_path = self.source_config.directory / relative_path_str
        return FileSourcePointer(self, document_id, absolute_path)

@pytest.fixture
def temp_file_env():
    temp_dir = Path(tempfile.mkdtemp())
    # Create some sample files
    (temp_dir / "doc1.txt").write_text("Title 1\n=======\n\nParagraph 1\n\nParagraph 2", encoding="utf-8")
    (temp_dir / "doc2.txt").write_text("Title 2\n-------\n\nSubtitle\n-----\n\nParagraph 1", encoding="utf-8")
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_context():
    return Context(db=None, embedding_fn=None, llm=None, config=None)

@pytest.fixture
def mock_col_config():
    # Manually create a config object that mimics the resolve_source_config behavior
    # or just a simple object with a source_blob
    class MockCfg:
        def __init__(self, directory):
            self.source_blob = {"directory": directory}
        def resolve_source_config(self, model):
            return model(**self.source_blob)
    return MockCfg

@pytest.mark.asyncio
async def test_file_source_discovery(temp_file_env, mock_context):
    class SimpleCfg:
        def __init__(self, directory):
            self.source_blob = {"directory": directory}
        def resolve_source_config(self, model):
            return model(**self.source_blob)
    
    cfg = SimpleCfg(temp_file_env)
    source = ConcreteFileSource("test_col", context=mock_context, collection_config=cfg)
    source.source_config = FileSourceConfig(directory=temp_file_env)
    
    docs = []
    async for doc in source.get_documents():
        docs.append(doc)
    
    assert len(docs) == 2
    assert any("doc1.txt" in doc.document_id for doc in docs)
    assert any("doc2.txt" in doc.document_id for doc in docs)

@pytest.mark.asyncio
async def test_split_text_rst_headers(mock_context):
    # The "rst header" case:
    # Title
    # =======
    # Subtitle
    # --------
    # Paragraph
    
    text = "Title\n=======\nSubtitle\n--------\nParagraph".encode("utf-8")
    
    class SimpleCfg:
        def __init__(self):
            self.source_blob = {}
        def resolve_source_config(self, model):
            return model(**self.source_blob)

    # Case 1: min_size is small, they split
    source = ConcreteFileSource("col", context=mock_context, collection_config=SimpleCfg())
    source.source_config = FileSourceConfig(directory=Path("."))
    
    results = []
    async for chunk in source.split_text({}, text, min_size=1, max_size=1000):
        results.append(chunk)
    
    # Should split at both ======= and --------
    assert len(results) >= 2

    # Case 2: min_size is large, Title + Subtitle should stay together
    # "Title\n=======\nSubtitle\n--------\n" is approx 30 chars.
    # If min_size = 50, the first boundary should be ignored.
    results_large = []
    async for chunk in source.split_text({}, text, min_size=50, max_size=1000):
        results_large.append(chunk)
    
    # Should merge the first few boundaries
    assert len(results_large) < len(results)

@pytest.mark.asyncio
async def test_file_source_pointer_retrieval(temp_file_env, mock_context):
    # Setup
    path = temp_file_env / "doc1.txt"
    path.write_text("Semantic 1\n\nSemantic 2", encoding="utf-8")
    
    class SimpleCfg:
        def __init__(self, directory):
            self.source_blob = {"directory": directory}
        def resolve_source_config(self, model):
            return model(**self.source_blob)
            
    source = ConcreteFileSource("col", context=mock_context, collection_config=SimpleCfg(temp_file_env))
    source.source_config = FileSourceConfig(directory=temp_file_env)
    
    pointer = source.fetch_document(f"{source.id_prefix}doc1.txt")
    
    # Test get_chunks -> split_text -> offsets
    # Note: current FileSourcePointer.get_chunks uses min_size=100, 
    # so "Semantic 1\n\nSemantic 2" (22 chars) will be one chunk.
    chunks = []
    async for chunk_info in pointer.get_chunks():
        chunks.append(chunk_info)
    
    assert len(chunks) == 1
    
    # Test fetch_chunk using the metadata from get_chunks
    meta, _ = chunks[0]
    text_list = await pointer.fetch_chunk(meta)
    assert "Semantic 1" in "".join(text_list)
