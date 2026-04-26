import pytest
import asyncio
from pathlib import Path
import shutil
import tempfile
from typing import Any
from lance_indexer.plugins.text_source import TextFileSource, TextFilePointer, TextFileSourceConfig
from lance_indexer.context import Context

class SimpleCfg:
    def __init__(self, directory):
        self.source_blob = {"directory": directory}
    def resolve_source_config(self, model):
        return model(**self.source_blob)

@pytest.fixture
def temp_text_env():
    temp_dir = Path(tempfile.mkdtemp())
    # Create a Markdown file
    (temp_dir / "test.md").write_text(
        "# Main Title\n\n## Subtitle\n\nThis is a paragraph.\n\nAnother paragraph.", 
        encoding="utf-8"
    )
    # Create an RST file
    (temp_dir / "test.rst").write_text(
        "Main Title\n===========\n\nSubtitle\n----------\n\nThis is a paragraph.", 
        encoding="utf-8"
    )
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_context():
    return Context(db=None, embedding_fn=None, llm=None, config=None)

@pytest.mark.asyncio
async def test_text_source_config_resolution(temp_text_env, mock_context):
    cfg = SimpleCfg(temp_text_env)
    source = TextFileSource("col1", context=mock_context, collection_config=cfg)
    
    assert isinstance(source.source_config, TextFileSourceConfig)
    assert source.source_config.directory == temp_text_env
    assert source.source_config.encoding == "utf-8"

@pytest.mark.asyncio
async def test_markdown_semantic_boundaries(mock_context):
    # Testing the Markdown boundary: rb"\n\s*#{1,6}\s+.*"
    text = b"Intro text\n\n# Header 1\nContent 1\n\n# Header 2\nContent 2"
    
    # Using a mock config as we only need the split_text method
    class MockCfg:
        def __init__(self): self.source_blob = {"directory": Path(".")}
        def resolve_source_config(self, model): return model(**self.source_blob)
        
    source = TextFileSource("col", context=mock_context, collection_config=MockCfg())
    source.source_config = TextFileSourceConfig(directory=Path("."))
    
    # min_size=1 to ensure we don't merge boundaries
    results = []
    async for chunk in source.split_text({}, text, min_size=1, max_size=1000):
        results.append(chunk)
    
    # We expect 3 semantic chunks:
    # 1. Intro text\n\n
    # 2. # Header 1\nContent 1\n\n
    # 3. # Header 2\nContent 2
    assert len(results) == 3
    assert b"# Header 1" in "".join(results[1][1]).encode("utf-8")

@pytest.mark.asyncio
async def test_rst_semantic_boundaries(mock_context):
    # Testing the RST boundary: rb"\n={3,}\n" and rb"\n-{3,}\n"
    text = b"Title\n=======\n\nSection 1\n----------\n\nContent"
    
    class MockCfg:
        def __init__(self): self.source_blob = {"directory": Path(".")}
        def resolve_source_config(self, model): return model(**self.source_blob)
        
    source = TextFileSource("col", context=mock_context, collection_config=MockCfg())
    source.source_config = TextFileSourceConfig(directory=Path("."))
    
    results = []
    async for chunk in source.split_text({}, text, min_size=1, max_size=1000):
        results.append(chunk)
    
    # We expect 3 semantic chunks:
    # 1. Title
    # 2. =======\n\nSection 1
    # 3. ----------\n\nContent
    assert len(results) == 3

@pytest.mark.asyncio
async def test_text_file_pointer_retrieval(temp_text_env, mock_context):
    path = temp_text_env / "test.md"
    path.write_text("# Title\n\nContent", encoding="utf-8")
    
    source = TextFileSource("col", context=mock_context, collection_config=SimpleCfg(temp_text_env))
    source.source_config = TextFileSourceConfig(directory=temp_text_env)
    
    pointer = source.fetch_document(f"{source.id_prefix}test.md")
    assert isinstance(pointer, TextFilePointer)
    
    # Test metadata
    meta = await pointer.get_metadata()
    assert meta["title"] == "test.md"
    
    # Test get_chunks retrieval
    chunks = []
    async for chunk in pointer.get_chunks():
        chunks.append(chunk)
    
    assert len(chunks) > 0
    # Verify that the retrieved content matches the file
    full_text = "".join(["".join(c[1]) for c in chunks])
    assert "# Title" in full_text
