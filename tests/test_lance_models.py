import pytest

from mcp_indexer.plugins.base import ChunkSummary, Document, DocumentChunk


def test_document_chunk_metadata_is_json_backed_and_read_only():
    chunk = DocumentChunk(
        document_id="doc-1",
        order=0,
        chunk_id="doc-1?c=0",
        text="chunk text",
        summary_span=0,
        embedding=[0.0] * 768,
        metadata={"path": "notes.txt", "o": 12, "s": 4},
    )

    assert chunk.metadata_str == '{"o": 12, "path": "notes.txt", "s": 4}'
    assert dict(chunk.metadata) == {"path": "notes.txt", "o": 12, "s": 4}

    with pytest.raises(TypeError):
        chunk.metadata["extra"] = "nope"

    chunk.metadata = {"path": "notes.txt", "o": 20, "s": 7}

    assert chunk.metadata_str == '{"o": 20, "path": "notes.txt", "s": 7}'
    assert dict(chunk.metadata) == {"path": "notes.txt", "o": 20, "s": 7}

    chunk_from_storage = DocumentChunk(
        document_id="doc-1",
        order=1,
        chunk_id="doc-1?c=1",
        text="chunk text",
        summary_span=0,
        embedding=[0.0] * 768,
        metadata_str='{"path": "stored.txt", "o": 3, "s": 2}',
    )
    assert dict(chunk_from_storage.metadata) == {"path": "stored.txt", "o": 3, "s": 2}


def test_lance_models_convert_to_arrow_schema():
    for model in (DocumentChunk, Document, ChunkSummary):
        schema = model.to_arrow_schema()
        assert schema is not None
        assert schema.names
