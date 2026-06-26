import pytest
import json
from mcp_indexer.models import ChunkSummary, Document, DocumentChunk
from types import MappingProxyType


def deserialize_metadata(value):
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed


def test_document_chunk_metadata_is_json_backed():
    """Test that metadata is stored as JSON in metadata_str."""
    metadata = {"path": "notes.txt", "o": 12, "s": 4}
    metadata_str = json.dumps(metadata, sort_keys=True)
    
    chunk = DocumentChunk(
        document_id="doc-1",
        order=0,
        chunk_id="doc-1?c=0",
        text="chunk text",
        summary_span=0,
        embedding=[0.0] * 768,
        metadata_str=metadata_str,
    )

    assert chunk.metadata_str == metadata_str
    assert deserialize_metadata(chunk.metadata_str) == metadata

    # Update metadata by setting metadata_str directly
    new_metadata = {"path": "notes.txt", "o": 20, "s": 7}
    chunk.metadata_str = json.dumps(new_metadata, sort_keys=True)
    
    assert json.loads(chunk.metadata_str) == new_metadata

    # Test metadata_dict property
    assert isinstance(chunk.metadata_dict, MappingProxyType)
    assert dict(chunk.metadata_dict) == new_metadata


def test_models_have_vector_embeddings():
    """Test that DocumentChunk and Document use Vector embeddings."""
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.orm import class_mapper
    
    chunk_mapper = class_mapper(DocumentChunk)
    chunk_embedding_col = chunk_mapper.columns['embedding']
    assert 'VECTOR' in str(chunk_embedding_col.type).upper()
    
    doc_mapper = class_mapper(Document)
    doc_embedding_col = doc_mapper.columns['embedding']
    assert 'VECTOR' in str(doc_embedding_col.type).upper()