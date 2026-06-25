#!/usr/bin/env python3
"""Create test LanceDB data for dump_lancedb.py testing."""

import os
import shutil
from datetime import datetime
import lancedb
from mcp_indexer.plugins.base import DocumentChunk, Document, ChunkSummary

# Clean up any existing test data
db_path = '/tmp/test_lancedb_dump'
if os.path.exists(db_path):
    shutil.rmtree(db_path)

db = lancedb.connect(db_path)

# Create test collection 1
chunk_data1 = [
    DocumentChunk(
        document_id='doc1',
        order=0,
        chunk_id='doc1?c=0',
        text='This is chunk one',
        embedding=[0.1] * 768,
        metadata={'path': 'test1.txt', 'o': 0, 's': 15}
    ),
    DocumentChunk(
        document_id='doc1',
        order=1,
        chunk_id='doc1?c=1',
        text='This is chunk two',
        embedding=[0.2] * 768,
        metadata={'path': 'test1.txt', 'o': 15, 's': 15}
    ),
    DocumentChunk(
        document_id='doc2',
        order=0,
        chunk_id='doc2?c=0',
        text='Second document chunk',
        embedding=[0.3] * 768,
        metadata={'path': 'test2.txt', 'o': 0, 's': 20}
    ),
]

meta_data1 = [
    Document(
        document_id='doc1',
        title='Test Document 1',
        embedding=[0.15] * 768,
        last_modified=datetime(2026, 1, 15, 10, 30, 0)
    ),
    Document(
        document_id='doc2',
        title='Test Document 2',
        embedding=[0.35] * 768,
        last_modified=datetime(2026, 1, 16, 14, 45, 0)
    )
]

summary_data1 = [
    ChunkSummary(
        document_id='doc1',
        summary_span=0,
        summary='First test document summary'
    ),
    ChunkSummary(
        document_id='doc2',
        summary_span=0,
        summary='Second test document summary'
    )
]

db.create_table('test_col', data=[c.model_dump() for c in chunk_data1])
db.create_table('test_col_meta', data=[m.model_dump() for m in meta_data1])
db.create_table('test_col_summary', data=[s.model_dump() for s in summary_data1])

# Create test collection 2
chunk_data2 = [
    DocumentChunk(
        document_id='spacedoc',
        order=0,
        chunk_id='spacedoc?c=0',
        text='Space is big',
        embedding=[0.5] * 768,
        metadata={'path': 'space.txt', 'o': 0, 's': 10}
    ),
]

meta_data2 = [
    Document(
        document_id='spacedoc',
        title='Space Document',
        embedding=[0.55] * 768,
        last_modified=datetime(2026, 2, 1, 9, 0, 0)
    )
]

summary_data2 = [
    ChunkSummary(
        document_id='spacedoc',
        summary_span=0,
        summary='About space being big'
    )
]

db.create_table('space_col', data=[c.model_dump() for c in chunk_data2])
db.create_table('space_col_meta', data=[m.model_dump() for m in meta_data2])
db.create_table('space_col_summary', data=[s.model_dump() for s in summary_data2])

print('Created test data successfully')
print(f'Tables: {sorted(db.table_names())}')
print(f'DB Path: {db_path}')