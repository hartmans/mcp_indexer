#!/usr/bin/env python3
"""Test script to verify model imports work with current dependencies."""

from mcp_indexer.models import Document, DocumentChunk, ChunkSummary, FailedDocument

print("All models imported successfully")
print(f"Document columns: {list(Document.__table__.columns.keys())}")
print(f"DocumentChunk columns: {list(DocumentChunk.__table__.columns.keys())}")
print(f"ChunkSummary columns: {list(ChunkSummary.__table__.columns.keys())}")
print(f"FailedDocument columns: {list(FailedDocument.__table__.columns.keys())}")