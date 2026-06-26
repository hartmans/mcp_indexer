#!/usr/bin/env python3
"""Test script to verify model imports work with current dependencies."""

from sqlalchemy import JSON, DateTime, Float, Index, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import List, Any, Dict, Mapping
from types import MappingProxyType
import json

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass

VECTOR_DIMENSIONS = 768

def serialize_metadata(value):
    if value is None:
        return '{}'
    if isinstance(value, str):
        parsed = deserialize_metadata(value)
        return json.dumps(parsed, sort_keys=True)
    return json.dumps(dict(value), sort_keys=True)

def deserialize_metadata(value):
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError('Metadata must decode to a JSON object.')
    return parsed


class DocumentChunk(Base):
    __tablename__ = "document_chunk"

    collection_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    order: Mapped[int]
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str | None]
    embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_DIMENSIONS))
    summary_span: Mapped[int | None]
    metadata_str: Mapped[str]

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(deserialize_metadata(self.metadata_str))


print("DocumentChunk defined successfully")


class Document(Base):
    __tablename__ = "document"

    collection_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str]
    title_strength: Mapped[int]
    embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_DIMENSIONS))
    keywords: Mapped[List[str]] = mapped_column(ARRAY(String))
    summary: Mapped[str]
    last_modified: Mapped[datetime]


print("Document defined successfully")


class ChunkSummary(Base):
    __tablename__ = "chunk_summary"

    collection_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    summary_span: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary: Mapped[str]


print("ChunkSummary defined successfully")
print("All models imported successfully")