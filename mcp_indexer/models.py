"""SQLAlchemy models for the document indexer database.

This module provides SQLAlchemy ORM models for storing document chunks,
documents, and summaries with vector embeddings.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import MappingProxyType
from typing import Any, Dict, List, Mapping

from sqlalchemy import ARRAY, JSON, DateTime, Float, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .llm import VECTOR_DIMENSIONS


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class DocumentChunk(Base):
    """A chunk of a document with its embedding.

    Stores the text content, position within the document, and vector embedding
    for semantic search.
    """
    __tablename__ = "document_chunk"

    collection_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    order: Mapped[int]
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str | None]
    embedding: Mapped[List[float]]
    summary_span: Mapped[int | None]
    metadata_str: Mapped[str]

    # Relationships
    document: Mapped[Document] = relationship(back_populates="chunks")
    summary: Mapped[ChunkSummary | None] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_document_chunk_document_id", "document_id"),
        Index("ix_document_chunk_order", "order"),
    )

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return metadata as a read-only mapping."""
        return MappingProxyType(deserialize_metadata(self.metadata_str))

    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None:
        """Set metadata from a mapping."""
        self.metadata_str = serialize_metadata(value)


class Document(Base):
    """Document metadata with its embedding.

    Stores document-level information including title, extracted keywords,
    and an embedding representing the document summary.
    """
    __tablename__ = "document"

    collection_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str]
    title_strength: Mapped[int]
    embedding: Mapped[List[float]]
    keywords: Mapped[List[str]]
    summary: Mapped[str]
    last_modified: Mapped[datetime]

    # Relationships - cascade merge to handle upsert of chunks atomically
    chunks: Mapped[List[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_document_title", "title"),
        Index("ix_document_last_modified", "last_modified"),
    )


class ChunkSummary(Base):
    """Summary for a span of chunks within a document.

    Stores a summary text that covers multiple adjacent chunks in a document.
    The summary_span is an increasing ordinal identifying which summary this is
    within the document.
    """
    __tablename__ = "chunk_summary"

    collection_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    summary_span: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary: Mapped[str]

    # Relationships
    chunks: Mapped[List[DocumentChunk]] = relationship(back_populates="summary")

    __table_args__ = (
        Index("ix_chunk_summary_document_id", "document_id"),
    )


def serialize_metadata(value: Mapping[str, Any] | str | None) -> str:
    """Serialize metadata dict to JSON string."""
    if value is None:
        return "{}"
    if isinstance(value, str):
        parsed = deserialize_metadata(value)
        return json.dumps(parsed, sort_keys=True)
    return json.dumps(dict(value), sort_keys=True)


def deserialize_metadata(value: str | None) -> Dict[str, Any]:
    """Deserialize metadata JSON string to dict."""
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("Metadata must decode to a JSON object.")
    return parsed