from sqlalchemy import Integer, JSON, DateTime, Float, Index, String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import List, Any, Dict


class Base(DeclarativeBase):
    pass

VECTOR_DIMENSIONS = 768


class Document(Base):
    __tablename__ = "document"

    collection_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str]
    title_strength: Mapped[int]
    embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_DIMENSIONS))
    summary: Mapped[str]
    last_modified: Mapped[datetime]

    chunks: Mapped[List["DocumentChunk"]] = relationship(back_populates="document")


class ChunkSummary(Base):
    __tablename__ = "chunk_summary"

    collection_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    summary_span: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary: Mapped[str]

    chunks: Mapped[List["DocumentChunk"]] = relationship(back_populates="summary")


class DocumentChunk(Base):
    __tablename__ = "document_chunk"

    collection_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, ForeignKey('document.document_id'), primary_key=True)
    order: Mapped[int]
    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str | None]
    embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_DIMENSIONS))
    summary_span: Mapped[int | None]

    metadata_str: Mapped[str]

    document: Mapped[Document] = relationship(back_populates="chunks")
    summary: Mapped[ChunkSummary | None] = relationship(
        back_populates="chunks",
        primaryjoin="and_(DocumentChunk.collection_id==ChunkSummary.collection_id, DocumentChunk.document_id==ChunkSummary.document_id, DocumentChunk.summary_span==ChunkSummary.summary_span)",
        foreign_keys=["DocumentChunk.document_id", "DocumentChunk.summary_span"]
    )


print("Models defined successfully")

# Try to inspect the mapper
from sqlalchemy.orm import class_mapper
from sqlalchemy import create_engine

try:
    chunk_mapper = class_mapper(DocumentChunk)
    doc_mapper = class_mapper(Document)
    summary_mapper = class_mapper(ChunkSummary)
    print("Mappers configured successfully")
    
    # Check relationships
    print("Document.chunks:", doc_mapper.relationships.get('chunks'))
    print("DocumentChunk.document:", chunk_mapper.relationships.get('document'))
    print("DocumentChunk.summary:", chunk_mapper.relationships.get('summary'))
    
    # Create tables
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    print("Tables created successfully")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")