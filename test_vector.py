from sqlalchemy import JSON, DateTime, Float, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import List, Any, Dict
import json

class Base(DeclarativeBase):
    pass

VECTOR_DIMENSIONS = 768

def serialize_metadata(value):
    if value is None:
        return '{}'
    if isinstance(value, str):
        parsed = json.loads(value)
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

    # NO property decorators

print("DocumentChunk defined successfully")