"""Shared chunk text retrieval.

The former ``DocumentSource.fetch_chunk`` wrapper is replaced by
``fetch_chunk``: an application-level helper that operates on a stored chunk
ID. It returns the text retained in the database whenever that text is
available, and otherwise falls back to a fresh source read. The helper is
usable by search, the indexer, and MCP retrieval; it does not belong to any
format plugin.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sqlalchemy import select

from .models import DocumentChunk, deserialize_metadata

if TYPE_CHECKING:
    from .plugins.base import DocumentSource

Scope = Literal["semantic", "embedding"]


async def fetch_chunk(
    source: DocumentSource, chunk_id: str, *, scope: Scope = "semantic",
) -> str:
    """Return the text of the chunk stored under ``chunk_id``.

    ``scope="semantic"`` returns the whole semantic chunk that the stored row
    belongs to, reconstructed by concatenating the retained text of every
    embedding row in the group, in order. ``scope="embedding"`` returns the
    requested row's own retained text.

    Each scope obtains its text and fallback information in one SQL statement;
    semantic scope does not first load an ORM row to discover its group. Retained
    text is returned without acquiring a document handle. When the scope has no
    retained text, the document is acquired through the source's handle cache and
    the semantic text is fetched from the handle; embedding scope then applies
    the row's stored character slice.

    A chunk ID that has no stored row raises LookupError. A row whose text is
    NULL triggers the fallback; an empty retained text is returned as-is. Incomplete
    retained text is never assembled or mixed with a different source version.
    """
    document_id = source.document_id(chunk_id)
    if scope == "embedding":
        statement = select(
            DocumentChunk.text,
            DocumentChunk.embedding_offset,
            DocumentChunk.embedding_length,
            DocumentChunk.metadata_str,
        ).where(
            DocumentChunk.collection_id == source.id,
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_id == chunk_id,
        )
    else:
        root_id = select(DocumentChunk.semantic_chunk_id).where(
            DocumentChunk.collection_id == source.id,
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_id == chunk_id,
        ).scalar_subquery()
        statement = select(
            DocumentChunk.text,
            DocumentChunk.metadata_str,
        ).where(
            DocumentChunk.collection_id == source.id,
            DocumentChunk.document_id == document_id,
            DocumentChunk.semantic_chunk_id == root_id,
        ).order_by(DocumentChunk.order)

    async with source.context.get_session() as session:
        result = await session.execute(statement)

    if scope == "embedding":
        row = result.first()
        if row is None:
            raise LookupError(f"Chunk not found: {chunk_id!r}")
        text, offset, length, metadata_str = row
        if text is not None:
            return text
        semantic = await _semantic_from_handle(source, document_id, metadata_str)
        return semantic[offset:offset + length]

    rows = result.all()
    if not rows:
        raise LookupError(f"Chunk not found: {chunk_id!r}")
    texts = [text for text, _ in rows]
    if all(text is not None for text in texts):
        return "".join(texts)
    return await _semantic_from_handle(source, document_id, rows[0][1])


async def _semantic_from_handle(
    source: DocumentSource, document_id: str, metadata_str: str | None,
) -> str:
    """Fetch a semantic chunk's full text from a freshly acquired handle.

    The handle is acquired through the source's process-local handle cache.
    Chunk-level offsets stored in legacy metadata are stripped before the
    handle fetch, which addresses the semantic chunk by its source metadata.
    """
    handle = await source.fetch_document(document_id)
    metadata = {
        key: value
        for key, value in deserialize_metadata(metadata_str).items()
        if key not in ("o", "s")
    }
    return await handle.fetch_chunk(metadata)
