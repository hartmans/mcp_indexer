"""Candidate discovery, reranking, and preparation of search results.

Hits are detached values. Database sessions never span source calls or model
inference, and response objects are hydrated after summary writes finish.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .models import Document, DocumentChunk

if TYPE_CHECKING:
    from .indexer import Indexer
    from .plugins.base import DocumentSource

logger = logging.getLogger(__name__)


@dataclass
class DocumentHit:
    document_id: str
    score: float
    summary: str | None = None


@dataclass
class ChunkHit:
    document_id: str
    chunk_metadata: dict[str, Any]
    score: float
    chunk_id: str | None = None


SearchHit = DocumentHit | ChunkHit


async def vector_search_hits(
    collection: DocumentSource, query: str, document_limit: int, chunk_limit: int,
) -> list[SearchHit]:
    """Return document hits followed by chunk hits, each ranked by distance.

    Scores are negative cosine distances (larger is better). They are not
    compared with native scores or across the two lists.
    """
    context = collection.context
    vector = await context.embedding.query(query)
    hits: list[SearchHit] = []
    for model, count in ((Document, document_limit), (DocumentChunk, chunk_limit)):
        try:
            distance = model.embedding.cosine_distance(vector)
            columns = (
                (Document.document_id, Document.summary, distance.label("distance"))
                if model is Document else
                (DocumentChunk.document_id, DocumentChunk.chunk_id,
                 DocumentChunk.metadata_str, distance.label("distance"))
            )
            statement = select(*columns).where(model.collection_id == collection.id)
            if model is Document:
                # Zero vectors cannot be useful document-level candidates.
                statement = statement.where(Document.summary != "")
            statement = statement.order_by(distance, model.document_id)
            if model is DocumentChunk:
                statement = statement.order_by(DocumentChunk.chunk_id)
            async with context.get_session() as session:
                rows = (await session.execute(statement.limit(count))).mappings().all()
            for row in rows:
                if model is Document:
                    hits.append(DocumentHit(row["document_id"], -row["distance"], row["summary"]))
                else:
                    from .models import deserialize_metadata

                    hits.append(ChunkHit(
                        row["document_id"], deserialize_metadata(row["metadata_str"]),
                        -row["distance"], row["chunk_id"],
                    ))
        except Exception:
            logger.exception("Vector search failed for %s (%s)", collection.id, model.__name__)
    return hits


async def _ensure_indexed(indexer: Indexer, source: DocumentSource, document_id: str) -> None:
    async with indexer.context.get_session() as session:
        exists = (await session.execute(select(Document.document_id).where(
            Document.collection_id == source.id, Document.document_id == document_id,
        ))).scalar_one_or_none()
    if exists is None:
        from .indexer import DocumentIndexingStats

        await indexer.index_document(
            source.fetch_document(document_id),
            DocumentIndexingStats(document_id, "index", source=source),
        )


async def _document_summary(indexer: Indexer, source: DocumentSource, document_id: str) -> str:
    async def read():
        async with indexer.context.get_session() as session:
            return (await session.execute(select(Document.summary).where(
                Document.collection_id == source.id, Document.document_id == document_id,
            ))).scalar_one_or_none()

    summary = await read()
    if not summary:
        await _ensure_indexed(indexer, source, document_id)
        # The source's existing summary interface is tried before generating
        # span summaries for sources that need the generic fallback.
        if not await indexer.summarize_document(source, document_id):
            await indexer.summarize_document_chunks(source, document_id)
            await indexer.summarize_document(source, document_id)
        summary = await read()
    if not summary:
        raise ValueError(f"No summary available for {document_id}")
    return summary


async def rerank(
    indexer: Indexer, collection: DocumentSource, hits: list[SearchHit],
    rerank_query: str, hit_limit: int,
) -> list[SearchHit]:
    """Prepare candidates independently of assembly and return selected hits."""
    prepared: list[SearchHit] = []
    texts: list[str] = []
    for hit in hits:
        try:
            if isinstance(hit, DocumentHit):
                text = hit.summary or await _document_summary(indexer, collection, hit.document_id)
                hit = replace(hit, summary=text)
            else:
                text = await collection.fetch_chunk(
                    hit.document_id, hit.chunk_metadata, scope="embedding",
                )
            texts.append(text)
            prepared.append(hit)
        except Exception:
            logger.exception("Could not prepare reranker hit %s", hit.document_id)
    if not prepared:
        return []
    try:
        await collection.reranker.setup()
        ranking = await collection.reranker.score(rerank_query, texts, top_n=hit_limit)
        return [replace(prepared[index], score=score) for index, score in ranking]
    except Exception:
        logger.exception("Reranking failed for %s; using initial ordering", collection.id)
        return prepared[:hit_limit]


async def assemble_hits(
    indexer: Indexer, collection: DocumentSource, hits: list[SearchHit],
) -> list[tuple[Document, list[DocumentChunk]]]:
    """Persist and summarize selected documents; hydrate in fresh sessions."""
    grouped: dict[str, list[ChunkHit]] = {}
    for hit in hits:
        chunks = grouped.setdefault(hit.document_id, [])
        if isinstance(hit, ChunkHit):
            chunks.append(hit)
    results = []
    for document_id, chunk_hits in grouped.items():
        try:
            await _ensure_indexed(indexer, collection, document_id)
            await _document_summary(indexer, collection, document_id)
            if chunk_hits:
                await indexer.summarize_results(collection.id, [document_id])
            async with indexer.context.get_session() as session:
                document = (await session.execute(
                    select(Document).where(
                        Document.collection_id == collection.id,
                        Document.document_id == document_id,
                    ).options(selectinload(Document.chunks).selectinload(DocumentChunk.summary))
                )).scalar_one()
            if not document.summary:
                raise ValueError(f"Missing document summary for {document_id}")
            selected = []
            seen = set()
            for hit in chunk_hits:
                chunk = next((chunk for chunk in document.chunks if (
                    chunk.chunk_id == hit.chunk_id if hit.chunk_id is not None
                    else dict(chunk.metadata_dict) == hit.chunk_metadata
                )), None)
                if chunk is None or chunk.summary is None or not chunk.summary.summary:
                    logger.warning("Missing chunk or summary for %s: %s", document_id, hit)
                    continue
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    selected.append(chunk)
            results.append((document, selected))
        except Exception:
            logger.exception("Could not assemble search result %s", document_id)
    return results


async def search(
    indexer: Indexer, collection_id: str, query: str, limit: int = 5, *,
    document_candidate_limit: int | None = None,
    chunk_candidate_limit: int | None = None,
    rerank_query: str | None = None,
) -> list[tuple[Document, list[DocumentChunk]]]:
    if limit <= 0:
        return []
    collection = indexer.context.collections[collection_id]
    use_reranker = collection.reranker is not None
    document_limit = document_candidate_limit if document_candidate_limit is not None else limit * 4
    chunk_limit = chunk_candidate_limit if chunk_candidate_limit is not None else limit * 4
    try:
        vector_hits = await vector_search_hits(collection, query, document_limit, chunk_limit)
    except Exception:
        logger.exception("Vector discovery failed for %s", collection_id)
        vector_hits = []
    try:
        native_hits = await collection.native_search(
            query, document_limit=document_limit, chunk_limit=chunk_limit,
        )
    except Exception:
        logger.exception("Native discovery failed for %s", collection_id)
        native_hits = []
    if use_reranker:
        hits = await rerank(indexer, collection, vector_hits + native_hits,
                            query if rerank_query is None else rerank_query, limit)
    else:
        hits = []
        for candidates in (vector_hits, native_hits):
            for kind in (DocumentHit, ChunkHit):
                hits.extend([hit for hit in candidates if isinstance(hit, kind)][:limit])
    return await assemble_hits(indexer, collection, hits)
