"""Candidate discovery, reranking, and preparation of search results.

Hits are detached values. Database sessions never span source calls or model
inference, and response objects are hydrated after summary writes finish.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .models import Document, DocumentChunk
from .search_state import (
    HitRecord, InvalidSearchCursor, PageWork, SearchAdvanceError, SearchSession,
)

if TYPE_CHECKING:
    from .indexer import Indexer
    from .plugins.base import DocumentSource

logger = logging.getLogger(__name__)


def set_debug_search(enabled: bool = True) -> None:
    """Enable or disable per-hit search score diagnostics.

    The diagnostics are logged by this module at DEBUG level; they stay
    quiet at the default logging level and appear once the effective level
    of this logger reaches DEBUG (for example, the search client's
    ``--debug-search`` argument).
    """
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)


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


@dataclass(frozen=True)
class NativeSearchBatch:
    hits: list[SearchHit]
    document_offset: int
    chunk_offset: int


@dataclass(frozen=True)
class SearchResult:
    results: list[tuple[Document, list[DocumentChunk]]]
    resume_cursor: str | None
    warnings: tuple[str, ...] = ()


@dataclass
class VectorSearchBatch:
    documents: list[SearchHit] | Exception
    chunks: list[SearchHit] | Exception


async def _vector_type_hits(collection, vector, model, count, offset, threshold):
    if count == 0:
        return []
    distance = model.embedding.cosine_distance(vector)
    columns = (
        (Document.document_id, Document.summary, distance.label("distance"))
        if model is Document else
        (DocumentChunk.document_id, DocumentChunk.chunk_id,
         DocumentChunk.metadata_str, distance.label("distance"))
    )
    statement = select(*columns).where(
        model.collection_id == collection.id, distance <= threshold,
    )
    if model is Document:
        statement = statement.where(Document.summary != "")
    statement = statement.order_by(distance, model.document_id)
    if model is DocumentChunk:
        statement = statement.order_by(DocumentChunk.chunk_id)
    async with collection.context.get_session() as db:
        rows = (await db.execute(statement.limit(count).offset(offset))).mappings().all()
    hits = []
    for row in rows:
        if model is Document:
            hit = DocumentHit(row["document_id"], -row["distance"], row["summary"])
        else:
            from .models import deserialize_metadata
            hit = ChunkHit(row["document_id"], deserialize_metadata(row["metadata_str"]),
                           -row["distance"], row["chunk_id"])
        logger.debug("vector hit for %s: %s cosine_distance=%.4f",
                     collection.id, hit.document_id, row["distance"])
        hits.append(hit)
    return hits


async def vector_search_batch(
    collection, vector, document_limit, chunk_limit, min_cosine_distance,
    *, document_offset=0, chunk_offset=0,
) -> VectorSearchBatch:
    async def discover(model, count, offset):
        try:
            return await _vector_type_hits(
                collection, vector, model, count, offset, min_cosine_distance,
            )
        except Exception as exc:
            logger.exception("Vector search failed for %s (%s)", collection.id, model.__name__)
            return exc
    documents, chunks = await asyncio.gather(
        discover(Document, document_limit, document_offset),
        discover(DocumentChunk, chunk_limit, chunk_offset),
    )
    return VectorSearchBatch(documents, chunks)


async def vector_search_hits(
    collection: DocumentSource, query: str, document_limit: int, chunk_limit: int,
    min_cosine_distance: float, *, document_offset: int = 0, chunk_offset: int = 0,
) -> list[SearchHit]:
    """Convenience discovery API; orchestration uses per-type error outcomes.

    Keep the configured ANN/HNSW query path. Offsets count raw candidates before
    higher-level suppression; an empty stream does not imply exhaustive recall.
    """
    vector = await collection.context.embedding.query(query)
    batch = await vector_search_batch(
        collection, vector, document_limit, chunk_limit, min_cosine_distance,
        document_offset=document_offset, chunk_offset=chunk_offset,
    )
    return [hit for outcome in (batch.documents, batch.chunks)
            if not isinstance(outcome, Exception) for hit in outcome]


async def _ensure_indexed(indexer: Indexer, source: DocumentSource, document_id: str) -> None:
    await indexer.ensure_document_indexed(source, document_id)


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


async def _prepare_hit(indexer, collection, hit):
    if isinstance(hit, DocumentHit):
        text = hit.summary or await _document_summary(indexer, collection, hit.document_id)
        return replace(hit, summary=text), text
    return hit, await collection.fetch_chunk(
        hit.document_id, hit.chunk_metadata, scope="embedding",
    )


async def _score_texts(reranker, query, texts):
    try:
        ranking = await reranker.score(query, texts, top_n=None)
        scores = {}
        for index, score in ranking:
            if type(index) is not int or not 0 <= index < len(texts) or index in scores:
                raise ValueError("Invalid reranker indices")
            if not math.isfinite(score):
                raise ValueError("Nonfinite reranker score")
            scores[index] = float(score)
        if len(scores) != len(texts):
            raise ValueError("Reranker omitted candidate scores")
        return scores
    except Exception as exc:
        raise SearchAdvanceError("Reranking failed; retry the same cursor") from exc


async def rerank(
    indexer: Indexer, collection: DocumentSource, hits: list[SearchHit],
    rerank_query: str, hit_limit: int, min_rerank_score: float,
) -> list[SearchHit]:
    """Standalone scoring helper. Resumable search retains scores separately."""
    unique = {}
    for hit in hits:
        unique.setdefault(_hit_key(hit), hit)
    outcomes = await asyncio.gather(*(
        _prepare_hit(indexer, collection, hit) for hit in unique.values()
    ), return_exceptions=True)
    prepared = []
    for outcome in outcomes:
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            logger.error("Could not prepare reranker hit: %s", outcome)
        else:
            prepared.append(outcome)
    if not prepared:
        return []
    scores = await _score_texts(collection.reranker, rerank_query, [text for _, text in prepared])
    ranked = []
    for index, (hit, _) in enumerate(prepared):
        score = scores[index]
        logger.debug("reranked %s: rerank_score=%.4f", hit.document_id, score)
        if score >= min_rerank_score:
            ranked.append(replace(hit, score=score))
    return sorted(ranked, key=lambda hit: -hit.score)[:hit_limit]


async def assemble_hits(
    indexer: Indexer, collection: DocumentSource, hits: list[SearchHit],
) -> list[tuple[Document, list[DocumentChunk]]]:
    """Persist and summarize selected documents; hydrate in fresh sessions."""
    grouped: dict[str, list[ChunkHit]] = {}
    for hit in hits:
        chunks = grouped.setdefault(hit.document_id, [])
        if isinstance(hit, ChunkHit):
            chunks.append(hit)
    async def assemble_document(
        document_id: str, chunk_hits: list[ChunkHit],
    ) -> tuple[Document, list[DocumentChunk]]:
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
        return document, selected

    outcomes = await asyncio.gather(*(
        assemble_document(document_id, chunk_hits)
        for document_id, chunk_hits in grouped.items()
    ), return_exceptions=True)
    results = []
    for document_id, outcome in zip(grouped, outcomes, strict=True):
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            logger.error("Could not assemble search result %s: %s", document_id, outcome)
        else:
            results.append(outcome)
    return results


def _hit_key(hit: SearchHit) -> tuple:
    if isinstance(hit, DocumentHit):
        return ("document", hit.document_id)
    if not isinstance(hit, ChunkHit) or not isinstance(hit.chunk_metadata, dict):
        raise ValueError("Invalid search hit")
    metadata = json.dumps(hit.chunk_metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return ("chunk", hit.document_id, metadata)


def _accept_hits(session, hits, warnings):
    for hit in hits:
        if hit.document_id in session.returned or hit.document_id in session.failed_documents:
            continue
        try:
            key = _hit_key(hit)
        except (ValueError, TypeError):
            warnings.append("Discarded a hit with invalid retrieval metadata")
            continue
        if key not in session.records:
            session.records[key] = HitRecord(copy.deepcopy(hit), session.encounter)
            session.encounter += 1


def _validate_native(batch, session):
    if batch is None:
        return
    if not isinstance(batch, NativeSearchBatch):
        raise ValueError("Native discovery must return NativeSearchBatch or None")
    old = (session.native_document_offset, session.native_chunk_offset)
    new = (batch.document_offset, batch.chunk_offset)
    limits = (session.document_limit, session.chunk_limit)
    for before, after, count in zip(old, new, limits):
        if type(after) is not int or after < before or (count == 0 and after != before):
            raise ValueError("Invalid native continuation offset")
    if new == old:
        raise ValueError("Native discovery did not advance")
    if (sum(isinstance(hit, DocumentHit) for hit in batch.hits) > session.document_limit
            or sum(isinstance(hit, ChunkHit) for hit in batch.hits) > session.chunk_limit
            or any(not isinstance(hit, (DocumentHit, ChunkHit)) for hit in batch.hits)):
        raise ValueError("Invalid native hit batch")


async def _discover(collection, session: SearchSession, work: PageWork):
    async def vector():
        dl = 0 if session.vector_documents_done else session.document_limit
        cl = 0 if session.vector_chunks_done else session.chunk_limit
        if not dl and not cl:
            return VectorSearchBatch([], [])
        try:
            if session.embedding is None:
                session.embedding = await collection.context.embedding.query(session.natural_query)
            return await vector_search_batch(
                collection, session.embedding, dl, cl, session.min_cosine_distance,
                document_offset=session.vector_document_offset,
                chunk_offset=session.vector_chunk_offset,
            )
        except Exception as exc:
            return VectorSearchBatch(exc if dl else [], exc if cl else [])

    async def native():
        if session.native_done:
            return None
        try:
            batch = await collection.native_search(
                session.query, document_limit=session.document_limit,
                chunk_limit=session.chunk_limit,
                document_offset=session.native_document_offset,
                chunk_offset=session.native_chunk_offset,
            )
            _validate_native(batch, session)
            return batch
        except Exception as exc:
            logger.exception("Native search failed for %s", collection.id)
            return exc

    attempted = (not session.vector_documents_done, not session.vector_chunks_done, not session.native_done)
    vb, nb = await asyncio.gather(vector(), native())
    successes = 0
    for active, kind, outcome in zip(attempted[:2], ("document", "chunk"), (vb.documents, vb.chunks)):
        if not active:
            continue
        if isinstance(outcome, Exception):
            work.warn(f"Vector {kind} discovery failed; continuation can retry")
            continue
        successes += 1
        name = f"vector_{kind}_offset"
        setattr(session, name, getattr(session, name) + len(outcome))
        if not outcome:
            setattr(session, f"vector_{kind}s_done", True)
        _accept_hits(session, outcome, work.warnings)
    if attempted[2]:
        if isinstance(nb, Exception):
            work.warn("Native discovery failed; continuation can retry")
        else:
            successes += 1
            if nb is None:
                session.native_done = True
            else:
                session.native_document_offset = nb.document_offset
                session.native_chunk_offset = nb.chunk_offset
                for hit in nb.hits:
                    logger.debug("native hit for %s: %s native_score=%.4f",
                                 collection.id, hit.document_id, hit.score)
                # Encounter order is documents then chunks, irrespective of source layout.
                _accept_hits(session, [hit for kind in (DocumentHit, ChunkHit)
                                      for hit in nb.hits if isinstance(hit, kind)], work.warnings)
    if any(attempted) and successes == 0 and not any(
        record.hit is not None for record in session.records.values()
    ):
        raise SearchAdvanceError("Discovery failed; retry the same cursor")
    work.discovery_done = True


async def _score_candidates(indexer, collection, session, work):
    unprepared = [record for record in session.records.values()
                  if not record.finished and record.hit is not None and record.text is None]
    if session.reranker is None:
        for record in unprepared:
            record.score = record.hit.score
            record.finished = True
        return
    outcomes = await asyncio.gather(*(
        _prepare_hit(indexer, collection, record.hit) for record in unprepared
    ), return_exceptions=True)
    for record, outcome in zip(unprepared, outcomes):
        record.preparation_attempts += 1
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            work.warn(f"Could not prepare hit for {record.hit.document_id}")
            logger.warning("Hit preparation failed: %s", outcome)
            if record.preparation_attempts >= 2:
                record.hit = None
                record.finished = True
        else:
            record.hit, record.text = outcome
    prepared = [record for record in session.records.values()
                if not record.finished and record.text is not None]
    if not prepared:
        return
    scores = await _score_texts(session.reranker, session.natural_query,
                                [record.text for record in prepared])
    for index, record in enumerate(prepared):
        record.score = scores[index]
        record.finished = True
        record.text = None
        logger.debug("reranked %s: rerank_score=%.4f", record.hit.document_id, record.score)
        if record.score < session.min_rerank_score:
            record.hit = None
        else:
            record.hit = replace(record.hit, score=record.score)


def _pending_documents(session):
    grouped = {}
    for record in session.records.values():
        if record.finished and record.hit is not None:
            grouped.setdefault(record.hit.document_id, []).append(record)
    def order(item):
        document_id, records = item
        encounter = min(record.encounter for record in records)
        if session.reranker is None:
            return (encounter, document_id)
        return (-max(record.score for record in records), encounter, document_id)
    return sorted(grouped.items(), key=order)


def _copy_page(page: SearchResult) -> SearchResult:
    """Copy loaded response data without copying SQLAlchemy instrumentation.

    deepcopy of instrumented relationship lists is unsafe. Rebuild transient
    ORM values from loaded columns and the eagerly loaded response relationships;
    no database reads occur, including on replay.
    """
    def clone_columns(record):
        return type(record)(**{
            column.key: copy.deepcopy(record.__dict__[column.key])
            for column in record.__table__.columns if column.key in record.__dict__
        })

    results = []
    for document, selected in page.results:
        if not isinstance(document, Document):
            results.append(copy.deepcopy((document, selected)))
            continue
        cloned_document = clone_columns(document)
        chunks = {}
        summaries = {}
        for chunk in document.chunks:
            cloned_chunk = clone_columns(chunk)
            summary = chunk.summary
            if summary is not None:
                if id(summary) not in summaries:
                    summaries[id(summary)] = clone_columns(summary)
                cloned_chunk.summary = summaries[id(summary)]
            else:
                cloned_chunk.summary = None
            chunks[chunk.chunk_id] = cloned_chunk
        cloned_document.chunks = list(chunks.values())
        cloned_document.summaries = list(summaries.values())
        results.append((cloned_document, [chunks[chunk.chunk_id] for chunk in selected]))
    return SearchResult(results, page.resume_cursor, page.warnings)


async def _advance(indexer, collection, session, store, *, initial):
    work = session.work
    if not work.discovery_done:
        await _discover(collection, session, work)
    await _score_candidates(indexer, collection, session, work)
    selected = _pending_documents(session)[:work.limit]
    hits = [record.hit for _, records in selected for record in records]
    todo = [hit for hit in hits if hit.document_id not in work.assembly_attempted]
    if todo:
        # Assembly's per-document failures are omitted; successful results are
        # retained in PageWork before snapshot/commit so a retry need not reassemble.
        results = await assemble_hits(indexer, collection, todo)
        for document, chunks in results:
            work.assembled[document.document_id] = (document, chunks)
        for hit in todo:
            work.assembly_attempted.add(hit.document_id)
        for document_id in {hit.document_id for hit in todo} - work.assembled.keys():
            count = session.assembly_attempts.get(document_id, 0) + 1
            session.assembly_attempts[document_id] = count
            work.warn(f"Could not assemble document {document_id}")
            if count >= 2:
                session.failed_documents.add(document_id)
    page_results = [work.assembled[document_id] for document_id, _ in selected
                    if document_id in work.assembled]
    successful = {document.document_id for document, _ in page_results}
    # Snapshot before updating returned IDs or token window.
    snapshot = _copy_page(SearchResult(page_results, None, tuple(work.warnings)))
    for record in session.records.values():
        if record.hit is not None and record.hit.document_id in successful | session.failed_documents:
            record.hit = None
            record.text = None
            record.finished = True
    session.returned.update(successful)
    complete = (session.vector_documents_done and session.vector_chunks_done and session.native_done
                and not any(record.hit is not None for record in session.records.values()))
    store.commit(session, snapshot, complete=complete, initial=initial)
    page = replace(snapshot, resume_cursor=session.current_cursor)
    session.cached_page = page
    if initial and complete:
        # A terminal initial response has no public replay token.
        store.remove(session)
    return page


async def search(
    indexer: Indexer, collection_id: str, query: str | None = None, limit: int = 5, *,
    document_candidate_limit: int | None = None,
    chunk_candidate_limit: int | None = None,
    rerank_query: str | None = None,
    cursor: str | None = None,
) -> SearchResult:
    """Advance discovery once or replay the latest page within a 60-minute session."""
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    collection = indexer.context.collections[collection_id]
    store = collection._search_sessions
    initial = cursor is None
    override = rerank_query if rerank_query and rerank_query.strip() else None
    if initial:
        if query is None:
            raise ValueError("query is required without a cursor")
        dl = document_candidate_limit if document_candidate_limit is not None else limit * 4
        cl = chunk_candidate_limit if chunk_candidate_limit is not None else limit * 4
        if type(dl) is not int or type(cl) is not int or dl < 0 or cl < 0 or not (dl or cl):
            raise ValueError("Candidate limits must be nonnegative with at least one enabled type")
        reranker = collection.reranker
        if reranker is not None:
            try:
                await reranker.setup()
            except Exception as exc:
                raise SearchAdvanceError("Could not initialize reranker") from exc
        session = SearchSession(
            query, override, override or query, dl, cl,
            collection.config.min_cosine_distance, collection.config.min_rerank_score, reranker,
            vector_documents_done=dl == 0, vector_chunks_done=cl == 0,
        )
        store.add(session)
        token = session.current_cursor
    else:
        token = cursor
        session = store.lookup(token)
    warnings = []
    if query is not None and query != session.query:
        warnings.append("query differs from this search; using the saved query")
    if rerank_query is not None and override != session.rerank_query:
        warnings.append("rerank_query differs from this search; using the saved rerank query")
    async with session.lock:
        # Recheck after waiting; another advancement may have expired this token.
        store.lookup(token)
        for supplied, saved in ((document_candidate_limit, session.document_limit),
                                (chunk_candidate_limit, session.chunk_limit)):
            if supplied is not None and supplied != saved:
                raise ValueError("Candidate limits cannot change while resuming")
        store.touch(session)
        if token == session.previous_cursor:
            return replace(_copy_page(session.cached_page),
                           warnings=session.cached_page.warnings + tuple(warnings))
        if token != session.current_cursor:
            raise InvalidSearchCursor()
        if session.task is None or session.task.done():
            if session.work is None:
                session.work = PageWork(limit)
            session.task = asyncio.create_task(
                _advance(indexer, collection, session, store, initial=initial),
            )
            # Retrieve unobserved exceptions when every waiter disconnects.
            session.task.add_done_callback(lambda task: store.task_finished(session, task))
        task = session.task
    try:
        page = await asyncio.shield(task)
    except BaseException:
        if initial and task.done():
            store.remove(session)
        raise
    return replace(_copy_page(page), warnings=page.warnings + tuple(warnings))
