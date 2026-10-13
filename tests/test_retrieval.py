"""Tests for the shared chunk text retrieval helper (mcp_indexer.retrieval).

Covers retained-text retrieval without handle acquisition, semantic scope
from a non-root chunk ID, fallback to a fresh source read for unretained
text (with the stored character slice applied for embedding scope),
missing-row versus NULL-text behavior, empty retained text, one SQL
statement per scope, and mapper-level deferral of the text column.
"""
from datetime import datetime

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import MissingGreenlet

from mcp_indexer.config import CollectionConfig
from mcp_indexer.context import Context
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.models import Document, DocumentChunk
from mcp_indexer.plugins.file_source import FileSource
from mcp_indexer.retrieval import fetch_chunk

METADATA = '{"b": 0, "e": 12}'
SEMANTIC_TEXT = "abcdefghijkl"
EMBEDDING = [0.0] * VECTOR_DIMENSIONS


def make_source(test_context, tmp_path):
    (tmp_path / "article.txt").write_text(SEMANTIC_TEXT)
    config = CollectionConfig(
        collection_id="col", tool_prefix="col",
        source_blob={"directory": str(tmp_path)},
    )
    context = Context(
        engine=test_context.engine,
        session_factory=test_context.session_factory,
        embedding=None, llm=None, config=None,
    )
    source = FileSource("col", context=context, collection_config=config)
    return source


async def store_chunks(session, texts):
    """Store the document and its two embedding rows.

    ``texts`` is a pair (row 0 text, row 1 text); None leaves the column
    NULL (unretained). The rows partition SEMANTIC_TEXT into 6-char spans.
    """
    session.add(Document(
        collection_id="col", document_id="article.txt", title="Article",
        summary="", keywords=[], title_strength=0, embedding=EMBEDDING,
        last_modified=datetime.now(),
    ))
    for order, text in enumerate(texts):
        session.add(DocumentChunk(
            collection_id="col", document_id="article.txt",
            order=order, chunk_id=f"article.txt?c={order}",
            text=text, embedding=EMBEDDING, summary_span=None,
            semantic_chunk_id=0, embedding_offset=order * 6,
            embedding_length=6, semantic_length=12, metadata_str=METADATA,
        ))
    await session.commit()


@pytest.mark.asyncio
async def test_retained_text_returns_without_handle_acquisition(test_context, tmp_path, monkeypatch):
    source = make_source(test_context, tmp_path)
    async with test_context.session_factory() as session:
        await store_chunks(session, ("abcdef", "ghijkl"))

    async def no_handle(document_id):
        raise AssertionError("retained text must not acquire a handle")
    monkeypatch.setattr(source, "fetch_document", no_handle)

    # Semantic scope reconstructs the group from the retained rows, in order.
    assert await fetch_chunk(source, "article.txt?c=0", scope="semantic") == SEMANTIC_TEXT
    # Embedding scope returns the requested row's own retained text.
    assert await fetch_chunk(source, "article.txt?c=1", scope="embedding") == "ghijkl"


@pytest.mark.asyncio
async def test_semantic_scope_from_non_root_chunk_id(test_context, tmp_path, monkeypatch):
    source = make_source(test_context, tmp_path)
    async with test_context.session_factory() as session:
        await store_chunks(session, (None, None))

    acquired = []
    original = source.fetch_document

    async def spy(document_id):
        acquired.append(document_id)
        return await original(document_id)
    monkeypatch.setattr(source, "fetch_document", spy)

    # A non-root ID selects its semantic group; unretained rows fall back to
    # a fresh source read instead of assembling partial text.
    assert await fetch_chunk(source, "article.txt?c=1", scope="semantic") == SEMANTIC_TEXT
    assert acquired == ["article.txt"]
    # Embedding scope falls back the same way and applies the stored slice.
    assert await fetch_chunk(source, "article.txt?c=1", scope="embedding") == "ghijkl"
    assert acquired == ["article.txt", "article.txt"]


@pytest.mark.asyncio
async def test_incomplete_retained_text_falls_back_instead_of_assembling(test_context, tmp_path, monkeypatch):
    source = make_source(test_context, tmp_path)
    async with test_context.session_factory() as session:
        await store_chunks(session, ("abcdef", None))

    acquired = []
    original = source.fetch_document

    async def spy(document_id):
        acquired.append(document_id)
        return await original(document_id)
    monkeypatch.setattr(source, "fetch_document", spy)

    assert await fetch_chunk(source, "article.txt?c=0", scope="semantic") == SEMANTIC_TEXT
    assert acquired == ["article.txt"]


@pytest.mark.asyncio
async def test_missing_row_is_distinct_from_null_text(test_context, tmp_path, monkeypatch):
    source = make_source(test_context, tmp_path)
    async with test_context.session_factory() as session:
        await store_chunks(session, (None, None))

    async def no_handle(document_id):
        raise AssertionError("a missing row must not fall back to a handle")
    monkeypatch.setattr(source, "fetch_document", no_handle)

    with pytest.raises(LookupError):
        await fetch_chunk(source, "article.txt?c=9", scope="embedding")
    with pytest.raises(LookupError):
        await fetch_chunk(source, "article.txt?c=9", scope="semantic")
    # NULL text is not missing: the fallback runs and succeeds.
    monkeypatch.undo()
    assert await fetch_chunk(source, "article.txt?c=0", scope="semantic") == SEMANTIC_TEXT


@pytest.mark.asyncio
async def test_empty_retained_text_is_not_null(test_context, tmp_path, monkeypatch):
    source = make_source(test_context, tmp_path)
    async with test_context.session_factory() as session:
        await store_chunks(session, ("", ""))

    async def no_handle(document_id):
        raise AssertionError("empty retained text is retained, not unretained")
    monkeypatch.setattr(source, "fetch_document", no_handle)

    assert await fetch_chunk(source, "article.txt?c=0", scope="semantic") == ""
    assert await fetch_chunk(source, "article.txt?c=1", scope="embedding") == ""


@pytest.mark.asyncio
async def test_one_retrieval_statement_per_scope(test_context, tmp_path, monkeypatch):
    source = make_source(test_context, tmp_path)
    async with test_context.session_factory() as session:
        await store_chunks(session, ("abcdef", "ghijkl"))

    statements = []
    sync_engine = test_context.engine.sync_engine

    def count(_conn, _cursor, statement, _params, _context, _executemany):
        statements.append(statement)
    event.listen(sync_engine, "before_cursor_execute", count)
    try:
        await fetch_chunk(source, "article.txt?c=0", scope="semantic")
        assert len(statements) == 1
        statements.clear()
        await fetch_chunk(source, "article.txt?c=1", scope="embedding")
        assert len(statements) == 1
    finally:
        event.remove(sync_engine, "before_cursor_execute", count)


@pytest.mark.asyncio
async def test_text_column_is_deferred_in_entity_loads(test_context, tmp_path):
    async with test_context.session_factory() as session:
        await store_chunks(session, ("abcdef", "ghijkl"))
        chunk = (await session.execute(select(DocumentChunk).where(
            DocumentChunk.collection_id == "col",
            DocumentChunk.chunk_id == "article.txt?c=0",
        ))).scalars().one()
        # Entity loads omit the deferred column; async access cannot
        # lazy-load it, so no code can silently trigger a text load.
        with pytest.raises(MissingGreenlet):
            chunk.text
    # Column-level selects still retrieve it.
    async with test_context.session_factory() as session:
        value = (await session.execute(select(DocumentChunk.text).where(
            DocumentChunk.collection_id == "col",
            DocumentChunk.chunk_id == "article.txt?c=0",
        ))).scalar_one()
    assert value == "abcdef"
