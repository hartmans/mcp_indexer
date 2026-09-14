"""PostgreSQL checks to run once the existing test database is available."""
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import os

import pytest
from sqlalchemy import select, text, update

from mcp_indexer.config import CollectionConfig, ServerConfig
from mcp_indexer.indexer import Indexer, DocumentIndexingStats
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.models import Document, DocumentChunk, ChunkSummary, FailedDocument
from mcp_indexer.plugins.text_source import TextFileSource


def make_indexer(test_context, tmp_path):
    @asynccontextmanager
    async def session():
        async with test_context.session_factory() as value:
            yield value
    async def embedding(texts):
        return [[1.0] + [0.0] * (VECTOR_DIMENSIONS - 1) for _ in texts]
    async def llm(prompts):
        return ["An article summary." for _ in prompts]
    context = SimpleNamespace(
        get_session=session, embedding=embedding, llm=llm,
        config=SimpleNamespace(get_server_config=lambda: ServerConfig(min_size=1, max_size=20)),
    )
    config = CollectionConfig(collection_id="wiki", tool_prefix="wiki", indexing_mode="indexed",
                              source_blob={"directory": str(tmp_path)})
    source = TextFileSource("wiki", context=context, collection_config=config)
    context.collections = {"wiki": source}
    return Indexer(context), source


async def test_refresh_failure_and_deletion_are_atomic(test_context, tmp_path, monkeypatch):
    indexer, source = make_indexer(test_context, tmp_path)
    path = tmp_path / "article.txt"
    path.write_text("Original article content " * 5)
    await indexer.ensure_document_indexed(source, path.name)
    await indexer.summarize_document_chunks(source, path.name)
    await indexer.summarize_document(source, path.name)
    async with source.context.get_session() as session:
        original = await session.get(Document, (source.id, path.name))
        assert original.summary
        assert (await session.execute(select(ChunkSummary))).scalars().all()
        old_mtime = original.last_modified

    path.write_text("Replacement article")
    timestamp = (old_mtime - timedelta(days=1)).timestamp()
    os.utime(path, (timestamp, timestamp))
    original_embed = indexer._embed_batch
    async def fail(texts):
        raise RuntimeError("embedding failed")
    monkeypatch.setattr(indexer, "_embed_batch", fail)
    with pytest.raises(RuntimeError, match="embedding failed"):
        await indexer.index_document(source.fetch_document(path.name),
                                     DocumentIndexingStats(path.name, "refresh", source=source),
                                     replace=True)
    async with source.context.get_session() as session:
        kept = await session.get(Document, (source.id, path.name))
        assert kept.last_modified == old_mtime
        assert kept.summary == original.summary
    monkeypatch.setattr(indexer, "_embed_batch", original_embed)
    await indexer.maintain_collection(source)
    await indexer.wait_for_idle()
    async with source.context.get_session() as session:
        current = await session.get(Document, (source.id, path.name))
        assert current.last_modified == source.fetch_document(path.name).last_modified
        assert current.summary == ""
        assert not (await session.execute(select(ChunkSummary))).scalars().all()
        chunks = (await session.execute(select(DocumentChunk))).scalars().all()
        assert len(chunks) == 1
        assert chunks[0].summary_span is None
    await indexer._record_failure(source, path.name, "old failure")
    path.unlink()
    await indexer.maintain_collection(source)
    async with source.context.get_session() as session:
        for model in (Document, DocumentChunk, ChunkSummary, FailedDocument):
            assert not (await session.execute(select(model))).scalars().all()


async def test_collection_rename_preserves_local_ids(test_context, tmp_path):
    indexer, source = make_indexer(test_context, tmp_path)
    (tmp_path / "article.txt").write_text("Article")
    await indexer.ensure_document_indexed(source, "article.txt")
    await indexer.summarize_document_chunks(source, "article.txt")
    await indexer._record_failure(source, "failed-only.txt", "failure")
    async with source.context.get_session() as session:
        await session.execute(text("SET CONSTRAINTS ALL DEFERRED"))
        for model in (Document, DocumentChunk, ChunkSummary, FailedDocument):
            await session.execute(update(model).where(model.collection_id == "wiki")
                                  .values(collection_id="renamed"))
        await session.commit()
    async with source.context.get_session() as session:
        assert await session.get(Document, ("renamed", "article.txt")) is not None
        chunks = (await session.execute(select(DocumentChunk))).scalars().all()
        assert chunks[0].chunk_id == "article.txt?c=0"
        assert chunks[0].collection_id == "renamed"


async def test_manual_migration_from_legacy_constraints(test_context):
    async with test_context.engine.begin() as connection:
        await connection.exec_driver_sql("ALTER TABLE document_chunk DROP CONSTRAINT fk_document_chunk_document")
        await connection.exec_driver_sql("ALTER TABLE document_chunk DROP CONSTRAINT fk_document_chunk_summary")
        await connection.exec_driver_sql("ALTER TABLE chunk_summary DROP CONSTRAINT fk_chunk_summary_document")
        await connection.exec_driver_sql("""ALTER TABLE document_chunk ADD CONSTRAINT legacy_document_fk
            FOREIGN KEY (collection_id, document_id) REFERENCES document (collection_id, document_id)""")
        await connection.exec_driver_sql("""ALTER TABLE document_chunk ADD CONSTRAINT legacy_summary_fk
            FOREIGN KEY (collection_id, document_id, summary_span)
            REFERENCES chunk_summary (collection_id, document_id, summary_span)
            ON DELETE SET NULL (summary_span)""")
        await connection.exec_driver_sql("""INSERT INTO chunk_summary VALUES ('wiki', 'orphan', 0, 'orphan summary')""")
    migration = (Path(__file__).parents[1] / "scripts/migrate_native_sources.sql").read_text()
    # The test supplies the transaction; retain the DO block's BEGIN/END.
    migration = migration.replace("BEGIN;\n", "", 1).replace("\nCOMMIT;", "")
    async with test_context.engine.begin() as connection:
        await connection.execute(text(migration))
        count = (await connection.exec_driver_sql("SELECT count(*) FROM chunk_summary")).scalar_one()
        assert count == 0
        constraints = (await connection.exec_driver_sql("""SELECT conname, condeferrable, confdeltype
            FROM pg_constraint WHERE conname IN
            ('fk_document_chunk_document', 'fk_document_chunk_summary', 'fk_chunk_summary_document')""")).all()
        assert set(constraints) == {
            ("fk_document_chunk_document", True, "c"),
            ("fk_document_chunk_summary", True, "n"),
            ("fk_chunk_summary_document", True, "c"),
        }
