"""PostgreSQL checks to run once the existing test database is available."""
from contextlib import asynccontextmanager
import asyncio
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import json
import os

import pytest
from sqlalchemy import select, text, update

from mcp_indexer.config import CollectionConfig, ServerConfig
from mcp_indexer.indexer import Indexer, DocumentIndexingStats
from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.models import Document, DocumentChunk, ChunkSummary, FailedDocument
from mcp_indexer.plugins.file_source import FileSource


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
    source = FileSource("wiki", context=context, collection_config=config)
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
        await indexer.index_document(await source.fetch_document(path.name),
                                     DocumentIndexingStats(path.name, "refresh", source=source))
    async with source.context.get_session() as session:
        kept = await session.get(Document, (source.id, path.name))
        assert kept.last_modified == old_mtime
        assert kept.summary == original.summary
    monkeypatch.setattr(indexer, "_embed_batch", original_embed)
    # The handle cache is process-local. A document whose handle is already
    # cached is not re-checked within this process: its (stale) mtime is
    # accepted, and change/deletion are noticed by a fresh process, which
    # starts with an empty cache. A new source simulates that restart.
    _, fresh_source = make_indexer(test_context, tmp_path)
    await indexer.maintain_collection(fresh_source)
    await indexer.wait_for_idle()
    async with fresh_source.context.get_session() as session:
        current = await session.get(Document, (source.id, path.name))
        assert current.last_modified == (await fresh_source.fetch_document(path.name)).last_modified
        # The refresh cleared the stale summary, and the completion trigger
        # regenerated it (from the handle summary) while the refresh settled.
        assert current.summary == original.summary
        assert not (await session.execute(select(ChunkSummary))).scalars().all()
        chunks = (await session.execute(select(DocumentChunk))).scalars().all()
        assert len(chunks) == 1
        assert chunks[0].summary_span is None
    await indexer._record_failure(fresh_source, path.name, "old failure")
    path.unlink()
    _, fresh_source2 = make_indexer(test_context, tmp_path)
    await indexer.maintain_collection(fresh_source2)
    async with fresh_source2.context.get_session() as session:
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
    # The indexing trigger scheduled a summary task; let it settle before
    # the test's database fixture is torn down.
    await indexer.wait_for_idle()


async def test_summary_submissions_cross_semantic_boundaries(test_context, tmp_path, monkeypatch):
    indexer, source = make_indexer(test_context, tmp_path)
    (tmp_path / "article.md").write_text("First section.\n\n# Second\nSecond section.")
    await indexer.ensure_document_indexed(source, "article.md")
    chunks = await indexer._reconstruct_semantic_chunks(source, "article.md")
    assert len(chunks) == 2
    started = []
    ready = asyncio.Event()

    async def summarize(text, prompt):
        started.append(text)
        if len(started) == 2:
            ready.set()
        await ready.wait()
        return "Summary: " + text

    monkeypatch.setattr(indexer, "_summarize_text", summarize)
    await asyncio.wait_for(indexer.summarize_document_chunks(source, "article.md"), 5)
    async with source.context.get_session() as session:
        summaries = (await session.execute(
            select(ChunkSummary).order_by(ChunkSummary.summary_span)
        )).scalars().all()
        assert [s.summary for s in summaries] == ["Summary: " + c.text for c in chunks]
    # Chunk summarization triggered a document summary task; settle it
    # before teardown.
    await indexer.wait_for_idle()


async def test_indexing_normalizes_metadata_keywords(test_context, tmp_path, monkeypatch):
    indexer, source = make_indexer(test_context, tmp_path)
    (tmp_path / "article.txt").write_text("Article")
    pointer = await source.fetch_document("article.txt")

    async def metadata():
        return {"title": "Article", "keywords": ["Good_Key", "bad keyword", "GOOD_KEY"]}

    monkeypatch.setattr(pointer, "get_metadata", metadata)
    await indexer.index_document(
        pointer, DocumentIndexingStats(pointer.document_id, "index", source=source),
    )
    async with source.context.get_session() as session:
        document = await session.get(Document, (source.id, pointer.document_id))
        assert document.keywords == ["good-key"]
    # Indexing triggered a document summary task; settle it before the
    # test's database fixture is torn down.
    await indexer.wait_for_idle()


async def test_manual_migration_from_legacy_constraints(test_context, pg_url):
    import importlib.util
    import sys
    import psycopg
    from psycopg.rows import dict_row

    spec = importlib.util.spec_from_file_location(
        "migrate_schema",
        Path(__file__).parents[1] / "scripts" / "migrate_schema.py",
    )
    migration = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = migration
    spec.loader.exec_module(migration)

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
    # The script owns the transaction for the FK step.
    conn = psycopg.connect(pg_url.replace("+psycopg", ""), row_factory=dict_row,
                           autocommit=False)
    try:
        migration.step_fks(conn.cursor(), dry=False)
        count = conn.cursor().execute(
            "SELECT count(*) AS n FROM chunk_summary").fetchone()["n"]
        assert count == 0
        constraints = conn.cursor().execute("""SELECT conname, condeferrable, confdeltype
            FROM pg_constraint WHERE conname IN
            ('fk_document_chunk_document', 'fk_document_chunk_summary',
             'fk_chunk_summary_document')""").fetchall()
        assert {(c["conname"], c["condeferrable"], c["confdeltype"])
                for c in constraints} == {
            ("fk_document_chunk_document", True, "c"),
            ("fk_document_chunk_summary", True, "n"),
            ("fk_chunk_summary_document", True, "c"),
        }
        conn.commit()
    finally:
        conn.close()


async def test_migration_from_semantic_complete_state(test_context, pg_url,
                                                      tmp_path):
    """migrate_schema.py completes a database whose semantic phase has
    already run but whose subdocument DDL has not: phase 1 no-ops,
    phase 2 applies, the exit code is 0, and a re-run is a no-op."""
    import importlib.util
    import sys
    from sqlalchemy import text

    spec = importlib.util.spec_from_file_location(
        "migrate_schema",
        Path(__file__).parents[1] / "scripts" / "migrate_schema.py",
    )
    migration = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = migration
    spec.loader.exec_module(migration)

    # The fixture created the full current schema; roll back the
    # subdocument parts to simulate a semantic-complete database.
    async with test_context.engine.begin() as connection:
        await connection.exec_driver_sql("DROP INDEX ix_indexing_work_parent")
        await connection.exec_driver_sql("DROP TABLE indexing_work")
        await connection.exec_driver_sql(
            "ALTER TABLE document DROP COLUMN metadata_dict")

    config_path = tmp_path / "migration.toml"
    config_path.write_text(f"[server]\ndb_uri = \"{pg_url}\"\n")

    assert migration.main(["--config", str(config_path)]) == 0

    async with test_context.engine.connect() as connection:
        cols = (await connection.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'document' "
            "AND column_name = 'metadata_dict'"))).fetchall()
        assert len(cols) == 1
        tables = (await connection.execute(text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' "
            "AND table_name = 'indexing_work'"))).fetchall()
        assert len(tables) == 1
        indexes = (await connection.execute(text(
            "SELECT 1 FROM pg_indexes "
            "WHERE indexname = 'ix_indexing_work_parent'"))).fetchall()
        assert len(indexes) == 1
        fks = (await connection.execute(text(
            "SELECT condeferrable FROM pg_constraint "
            "WHERE conname = 'fk_indexing_work_parent'"))).fetchall()
        assert len(fks) == 1 and fks[0][0] is True

    # Fully migrated state: a second run is a complete no-op.
    assert migration.main(["--config", str(config_path)]) == 0


async def test_migration_backfill_derives_semantic_columns(test_context,
                                                            pg_url, tmp_path):
    """The backfill steps derive the semantic columns from legacy metadata.

    Seeds two documents with legacy chunk metadata (b, e, o, s), relaxes
    the NOT NULL constraints to simulate the pre-backfill state, then
    runs the backfill steps — including the file pass against a real
    file — and asserts the derived values.
    """
    import importlib.util
    import sys
    import psycopg
    from psycopg.rows import dict_row

    spec = importlib.util.spec_from_file_location(
        "migrate_schema",
        Path(__file__).parents[1] / "scripts" / "migrate_schema.py",
    )
    migration = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = migration
    spec.loader.exec_module(migration)

    conn = psycopg.connect(pg_url.replace("+psycopg", ""),
                           row_factory=dict_row, autocommit=False)
    cur = conn.cursor()
    try:
        zero_vec = "[" + ",".join(["0.0"] * VECTOR_DIMENSIONS) + "]"

        # Post-DDL, pre-backfill: the columns exist but are nullable and
        # empty (create_all makes two of them NOT NULL; relax that).
        for col in ("semantic_chunk_id", "embedding_offset"):
            cur.execute(
                f"ALTER TABLE document_chunk "
                f"ALTER COLUMN {col} DROP NOT NULL")

        for doc in ("doc-a.txt", "doc-b.txt"):
            cur.execute(
                "INSERT INTO document (collection_id, document_id, title, "
                "title_strength, embedding, keywords, summary, "
                "last_modified) VALUES ('col1', %s, %s, 1, %s, '{}', "
                "'s', now())", (doc, doc, zero_vec))
        # Legacy rows: doc-a is one semantic chunk [0, 1000) tiled by two
        # embedding chunks (last s != max_size, so no tail-merge flag);
        # doc-b's last chunk has s == max_size, so it is flagged and its
        # length comes from the file (1500 chars), not from s.
        rows = [
            ("doc-a.txt", 0, "a0", 0, 1000, 0, 300),
            ("doc-a.txt", 1, "a1", 0, 1000, 300, 700),
            ("doc-b.txt", 0, "b0", 0, 1500, 0, 500),
            ("doc-b.txt", 1, "b1", 0, 1500, 500, 500),
        ]
        for doc, order, chunk, b, e, o, s in rows:
            cur.execute(
                "INSERT INTO document_chunk (collection_id, document_id, "
                "chunk_id, \"order\", embedding, summary_span, "
                "semantic_chunk_id, embedding_offset, semantic_length, "
                "embedding_length, metadata_str) "
                "VALUES ('col1', %s, %s, %s, %s, NULL, NULL, NULL, NULL, "
                "NULL, %s)",
                (doc, chunk, order, zero_vec,
                 json.dumps({"b": b, "e": e, "o": o, "s": s})))
        conn.commit()

        (tmp_path / "doc-b.txt").write_text("x" * 1500)
        source_info = {"col1": (tmp_path, "utf-8")}

        migration.step_semantic_chunk_id(cur, dry=False)
        migration.step_embedding_offset(cur, dry=False)
        assert migration.ensure_flagged_table(
            cur, max_size=500, only_null=True) == 1
        migration.step_embedding_length(cur, dry=False, max_size=500)
        conn.commit()
        migration.file_pass(migration.select_flagged_rows(cur),
                            source_info, cur, 1, dry=False)
        conn.commit()
        migration.step_semantic_length(cur, dry=False)
        migration.step_strip_metadata(cur, dry=False)
        conn.commit()
        assert migration.step_finalize(cur, dry=False)
        conn.commit()

        def row(doc, order):
            cur.execute(
                "SELECT semantic_chunk_id, embedding_offset, "
                "embedding_length, semantic_length, metadata_str "
                "FROM document_chunk WHERE document_id = %s AND "
                "\"order\" = %s", (doc, order))
            r = cur.fetchone()
            return (r["semantic_chunk_id"], r["embedding_offset"],
                    r["embedding_length"], r["semantic_length"],
                    json.loads(r["metadata_str"]))

        assert row("doc-a.txt", 0) == (0, 0, 300, 1000, {"b": 0, "e": 1000})
        assert row("doc-a.txt", 1) == (0, 300, 700, 1000, {"b": 0, "e": 1000})
        assert row("doc-b.txt", 0) == (0, 0, 500, 1500, {"b": 0, "e": 1500})
        # The flagged row got its length from the file read, not s.
        assert row("doc-b.txt", 1) == (0, 500, 1000, 1500, {"b": 0, "e": 1500})
    finally:
        conn.close()
