#!/usr/bin/env python3
"""Idempotent, resumable schema migration for the semantic-chunk layout.

Folded in (previously run separately as ``migrate_native_sources.sql``):
orphan ``chunk_summary`` cleanup and re-creation of the
``document_chunk``/``chunk_summary`` foreign keys as deferrable.

Adds to the legacy layout (``o``/``s`` char offsets inside each chunk's
``metadata_str`` JSON, no semantic-chunk grouping, no parent-document
hierarchy):

* ``document.parent_document_id`` (nullable; left nullable even when
  fully populated, to stay open for documents that are not sub-documents).
* ``document_chunk.semantic_chunk_id`` / ``embedding_offset`` /
  ``semantic_length`` / ``embedding_length``.

Ordered steps (each inspects before acting, so re-runs are no-ops):

1. Drop orphan ``chunk_summary`` rows (their document was removed).
2. Replace the chunk/summary FKs with the deferrable canonical set
   (any existing constraint on those endpoints is dropped first,
   whether PG-generated or named).
3. ``document.parent_document_id`` + deferrable self FK.
4. ``document_chunk`` columns + semantic self FK + order index.
5. ``semantic_chunk_id`` backfill: ``min(order)`` of each
   ``(collection, document, b, e)`` group.
6. ``embedding_offset`` backfill from metadata ``o`` (always a true
   value under the legacy chunker).
7. ``embedding_length`` backfill from metadata ``s`` for every row
   except the legacy tail-merge case: the last row of a group whose
   ``s == max_size``. The legacy chunker only ever mutated a chunk's
   text via that one tail-merge, so ``s`` is a true char length
   everywhere else.
8. File pass (threaded): for the flagged rows, read the semantic
   chunk's ``[b, e)`` bytes from the source file, decode in the
   source's encoding, and set ``embedding_length = char_len - o``.
9. ``semantic_length`` backfill: the sum of each semantic chunk
   group's ``embedding_length``. The legacy chunker tiles the semantic
   chunk with its embedding chunks, so the sum is the semantic chunk's
   char length.
10. Strip the ``o``/``s`` keys from ``metadata_str`` (every row is
    populated by this point; the metadata dict then belongs entirely to
    the source).
11. Verify zero ``NULL``s and ``SET NOT NULL`` on the four
    ``document_chunk`` columns; report counts.

Resume: unprocessed rows are identified by the columns being
``NULL``; nothing is dropped. An interruption loses at most the
in-flight batch, which is re-selected on restart.

Usage (per our standard argument processing):
    python scripts/migrate_semantic_schema.py --config config.toml
    python scripts/migrate_semantic_schema.py --config a.toml --config b.toml

``max_size`` is read from the server config (the tail-merge detector
depends on the value the corpus was indexed with); per-collection file
locations and encodings come from each collection's source config.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_indexer.config import ConfigManager
from mcp_indexer.models import serialize_metadata
from mcp_indexer.plugins.file_source import FileSourceConfig

SEMANTIC_CHUNK_ID = "semantic_chunk_id"
EMBEDDING_OFFSET = "embedding_offset"
SEMANTIC_LENGTH = "semantic_length"
EMBEDDING_LENGTH = "embedding_length"

CHUNK_COLS = (SEMANTIC_CHUNK_ID, EMBEDDING_OFFSET,
              SEMANTIC_LENGTH, EMBEDDING_LENGTH)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate document_chunk/document to the semantic-chunk "
                    "schema (idempotent, resumable).",
    )
    parser.add_argument(
        "-c", "--config", action="append", default=[], metavar="PATH",
        help="Config TOML file (repeatable; last wins)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the work that would be done without writing.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Concurrent file readers for the file pass (default: 8).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def table_columns(cur, table: str) -> set[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s", (table,),
    )
    return {r["column_name"] for r in cur.fetchall()}


def table_indexes(cur, table: str) -> set[str]:
    cur.execute(
        "SELECT indexname FROM pg_indexes "
        "WHERE schemaname = 'public' AND tablename = %s", (table,),
    )
    return {r["indexname"] for r in cur.fetchall()}


def fk_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM pg_constraint WHERE conname = %s", (name,))
    return cur.fetchone() is not None


def source_fks(cur) -> list[tuple[str, str, str, bool]]:
    """(constraint name, owning table, referenced table, deferrable) for
    FKs between document, document_chunk and chunk_summary — the set the
    native-sources migration owns (excluding the parent self-FK on
    document)."""
    cur.execute("""
        SELECT con.conname,
               con.conrelid::regclass::text AS own,
               con.confrelid::regclass::text AS ref,
               con.condeferrable AS deferrable
        FROM pg_constraint con
        WHERE con.contype = 'f'
          AND con.conrelid::regclass::text
              IN ('document_chunk', 'chunk_summary')
          AND con.confrelid::regclass::text IN ('document', 'chunk_summary')
    """)
    return [(r["conname"], r["own"], r["ref"], r["deferrable"])
            for r in cur.fetchall()]


def not_null(cur, table: str, column: str) -> bool:
    cur.execute(
        "SELECT is_nullable FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s "
        "AND column_name = %s", (table, column),
    )
    row = cur.fetchone()
    return row is not None and row["is_nullable"] == "NO"


def run(cur, sql: str, params: tuple = ()) -> int:
    cur.execute(sql, params)
    return cur.rowcount


# ---------------------------------------------------------------------------
# Steps 1-4: schema (folded-in native-sources SQL + new columns/FKs)
# ---------------------------------------------------------------------------

CANONICAL_FK_DDL = [
    "ALTER TABLE document_chunk ADD CONSTRAINT fk_document_chunk_document "
    "FOREIGN KEY (collection_id, document_id) REFERENCES "
    "document (collection_id, document_id) ON DELETE CASCADE DEFERRABLE",
    "ALTER TABLE chunk_summary ADD CONSTRAINT fk_chunk_summary_document "
    "FOREIGN KEY (collection_id, document_id) REFERENCES "
    "document (collection_id, document_id) ON DELETE CASCADE DEFERRABLE",
    "ALTER TABLE document_chunk ADD CONSTRAINT fk_document_chunk_summary "
    "FOREIGN KEY (collection_id, document_id, summary_span) REFERENCES "
    "chunk_summary (collection_id, document_id, summary_span) "
    "ON DELETE SET NULL (summary_span) DEFERRABLE",
]


def step_fks(cur, dry: bool) -> None:
    """Orphan summary cleanup + deferrable FKs (migrate_native_sources)."""
    if not dry:
        n = run(cur, """
            DELETE FROM chunk_summary s
            WHERE NOT EXISTS (
                SELECT 1 FROM document d
                WHERE d.collection_id = s.collection_id
                  AND d.document_id = s.document_id
            )
        """)
        print(f"1. orphan chunk_summary rows deleted: {n}")
    else:
        print("1. orphan chunk_summary cleanup: (dry-run)")

    existing = source_fks(cur)
    current = {(own, ref) for _name, own, ref, _def in existing}
    canonical = {
        ("document_chunk", "document"),
        ("chunk_summary", "document"),
        ("document_chunk", "chunk_summary"),
    }
    all_deferrable = all(defr for _n, _o, _r, defr in existing)
    if current == canonical and len(existing) == len(canonical) \
            and all_deferrable:
        print("2. source FKs already in the canonical deferrable state")
        return
    # Existing constraints may carry PostgreSQL-generated names; resolve
    # by endpoints and replace the whole set in one locked window.
    for name, own, _ref, _def in sorted(existing):
        if not dry:
            run(cur, f"ALTER TABLE {own} DROP CONSTRAINT {name}")
        print(f"2. dropped {name} ({own})" + (" (dry-run)" if dry else ""))
    if not dry:
        run(cur, "LOCK TABLE document, document_chunk, chunk_summary "
                 "IN ACCESS EXCLUSIVE MODE")
    for ddl in CANONICAL_FK_DDL:
        if not dry:
            run(cur, ddl)
    print("2. added deferrable canonical FKs" + (" (dry-run)" if dry else ""))


def step_parent(cur, dry: bool) -> None:
    if "parent_document_id" not in table_columns(cur, "document"):
        if not dry:
            run(cur, "ALTER TABLE document ADD COLUMN parent_document_id VARCHAR")
        print("3. added document.parent_document_id"
              + (" (dry-run)" if dry else ""))
    if not fk_exists(cur, "fk_document_parent"):
        if not dry:
            run(cur, """
                ALTER TABLE document ADD CONSTRAINT fk_document_parent
                FOREIGN KEY (collection_id, parent_document_id)
                REFERENCES document (collection_id, document_id)
                ON DELETE CASCADE DEFERRABLE
            """)
        print("3. added deferrable fk_document_parent"
              + (" (dry-run)" if dry else ""))


def step_chunk_columns(cur, dry: bool) -> None:
    cols = table_columns(cur, "document_chunk")
    for col in CHUNK_COLS:
        if col not in cols:
            if not dry:
                run(cur, f"ALTER TABLE document_chunk ADD COLUMN {col} INTEGER")
            print(f"4. added document_chunk.{col}" + (" (dry-run)" if dry else ""))
    if not fk_exists(cur, "fk_document_chunk_semantic"):
        # The self-FK's referenced key is the (collection, document, order)
        # unique constraint, which models.py declares but a legacy DB may
        # lack.
        cur.execute("""
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'document_chunk'::regclass AND contype = 'u'
        """)
        if cur.fetchone() is None:
            if not dry:
                run(cur, """
                    ALTER TABLE document_chunk ADD CONSTRAINT
                    uq_document_chunk_order
                    UNIQUE (collection_id, document_id, "order")
                """)
            print("4. added uq_document_chunk_order"
                  + (" (dry-run)" if dry else ""))
        if not dry:
            run(cur, """
                ALTER TABLE document_chunk ADD CONSTRAINT
                fk_document_chunk_semantic
                FOREIGN KEY (collection_id, document_id, semantic_chunk_id)
                REFERENCES document_chunk (collection_id, document_id, "order")
                DEFERRABLE
            """)
        print("4. added deferrable fk_document_chunk_semantic"
              + (" (dry-run)" if dry else ""))
    if "ix_document_chunk_order" not in table_indexes(cur, "document_chunk"):
        if not dry:
            run(cur, "CREATE INDEX ix_document_chunk_order "
                     "ON document_chunk (\"order\")")
        print("4. added ix_document_chunk_order" + (" (dry-run)" if dry else ""))


# ---------------------------------------------------------------------------
# Steps 5-7: pure-SQL backfills
# ---------------------------------------------------------------------------

def step_semantic_chunk_id(cur, dry: bool) -> None:
    if dry:
        print("5. semantic_chunk_id: (dry-run)")
        return
    # Group min is computed over every row (unfiltered) so a resume never
    # recomputes a group from a partial row set.
    n = run(cur, f"""
        UPDATE document_chunk c
        SET {SEMANTIC_CHUNK_ID} = g.min_order
        FROM (
            SELECT collection_id, document_id, b, e, MIN("order") AS min_order
            FROM (
                SELECT collection_id, document_id,
                       (metadata_str::json->>'b')::bigint AS b,
                       (metadata_str::json->>'e')::bigint AS e,
                       "order"
                FROM document_chunk
            ) t
            GROUP BY collection_id, document_id, b, e
        ) g
        WHERE c.collection_id = g.collection_id
          AND c.document_id = g.document_id
          AND c.{SEMANTIC_CHUNK_ID} IS NULL
          AND (c.metadata_str::json->>'b')::bigint = g.b
          AND (c.metadata_str::json->>'e')::bigint = g.e
    """)
    print(f"5. semantic_chunk_id: {n} rows updated")


def step_embedding_offset(cur, dry: bool) -> None:
    if dry:
        print("6. embedding_offset: (dry-run)")
        return
    n = run(cur, f"""
        UPDATE document_chunk
        SET {EMBEDDING_OFFSET} = (metadata_str::json->>'o')::bigint
        WHERE {EMBEDDING_OFFSET} IS NULL
    """)
    print(f"6. embedding_offset: {n} rows updated")


def step_embedding_length(cur, dry: bool, max_size: int) -> None:
    """Set embedding_length from metadata s for every non-flagged row.

    A flagged row is the last row of its (b, e) group with
    s == max_size: the legacy tail-merge may have grown its text
    without updating s, so those go to the file pass instead.
    """
    if dry:
        print(f"7. embedding_length (detector: s == {max_size}): (dry-run)")
        return
    n = run(cur, f"""
        UPDATE document_chunk c
        SET {EMBEDDING_LENGTH} = m.s
        FROM (
            SELECT collection_id, document_id, chunk_id,
                   (metadata_str::json->>'b')::bigint AS b,
                   (metadata_str::json->>'e')::bigint AS e,
                   (metadata_str::json->>'o')::bigint AS o,
                   (metadata_str::json->>'s')::bigint AS s
            FROM document_chunk
            WHERE {EMBEDDING_LENGTH} IS NULL
        ) m
        WHERE c.collection_id = m.collection_id
          AND c.document_id = m.document_id
          AND c.chunk_id = m.chunk_id
          AND NOT (
              m.s = %s
              AND NOT EXISTS (
                  SELECT 1 FROM document_chunk c2
                  WHERE c2.collection_id = m.collection_id
                    AND c2.document_id = m.document_id
                    AND (c2.metadata_str::json->>'b')::bigint = m.b
                    AND (c2.metadata_str::json->>'e')::bigint = m.e
                    AND (c2.metadata_str::json->>'o')::bigint > m.o
              )
          )
    """, (max_size,))
    print(f"7. embedding_length: {n} rows updated from metadata s")


# ---------------------------------------------------------------------------
# Step 8: threaded file pass over flagged rows
# ---------------------------------------------------------------------------

FLAGGED_SQL = """
    SELECT m.collection_id, m.document_id, m.chunk_id, m.b, m.e, m.o
    FROM (
        SELECT collection_id, document_id, chunk_id,
               (metadata_str::json->>'b')::bigint AS b,
               (metadata_str::json->>'e')::bigint AS e,
               (metadata_str::json->>'o')::bigint AS o,
               (metadata_str::json->>'s')::bigint AS s
        FROM document_chunk
        WHERE {null_filter}
    ) m
    WHERE m.s = %s
      AND NOT EXISTS (
          SELECT 1 FROM document_chunk c2
          WHERE c2.collection_id = m.collection_id
            AND c2.document_id = m.document_id
            AND (c2.metadata_str::json->>'b')::bigint = m.b
            AND (c2.metadata_str::json->>'e')::bigint = m.e
            AND (c2.metadata_str::json->>'o')::bigint > m.o
      )
"""


def build_tasks(cur, max_size: int,
                only_null: bool = True) -> list[dict]:
    """Flagged rows: the last row of a (b, e) group with s == max_size.

    ``only_null`` restricts to rows whose embedding_length is still NULL;
    the column may not exist yet on a fresh dry-run, so the caller may
    pass False in that case.
    """
    cur.execute(FLAGGED_SQL.format(
        null_filter=(f"{EMBEDDING_LENGTH} IS NULL" if only_null else "TRUE")),
        (max_size,))
    return cur.fetchall()


@dataclass
class FileTask:
    collection_id: str
    document_id: str
    chunk_id: str
    path: Path
    encoding: str
    b: int
    e: int
    o_last: int


def file_pass(rows: list[dict], source_info: dict[str, tuple[Path, str]],
              cur, workers: int, dry: bool) -> bool:
    """Read flagged semantic chunks and set their true embedding_length.

    Threads do the reads; all writes happen on the main thread, with one
    commit at the end. Returns True on success.
    """
    tasks: list[FileTask] = []
    for r in rows:
        directory, encoding = source_info[r["collection_id"]]
        relative = Path(unquote(r["document_id"]))
        if relative.is_absolute():
            print(f"  ERROR {r['collection_id']}:{r['document_id']}: "
                  f"absolute document path", file=sys.stderr)
            return False
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()):
            print(f"  ERROR {r['collection_id']}:{r['document_id']}: "
                  f"path outside source directory", file=sys.stderr)
            return False
        tasks.append(FileTask(
            collection_id=r["collection_id"], document_id=r["document_id"],
            chunk_id=r["chunk_id"], path=path, encoding=encoding,
            b=r["b"], e=r["e"], o_last=r["o"],
        ))
    if not tasks:
        print("8. file pass: no flagged rows")
        return True
    if dry:
        by_file = {t.path for t in tasks}
        print(f"8. file pass: {len(tasks)} flagged rows "
              f"across {len(by_file)} files (dry-run)")
        return True

    # Batch by file so each file is opened once for all its groups.
    by_path: dict[Path, list[tuple[int, FileTask]]] = {}
    for i, t in enumerate(tasks):
        by_path.setdefault(t.path, []).append((i, t))

    print(f"8. file pass: {len(tasks)} flagged rows "
          f"across {len(by_path)} files")
    results: dict[int, tuple[int | None, Exception | None]] = {}
    lock = threading.Lock()

    def worker(path: Path, items: list[tuple[int, FileTask]]) -> None:
        for i, t in items:
            try:
                with open(path, "rb") as fh:
                    fh.seek(t.b)
                    data = fh.read(t.e - t.b)
                char_len = len(data.decode(t.encoding))
                with lock:
                    results[i] = (char_len - t.o_last, None)
            except Exception as exc:  # noqa: BLE001 - report, abort
                with lock:
                    results[i] = (None, exc)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, p, items) for p, items in by_path.items()]
        for f in futures:
            f.result()

    errors = 0
    for i, t in enumerate(tasks):
        value, exc = results[i]
        if exc is not None:
            errors += 1
            print(f"  ERROR {t.collection_id}:{t.document_id} "
                  f"({t.chunk_id}): {exc}", file=sys.stderr)
    if errors:
        print(f"8. file pass: {errors} errors; aborting", file=sys.stderr)
        return False

    for i, t in enumerate(tasks):
        run(cur, f"""
            UPDATE document_chunk
            SET {EMBEDDING_LENGTH} = %s
            WHERE collection_id = %s AND document_id = %s AND chunk_id = %s
        """, (results[i][0], t.collection_id, t.document_id, t.chunk_id))
    print(f"8. file pass: {len(tasks)} rows updated from file reads")
    return True


# ---------------------------------------------------------------------------
# Steps 9-11
# ---------------------------------------------------------------------------

def step_semantic_length(cur, dry: bool) -> None:
    if dry:
        print("9. semantic_length: (dry-run)")
        return
    # Only update groups where every row has a length, so a partial
    # resume never writes a short sum.
    n = run(cur, f"""
        UPDATE document_chunk c
        SET {SEMANTIC_LENGTH} = s.total
        FROM (
            SELECT collection_id, document_id, {SEMANTIC_CHUNK_ID},
                   SUM({EMBEDDING_LENGTH}) AS total
            FROM document_chunk
            GROUP BY collection_id, document_id, {SEMANTIC_CHUNK_ID}
            HAVING COUNT(*) =
                   SUM(CASE WHEN {EMBEDDING_LENGTH} IS NOT NULL THEN 1
                            ELSE 0 END)
        ) s
        WHERE c.collection_id = s.collection_id
          AND c.document_id = s.document_id
          AND c.{SEMANTIC_CHUNK_ID} = s.{SEMANTIC_CHUNK_ID}
          AND c.{SEMANTIC_LENGTH} IS NULL
    """)
    print(f"9. semantic_length: {n} rows updated (sum of embedding lengths)")


def step_strip_metadata(cur, dry: bool) -> None:
    """Strip the o/s keys now that the columns own them."""
    if dry:
        print("10. metadata_str strip: (dry-run)")
        return
    cur.execute(f"""
        SELECT collection_id, document_id, chunk_id, metadata_str
        FROM document_chunk
        WHERE {EMBEDDING_OFFSET} IS NOT NULL
          AND {EMBEDDING_LENGTH} IS NOT NULL
          AND metadata_str::jsonb ?| ARRAY['o', 's']
    """)
    rows = cur.fetchall()
    stripped = 0
    for row in rows:
        meta = json.loads(row["metadata_str"])
        meta.pop("o", None)
        meta.pop("s", None)
        new_str = serialize_metadata(meta)
        if new_str != row["metadata_str"]:
            run(cur, """
                UPDATE document_chunk SET metadata_str = %s
                WHERE collection_id = %s AND document_id = %s
                  AND chunk_id = %s
            """, (new_str, row["collection_id"], row["document_id"],
                  row["chunk_id"]))
            stripped += 1
    print(f"10. metadata_str: stripped o/s from {stripped} rows")


def step_finalize(cur, dry: bool) -> bool:
    if dry:
        print("11. verify + SET NOT NULL: (dry-run)")
        return True
    total_null = 0
    for col in CHUNK_COLS:
        cur.execute(
            f"SELECT COUNT(*) AS n FROM document_chunk WHERE {col} IS NULL")
        nulls = cur.fetchone()["n"]
        total_null += nulls
        if nulls:
            print(f"11. {col}: {nulls} rows still NULL", file=sys.stderr)
    if total_null:
        print("11. aborting; re-run to resume", file=sys.stderr)
        return False
    for col in CHUNK_COLS:
        if not not_null(cur, "document_chunk", col):
            run(cur, f"ALTER TABLE document_chunk "
                     f"ALTER COLUMN {col} SET NOT NULL")
    print("11. all four columns populated; SET NOT NULL applied")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config:
        print("error: at least one --config is required", file=sys.stderr)
        return 2
    config = ConfigManager(args.config)
    server = config.get_server_config()
    max_size = server.max_size
    # The server config may carry a SQLAlchemy URI; psycopg takes the
    # plain postgresql scheme.
    db_uri = server.db_uri.replace("+psycopg", "")

    source_info: dict[str, tuple[Path, str]] = {}
    for cid in config.list_collections():
        col = config.get_collection_config(cid)
        scfg = col.resolve_source_config(FileSourceConfig)
        source_info[cid] = (scfg.directory.resolve(), scfg.encoding)

    conn = psycopg.connect(db_uri, row_factory=dict_row,
                           autocommit=False)
    cur = conn.cursor()
    dry = args.dry_run
    ok = True

    try:
        step_fks(cur, dry)
        step_parent(cur, dry)
        step_chunk_columns(cur, dry)
        if not dry:
            conn.commit()

        step_semantic_chunk_id(cur, dry)
        step_embedding_offset(cur, dry)
        step_embedding_length(cur, dry, max_size)
        if not dry:
            conn.commit()

        if (not dry
                or EMBEDDING_LENGTH in table_columns(cur, "document_chunk")):
            flagged = build_tasks(cur, max_size, only_null=True)
        else:
            # Fresh dry-run: the columns do not exist yet, so select the
            # flagged set from the metadata alone.
            flagged = build_tasks(cur, max_size, only_null=False)
        ok = file_pass(flagged, source_info, cur, args.workers, dry)
        if ok and not dry:
            conn.commit()

        if ok:
            step_semantic_length(cur, dry)
            if not dry:
                conn.commit()

        if ok:
            step_strip_metadata(cur, dry)
            ok = step_finalize(cur, dry)
        if ok and not dry:
            conn.commit()
            print("migration complete")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
