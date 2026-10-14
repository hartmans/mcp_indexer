#!/usr/bin/env python3
"""Idempotent, resumable schema migration for the current database layout.

Combined from ``migrate_semantic_schema.py`` (the semantic-chunk layout,
itself folded in from ``migrate_native_sources.sql``) and
``migrate_subdoc_schema.py`` (the subdocument spike's storage), so a
database reaches the current schema with exactly one script, from any
starting state: a fresh legacy database, a database where the semantic
phase already ran (phase 1 is then a no-op and phase 2 applies), or a
fully migrated database (complete no-op).

Phase 1 — semantic-chunk schema. Adds to the legacy layout
(``o``/``s`` char offsets inside each chunk's ``metadata_str`` JSON, no
semantic-chunk grouping, no parent-document hierarchy):

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

Phase 2 — subdocument schema. Adds the storage introduced by the
subdocument/Docling spike:

* ``document.metadata_dict`` (JSON, nullable) — the JSON-compatible
  remainder of ``get_metadata()`` after title and keywords are stored in
  their dedicated columns.
* ``indexing_work`` — durable pending sub-document work: the full
  collection-local ID of a discovered child (the target) plus its
  parent, published atomically with the parent's own rows and retired
  when the target is indexed or fails.

Ordered steps (each inspects before acting, so re-runs are no-ops):

12. ``document.metadata_dict`` (JSON, nullable).
13. ``indexing_work`` table (composite primary key
    ``(collection_id, document_id)``, NOT NULL ``parent_document_id``).
14. ``ix_indexing_work_parent`` index.
15. Deferrable ``fk_indexing_work_parent`` (the parent must have a
    Document row; CASCADE removes a parent's work entries with it).

Phase 2 is pure DDL with no data dependency on phase 1's backfills, so
it is applied even when phase 1's backfill is still pending; the exit
code is then 1 and a re-run resumes phase 1 while phase 2 no-ops.

Steps 4 and 15 also replace non-canonical FK shapes — an
INITIALLY DEFERRED self-FK or work FK from an earlier model revision,
or a cascading self-FK — so every re-run converges toward the
canonical constraints.

``document_chunk.text`` (exact-text retention) already exists in any
database created after the model added the column; this migration does
not touch it.

Resume (phase 1): unprocessed rows are identified by the columns being
``NULL``; nothing is dropped. An interruption loses at most the
in-flight batch, which is re-selected on restart.

Usage (per our standard argument processing):
    python scripts/migrate_schema.py --config config.toml
    python scripts/migrate_schema.py --config a.toml --config b.toml
    python scripts/migrate_schema.py --config config.toml --dry-run

``max_size`` is read from the server config (the tail-merge detector
depends on the value the corpus was indexed with); per-collection file
locations and encodings come from each collection's source config.

DDL steps take ``ACCESS EXCLUSIVE`` locks; run with writers stopped.

The backfill steps are sized for multi-million-row tables: group
aggregates run as window functions joined back on the full primary key
(``collection_id``, ``document_id``, ``chunk_id``), and the
session sets ``work_mem``/``maintenance_work_mem`` to 4GB and
``synchronous_commit = off`` (resumable bulk migration).
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
        description="Migrate to the current schema (semantic-chunk layout "
                    "plus subdocument storage; idempotent, resumable).",
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


def table_exists(cur, table: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s", (table,),
    )
    return cur.fetchone() is not None


def column_exists(cur, table: str, column: str) -> bool:
    return column in table_columns(cur, table)


def constraint_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM pg_constraint WHERE conname = %s", (name,))
    return cur.fetchone() is not None


def index_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", (name,))
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


# Canonical FK states: (condeferrable, condeferred, confdeltype) — both
# deferrable and initially immediate; ON DELETE CASCADE ('c') for the
# work FK, NO ACTION ('a') for the semantic self-FK (a document's chunk
# rows are deleted as a set, or not at all). Databases created from an
# earlier model revision carry other shapes (INITIALLY DEFERRED,
# ON DELETE CASCADE on the self-FK); steps 4 and 15 replace any
# non-canonical state so re-runs converge.
SELF_FK_STATE = (True, False, "a")
WORK_FK_STATE = (True, False, "c")


def fk_state(cur, name: str) -> tuple[bool, bool, str] | None:
    """(condeferrable, condeferred, confdeltype) for a constraint,
    or None if it does not exist."""
    cur.execute("""
        SELECT condeferrable, condeferred, confdeltype
        FROM pg_constraint WHERE conname = %s
    """, (name,))
    row = cur.fetchone()
    if row is None:
        return None
    return (row["condeferrable"], row["condeferred"], row["confdeltype"])


def run(cur, sql: str, params: tuple = ()) -> int:
    cur.execute(sql, params)
    return cur.rowcount


# ---------------------------------------------------------------------------
# Phase 1, steps 1-4: schema (folded-in native-sources SQL + new columns/FKs)
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

# Canonical shape for the two FKs whose earlier revisions differed from
# models.py (the semantic self-FK was INITIALLY DEFERRED or cascading;
# the work FK was INITIALLY DEFERRED). Both are deferrable and initially
# immediate; the work FK cascades, the self-FK is NO ACTION.
SELF_FK_DDL = """
    ALTER TABLE document_chunk ADD CONSTRAINT fk_document_chunk_semantic
    FOREIGN KEY (collection_id, document_id, semantic_chunk_id)
    REFERENCES document_chunk (collection_id, document_id, "order")
    DEFERRABLE
"""

WORK_FK_DDL = """
    ALTER TABLE indexing_work ADD CONSTRAINT fk_indexing_work_parent
    FOREIGN KEY (collection_id, parent_document_id)
    REFERENCES document (collection_id, document_id)
    ON DELETE CASCADE DEFERRABLE
"""


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
    if not constraint_exists(cur, "fk_document_parent"):
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
    state = fk_state(cur, "fk_document_chunk_semantic")
    if state is None:
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
            run(cur, SELF_FK_DDL)
        print("4. added deferrable fk_document_chunk_semantic"
              + (" (dry-run)" if dry else ""))
    elif state != SELF_FK_STATE:
        # Databases from an earlier model revision carry an
        # INITIALLY DEFERRED self-FK; replace it so re-runs converge.
        if not dry:
            run(cur, "ALTER TABLE document_chunk DROP CONSTRAINT "
                     "fk_document_chunk_semantic")
            run(cur, SELF_FK_DDL)
        print("4. replaced non-canonical fk_document_chunk_semantic"
              + (" (dry-run)" if dry else ""))
    # Note: models.py declares no standalone indexes on document_chunk's
    # document_id or order — those are only meaningful scoped to
    # (collection_id, document_id), which the uq_document_chunk_order
    # index serves. Databases created from an earlier model revision may
    # still carry ix_document_chunk_order; drop it by hand.


# ---------------------------------------------------------------------------
# Phase 1, steps 5-7: pure-SQL backfills
# ---------------------------------------------------------------------------

def step_semantic_chunk_id(cur, dry: bool) -> None:
    if dry:
        print("5. semantic_chunk_id: (dry-run)")
        return
    # Group min is computed over every row (unfiltered) so a resume never
    # recomputes a group from a partial row set. The MIN() window is one
    # scan + sort of the whole table. Join back on the full primary key;
    # chunk_id alone is not guaranteed to be globally unique.
    n = run(cur, f"""
        UPDATE document_chunk c
        SET {SEMANTIC_CHUNK_ID} = f.min_order
        FROM (
            SELECT collection_id, document_id, chunk_id,
                   MIN("order") OVER (
                       PARTITION BY collection_id, document_id,
                           (metadata_str::json->>'b')::bigint,
                           (metadata_str::json->>'e')::bigint
                   ) AS min_order
            FROM document_chunk
        ) f
        WHERE c.chunk_id = f.chunk_id
          AND c.collection_id = f.collection_id
          AND c.document_id = f.document_id
          AND c.{SEMANTIC_CHUNK_ID} IS NULL
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
    without updating s, so those go to the file pass instead. The
    flagged set is materialized in the session-local flagged_rows
    temp table by ensure_flagged_table, which must run first.
    """
    if dry:
        print(f"7. embedding_length (detector: s == {max_size}): (dry-run)")
        return
    n = run(cur, f"""
        UPDATE document_chunk c
        SET {EMBEDDING_LENGTH} = m.s
        FROM (
            SELECT collection_id, document_id, chunk_id,
                   (metadata_str::json->>'s')::bigint AS s
            FROM document_chunk
            WHERE {EMBEDDING_LENGTH} IS NULL
              AND (collection_id, document_id, chunk_id) NOT IN (
                  SELECT collection_id, document_id, chunk_id FROM flagged_rows
              )
        ) m
        WHERE c.chunk_id = m.chunk_id
          AND c.collection_id = m.collection_id
          AND c.document_id = m.document_id
    """)
    print(f"7. embedding_length: {n} rows updated from metadata s")


# ---------------------------------------------------------------------------
# Phase 1, step 8: threaded file pass over flagged rows
# ---------------------------------------------------------------------------

# Flagged rows for the file pass: the last row (by `o`) of each legacy
# `(b, e)` group whose `s` equals max_size — the legacy tail-merge may
# have grown that chunk's text without updating s. The MAX() window is
# one scan + sort of the table; the previous formulation was a
# correlated anti-join that probed every row of each group with JSON
# comparisons, which is O(n^2) on a multi-million-row table.
FLAGGED_WINDOW_SQL = """
    SELECT chunk_id, collection_id, document_id, b, e, o
    FROM (
        SELECT chunk_id, collection_id, document_id,
               (metadata_str::json->>'b')::bigint AS b,
               (metadata_str::json->>'e')::bigint AS e,
               (metadata_str::json->>'o')::bigint AS o,
               (metadata_str::json->>'s')::bigint AS s,
               MAX((metadata_str::json->>'o')::bigint) OVER (
                   PARTITION BY collection_id, document_id,
                       (metadata_str::json->>'b')::bigint,
                       (metadata_str::json->>'e')::bigint
               ) AS group_max_o
        FROM document_chunk
        {null_filter}
    ) t
    WHERE o = group_max_o AND s = %s
"""


def ensure_flagged_table(cur, max_size: int, only_null: bool) -> int:
    """(Re)build the session-local flagged_rows temp table.

    ON COMMIT PRESERVE ROWS keeps it across the commit between the
    backfill steps and the file pass; it is dropped automatically when
    the session ends. ``only_null`` restricts to rows whose
    embedding_length is still NULL (the column may not exist on a fresh
    dry-run, so the caller passes False there). Harmless in a dry-run:
    it is session scratch, never written to the catalog. Returns the
    flagged count.
    """
    sql = (
        "CREATE TEMP TABLE flagged_rows ON COMMIT PRESERVE ROWS AS\n"
        + FLAGGED_WINDOW_SQL.format(null_filter=(
            f"WHERE {EMBEDDING_LENGTH} IS NULL" if only_null else "WHERE TRUE"))
    )
    cur.execute(sql, (max_size,))
    return cur.rowcount


def select_flagged_rows(cur) -> list[dict]:
    cur.execute("""
        SELECT chunk_id, collection_id, document_id, b, e, o
        FROM flagged_rows
        ORDER BY collection_id, document_id, o
    """)
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
# Phase 1, steps 9-11
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


# ---------------------------------------------------------------------------
# Phase 2, steps 12-15: subdocument storage
# ---------------------------------------------------------------------------

def step_metadata_dict(cur, dry: bool) -> None:
    if not column_exists(cur, "document", "metadata_dict"):
        if not dry:
            run(cur, "ALTER TABLE document ADD COLUMN metadata_dict JSON")
        print("12. added document.metadata_dict"
              + ("" if not dry else " (dry-run)"))
    else:
        print("12. document.metadata_dict already present")


def step_indexing_work_table(cur, dry: bool) -> None:
    if not table_exists(cur, "indexing_work"):
        if not dry:
            run(cur, """
                CREATE TABLE indexing_work (
                    collection_id VARCHAR NOT NULL,
                    document_id VARCHAR NOT NULL,
                    parent_document_id VARCHAR NOT NULL,
                    PRIMARY KEY (collection_id, document_id)
                )
            """)
        print("13. created indexing_work" + ("" if not dry else " (dry-run)"))
    else:
        # A partial run may have created the table without the column;
        # verify the canonical shape.
        if not column_exists(cur, "indexing_work", "parent_document_id"):
            if not dry:
                run(cur, "ALTER TABLE indexing_work "
                        "ADD COLUMN parent_document_id VARCHAR NOT NULL")
            print("13. added indexing_work.parent_document_id"
                  + ("" if not dry else " (dry-run)"))
        else:
            print("13. indexing_work already present")


def step_indexing_work_index(cur, dry: bool) -> None:
    if not index_exists(cur, "ix_indexing_work_parent"):
        if not dry:
            run(cur, "CREATE INDEX ix_indexing_work_parent "
                    "ON indexing_work (collection_id, parent_document_id)")
        print("14. added ix_indexing_work_parent"
              + ("" if not dry else " (dry-run)"))
    else:
        print("14. ix_indexing_work_parent already present")


def step_indexing_work_fk(cur, dry: bool) -> None:
    state = fk_state(cur, "fk_indexing_work_parent")
    if state is None:
        if not dry:
            run(cur, WORK_FK_DDL)
        print("15. added deferrable fk_indexing_work_parent"
              + (" (dry-run)" if dry else ""))
    elif state != WORK_FK_STATE:
        # Migrations from an earlier revision created this FK
        # INITIALLY DEFERRED; replace it so re-runs converge.
        if not dry:
            run(cur, "ALTER TABLE indexing_work DROP CONSTRAINT "
                     "fk_indexing_work_parent")
            run(cur, WORK_FK_DDL)
        print("15. replaced non-canonical fk_indexing_work_parent"
              + (" (dry-run)" if dry else ""))
    else:
        print("15. fk_indexing_work_parent already present")


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
    # Session settings for multi-million-row backfills: keep the window
    # sorts, hash tables, and the one-off index build in RAM (with the
    # 4MB work_mem default the step-5 aggregate spills to disk and can
    # take hours on a 6M-row table), and skip per-commit fsyncs — the
    # migration is resumable, so a crash loses at most the in-flight
    # batch, which the resume re-selects.
    cur = conn.cursor()
    cur.execute("SET work_mem = '4GB'")
    cur.execute("SET maintenance_work_mem = '4GB'")
    cur.execute("SET synchronous_commit = off")
    dry = args.dry_run
    sem_ok = True

    try:
        print("== semantic-chunk schema ==")
        step_fks(cur, dry)
        step_parent(cur, dry)
        step_chunk_columns(cur, dry)
        if not dry:
            conn.commit()

        step_semantic_chunk_id(cur, dry)
        step_embedding_offset(cur, dry)
        only_null = (not dry
                     or EMBEDDING_LENGTH in table_columns(cur,
                                                          "document_chunk"))
        # Session scratch shared by step 7 (exclusion) and the file pass
        # (read): the flagged set, computed once. Harmless in a dry-run.
        n_flagged = ensure_flagged_table(cur, max_size, only_null)
        print(f"7. flagged rows (detector: s == {max_size}): {n_flagged}")
        step_embedding_length(cur, dry, max_size)
        if not dry:
            conn.commit()

        flagged = select_flagged_rows(cur)
        sem_ok = file_pass(flagged, source_info, cur, args.workers, dry)
        if sem_ok and not dry:
            conn.commit()

        if sem_ok:
            step_semantic_length(cur, dry)
            if not dry:
                conn.commit()

        if sem_ok:
            step_strip_metadata(cur, dry)
            # Commit before finalize: databases created from an earlier
            # model revision still carry an INITIALLY DEFERRED self-FK,
            # and the strip's UPDATEs leave pending trigger events that
            # would block its SET NOT NULL ALTERs.
            if not dry:
                conn.commit()
            sem_ok = step_finalize(cur, dry)

        # Phase 2 is pure DDL with no data dependency on the semantic
        # backfills, so it proceeds even when the backfill is pending.
        print("== subdocument schema ==")
        step_metadata_dict(cur, dry)
        step_indexing_work_table(cur, dry)
        step_indexing_work_index(cur, dry)
        step_indexing_work_fk(cur, dry)
        if not dry:
            conn.commit()

        if sem_ok and not dry:
            print("migration complete")
        elif not sem_ok:
            print("subdocument DDL applied; semantic backfill "
                  "incomplete - re-run to resume", file=sys.stderr)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0 if sem_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
