#!/usr/bin/env python3
"""Idempotent schema migration for the subdocument spike.

Adds the storage introduced by the subdocument/Docling spike to a
database already migrated by ``migrate_semantic_schema.py``:

* ``document.metadata_dict`` (JSON, nullable) — the JSON-compatible
  remainder of ``get_metadata()`` after title and keywords are stored in
  their dedicated columns.
* ``indexing_work`` — durable pending sub-document work: the full
  collection-local ID of a discovered child (the target) plus its
  parent, published atomically with the parent's own rows and retired
  when the target is indexed or fails.

The target needs no backfill: existing rows simply carry no metadata
and there is no pending work before the spike. ``document_chunk.text``
(exact-text retention) already exists in any database created after
the model added the column; this migration does not touch it.

Ordered steps (each inspects before acting, so re-runs are no-ops):

1. ``document.metadata_dict`` (JSON, nullable).
2. ``indexing_work`` table (composite primary key
   ``(collection_id, document_id)``, NOT NULL ``parent_document_id``).
3. ``ix_indexing_work_parent`` index.
4. Deferrable ``fk_indexing_work_parent`` (the parent must have a
   Document row; CASCADE removes a parent's work entries with it).

Usage (per our standard argument processing):
    python scripts/migrate_subdoc_schema.py --config config.toml
    python scripts/migrate_subdoc_schema.py --config a.toml --config b.toml
    python scripts/migrate_subdoc_schema.py --config config.toml --dry-run

DDL steps take ``ACCESS EXCLUSIVE`` locks; run with writers stopped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_indexer.config import ConfigManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate to the subdocument schema "
                    "(metadata_dict + indexing_work; idempotent).",
    )
    parser.add_argument(
        "-c", "--config", action="append", default=[], metavar="PATH",
        help="Config TOML file (repeatable; last wins)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the work that would be done without writing.",
    )
    return parser.parse_args(argv)


def column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s "
        "AND column_name = %s", (table, column),
    )
    return cur.fetchone() is not None


def table_exists(cur, table: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s", (table,),
    )
    return cur.fetchone() is not None


def index_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", (name,))
    return cur.fetchone() is not None


def constraint_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM pg_constraint WHERE conname = %s", (name,))
    return cur.fetchone() is not None


def run(cur, sql: str, params: tuple = ()) -> None:
    cur.execute(sql, params)


def step_metadata_dict(cur, dry: bool) -> None:
    if not column_exists(cur, "document", "metadata_dict"):
        if not dry:
            run(cur, "ALTER TABLE document ADD COLUMN metadata_dict JSON")
        print("1. added document.metadata_dict" + ("" if not dry else " (dry-run)"))
    else:
        print("1. document.metadata_dict already present")


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
        print("2. created indexing_work" + ("" if not dry else " (dry-run)"))
    else:
        # A partial run may have created the table without the column;
        # verify the canonical shape.
        if not column_exists(cur, "indexing_work", "parent_document_id"):
            if not dry:
                run(cur, "ALTER TABLE indexing_work "
                        "ADD COLUMN parent_document_id VARCHAR NOT NULL")
            print("2. added indexing_work.parent_document_id"
                  + ("" if not dry else " (dry-run)"))
        else:
            print("2. indexing_work already present")


def step_indexing_work_index(cur, dry: bool) -> None:
    if not index_exists(cur, "ix_indexing_work_parent"):
        if not dry:
            run(cur, "CREATE INDEX ix_indexing_work_parent "
                    "ON indexing_work (collection_id, parent_document_id)")
        print("3. added ix_indexing_work_parent" + ("" if not dry else " (dry-run)"))
    else:
        print("3. ix_indexing_work_parent already present")


def step_indexing_work_fk(cur, dry: bool) -> None:
    if not constraint_exists(cur, "fk_indexing_work_parent"):
        if not dry:
            run(cur, """
                ALTER TABLE indexing_work ADD CONSTRAINT fk_indexing_work_parent
                FOREIGN KEY (collection_id, parent_document_id)
                REFERENCES document (collection_id, document_id)
                ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
            """)
        print("4. added deferrable fk_indexing_work_parent"
              + ("" if not dry else " (dry-run)"))
    else:
        print("4. fk_indexing_work_parent already present")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config:
        print("error: at least one --config is required", file=sys.stderr)
        return 2
    config = ConfigManager(args.config)
    server = config.get_server_config()
    # The server config may carry a SQLAlchemy URI; psycopg takes the
    # plain postgresql scheme.
    db_uri = server.db_uri.replace("+psycopg", "")

    conn = psycopg.connect(db_uri, row_factory=dict_row,
                           autocommit=False)
    cur = conn.cursor()
    dry = args.dry_run

    try:
        step_metadata_dict(cur, dry)
        step_indexing_work_table(cur, dry)
        step_indexing_work_index(cur, dry)
        step_indexing_work_fk(cur, dry)
        if not dry:
            conn.commit()
            print("migration complete")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
