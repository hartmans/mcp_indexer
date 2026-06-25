#!/usr/bin/env python3
"""Benchmark LanceDB merge_insert for the document meta table shape."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import lancedb

from mcp_indexer.llm import VECTOR_DIMENSIONS
from mcp_indexer.plugins.base import Document


def make_document_id(doc_index: int) -> str:
    return f"doc-{doc_index:07d}"


def make_documents(
    start_document_index: int,
    document_count: int,
    *,
    generation: int,
    zero_embedding: bool,
) -> list[Document]:
    documents: list[Document] = []
    base_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=generation)
    zero_vector = [0.0] * VECTOR_DIMENSIONS

    for offset in range(document_count):
        document_id = make_document_id(start_document_index + offset)
        if zero_embedding:
            embedding = zero_vector
        else:
            embedding = [float(generation), float(offset)] + [0.0] * (VECTOR_DIMENSIONS - 2)

        documents.append(Document(
            document_id=document_id,
            title=f"title {document_id}",
            title_strength=0,
            embedding=embedding,
            keywords=[f"g{generation}", f"d{offset % 10}"],
            summary=f"summary generation={generation} document={document_id}",
            last_modified=base_time,
        ))

    return documents


def create_table(db, table_name: str, rows: list[Document]) -> None:
    db.create_table(table_name, data=rows, schema=Document)


def seed_table(
    db,
    table_name: str,
    seed_documents: int,
    *,
    zero_embedding: bool,
) -> None:
    rows = make_documents(
        0,
        seed_documents,
        generation=0,
        zero_embedding=zero_embedding,
    )
    started = time.perf_counter()
    create_table(db, table_name, rows)
    elapsed = time.perf_counter() - started
    print(f"seed complete documents={len(rows)} seconds={elapsed:.3f}")


def run_benchmark(
    db,
    table_name: str,
    seed_documents: int,
    batch_documents: int,
    rounds: int,
    *,
    zero_embedding: bool,
) -> None:
    table = db.open_table(table_name)
    next_document_index = seed_documents

    for generation in range(1, rounds + 1):
        rows = make_documents(
            next_document_index,
            batch_documents,
            generation=generation,
            zero_embedding=zero_embedding,
        )
        next_document_index += batch_documents

        started = time.perf_counter()
        table.merge_insert(["document_id"]) \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute(rows)
        elapsed = time.perf_counter() - started

        print(
            f"merge_insert batch_documents={batch_documents} "
            f"generation={generation} seconds={elapsed:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone benchmark for the LanceDB document meta-table merge_insert shape."
    )
    parser.add_argument(
        "--db-uri",
        type=Path,
        default=Path("/tmp/lancedb-meta-merge-insert-benchmark"),
        help="Directory to use for the temporary benchmark database.",
    )
    parser.add_argument(
        "--table-name",
        default="benchmark_meta",
        help="Table name to create and benchmark.",
    )
    parser.add_argument(
        "--seed-documents",
        type=int,
        default=190000,
        help="Number of document rows to seed before timed runs.",
    )
    parser.add_argument(
        "--batch-documents",
        type=int,
        default=300,
        help="Number of fresh documents to insert per merge_insert run.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="How many timed merge_insert runs to execute.",
    )
    parser.add_argument(
        "--nonzero-embedding",
        action="store_true",
        help="Use distinct synthetic embeddings instead of the all-zero production-like default.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep the benchmark database directory instead of recreating it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed_documents <= 0:
        raise SystemExit("--seed-documents must be positive")
    if args.batch_documents <= 0:
        raise SystemExit("--batch-documents must be positive")
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")

    db_uri = args.db_uri.expanduser().resolve()
    if db_uri.exists() and not args.keep_db:
        shutil.rmtree(db_uri)
    db_uri.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(db_uri))
    if args.table_name in db.list_tables():
        db.drop_table(args.table_name)

    zero_embedding = not args.nonzero_embedding
    seed_table(
        db,
        args.table_name,
        args.seed_documents,
        zero_embedding=zero_embedding,
    )
    run_benchmark(
        db,
        args.table_name,
        args.seed_documents,
        args.batch_documents,
        args.rounds,
        zero_embedding=zero_embedding,
    )


if __name__ == "__main__":
    main()
