#!/usr/bin/env python3
"""Benchmark LanceDB merge_insert with when_not_matched_by_source_delete."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import lancedb
from lancedb import col, lit

from mcp_indexer.plugins.base import DocumentChunk
from mcp_indexer.llm import VECTOR_DIMENSIONS


def make_document_id(doc_index: int) -> str:
    return f"doc-{doc_index:07d}"


def chunk_counts_for_batch(document_count: int, total_chunk_rows: int) -> list[int]:
    if document_count <= 0:
        raise ValueError("document_count must be positive")
    if total_chunk_rows < document_count:
        raise ValueError("total_chunk_rows must be at least document_count")

    base = total_chunk_rows // document_count
    extra = total_chunk_rows % document_count
    return [base + (1 if index < extra else 0) for index in range(document_count)]


def make_chunk_rows(
    document_ids: list[str],
    chunk_counts: list[int],
    *,
    generation: int,
) -> list[DocumentChunk]:
    rows: list[DocumentChunk] = []
    for document_id, chunk_count in zip(document_ids, chunk_counts, strict=True):
        for order in range(chunk_count):
            rows.append(DocumentChunk(
                document_id=document_id,
                order=order,
                chunk_id=f"{document_id}?c={order}",
                text=f"generation={generation} document={document_id} chunk={order}",
                embedding=[float(generation), float(order)] + [0.0] * (VECTOR_DIMENSIONS - 2),
                summary_span=None,
                metadata={
                    "generation": generation,
                    "document_id": document_id,
                    "order": order,
                    "span": chunk_count,
                },
            ))
    return rows


def document_filter(document_id: str) -> str:
    return (col("document_id") == lit(document_id)).to_sql()


def delete_filter_for_documents(document_ids: list[str]) -> str:
    return " OR ".join(f"({document_filter(document_id)})" for document_id in document_ids)


def create_table(db, table_name: str, rows: list[DocumentChunk]) -> None:
    db.create_table(table_name, data=rows, schema=DocumentChunk)


def seed_table(
    db,
    table_name: str,
    total_documents: int,
    total_chunk_rows: int,
) -> None:
    document_ids = [make_document_id(doc_index) for doc_index in range(total_documents)]
    chunk_counts = chunk_counts_for_batch(total_documents, total_chunk_rows)
    rows = make_chunk_rows(
        document_ids,
        chunk_counts,
        generation=0,
    )
    start = time.perf_counter()
    create_table(db, table_name, rows)
    elapsed = time.perf_counter() - start
    print(
        f"seed complete documents={total_documents} chunk_rows={len(rows)} "
        f"seconds={elapsed:.3f}"
    )


def run_benchmark(
    db,
    table_name: str,
    seed_documents: int,
    batch_documents: int,
    batch_chunk_rows: int,
    rounds: int,
) -> None:
    table = db.open_table(table_name)
    next_document_index = seed_documents

    for generation in range(1, rounds + 1):
        document_ids = [
            make_document_id(next_document_index + offset)
            for offset in range(batch_documents)
        ]
        next_document_index += batch_documents
        chunk_counts = chunk_counts_for_batch(batch_documents, batch_chunk_rows)
        rows = make_chunk_rows(
            document_ids,
            chunk_counts,
            generation=generation,
        )
        delete_filter = delete_filter_for_documents(document_ids)

        started = time.perf_counter()
        table.merge_insert(["document_id", "chunk_id"]) \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .when_not_matched_by_source_delete(delete_filter) \
            .execute(rows)
        elapsed = time.perf_counter() - started

        print(
            f"merge_insert batch_documents={batch_documents} chunk_rows={len(rows)} "
            f"generation={generation} seconds={elapsed:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone benchmark for the LanceDB merge_insert(...)."
            "when_not_matched_by_source_delete(...) shape used by chunk upserts."
        )
    )
    parser.add_argument(
        "--db-uri",
        type=Path,
        default=Path("/tmp/lancedb-merge-insert-benchmark"),
        help="Directory to use for the temporary benchmark database.",
    )
    parser.add_argument(
        "--table-name",
        default="benchmark_chunks",
        help="Table name to create and benchmark.",
    )
    parser.add_argument(
        "--seed-documents",
        type=int,
        default=190000,
        help="Number of documents to seed into the benchmark table before timed runs.",
    )
    parser.add_argument(
        "--seed-chunk-rows",
        type=int,
        default=700000,
        help="Total number of chunk rows to seed before timed runs.",
    )
    parser.add_argument(
        "--batch-documents",
        type=int,
        default=300,
        help="Number of fresh documents to insert per merge_insert run.",
    )
    parser.add_argument(
        "--batch-chunk-rows",
        type=int,
        default=3500,
        help="Total number of fresh chunk rows to insert per merge_insert run.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="How many timed merge_insert runs to execute.",
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
    if args.seed_chunk_rows < args.seed_documents:
        raise SystemExit("--seed-chunk-rows must be at least --seed-documents")
    if args.batch_documents <= 0:
        raise SystemExit("--batch-documents must be positive")
    if args.batch_chunk_rows < args.batch_documents:
        raise SystemExit("--batch-chunk-rows must be at least --batch-documents")
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    db_uri = args.db_uri.expanduser().resolve()
    if db_uri.exists() and not args.keep_db:
        shutil.rmtree(db_uri)
    db_uri.mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(str(db_uri))
    if args.table_name in db.list_tables():
        db.drop_table(args.table_name)

    seed_table(
        db,
        args.table_name,
        args.seed_documents,
        args.seed_chunk_rows,
    )
    run_benchmark(
        db,
        args.table_name,
        args.seed_documents,
        args.batch_documents,
        args.batch_chunk_rows,
        args.rounds,
    )


if __name__ == "__main__":
    main()
