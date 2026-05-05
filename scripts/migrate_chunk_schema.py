#!/usr/bin/env python
"""Destructively migrate collection chunk tables to the split summary schema."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable

import lancedb
import pandas as pd

from mcp_indexer.config import ConfigManager
from mcp_indexer.plugins.base import ChunkSummary, DocumentChunk, Document


CHUNK_ORDER_RE = re.compile(r"(?:^|[?&])c=(\d+)(?:&|$)")


@dataclass(frozen=True)
class MigrationStats:
    collection_id: str
    chunks: int
    summaries: int


def chunk_order(chunk_id: str) -> int:
    match = CHUNK_ORDER_RE.search(chunk_id)
    if match is None:
        raise ValueError(f"chunk_id does not contain a '?c=' order component: {chunk_id!r}")
    return int(match.group(1))


def strip_collection_prefix(val: str, collection_id: str) -> str:
    """Remove the 'source_type:collection_id:' prefix from an identifier."""
    marker = f":{collection_id}:"
    if marker in val:
        return val.split(marker, 1)[-1]
    return val


def migrate_chunk_dataframe(chunks: pd.DataFrame, collection_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return new chunk and summary dataframes from an old chunk dataframe."""
    required_columns = {"document_id", "chunk_id", "embedding", "metadata_str"}
    missing_columns = required_columns - set(chunks.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"chunk table is missing required column(s): {missing}")

    if "summary" not in chunks.columns:
        raise ValueError("chunk table is missing required column: summary")

    # Strip the source_type:collection_id: prefix from identifiers
    chunks["document_id"] = chunks["document_id"].apply(lambda x: strip_collection_prefix(str(x), collection_id))
    chunks["chunk_id"] = chunks["chunk_id"].apply(lambda x: strip_collection_prefix(str(x), collection_id))

    migrated = chunks.copy()
    migrated["order"] = migrated["chunk_id"].map(chunk_order)
    migrated = migrated.sort_values(
        by=["document_id", "order", "chunk_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary_span_ids: list[int | None] = []
    summary_rows: list[dict[str, object]] = []
    seen_by_document: dict[str, dict[str, int]] = {}

    for row in migrated.itertuples(index=False):
        document_id = str(getattr(row, "document_id"))
        summary = getattr(row, "summary")
        if pd.isna(summary):
            summary_span_ids.append(None)
            continue

        summary_text = str(summary)
        seen = seen_by_document.setdefault(document_id, {})
        summary_span = seen.get(summary_text)
        if summary_span is None:
            summary_span = len(seen)
            seen[summary_text] = summary_span
            summary_rows.append({
                "document_id": document_id,
                "summary_span": summary_span,
                "summary": summary_text,
            })
        summary_span_ids.append(summary_span)

    migrated["summary_span"] = summary_span_ids
    if "text" not in migrated.columns:
        migrated["text"] = None

    chunk_columns = ["document_id", "order", "chunk_id", "text", "embedding", "summary_span", "metadata_str"]
    return migrated[chunk_columns], pd.DataFrame(summary_rows, columns=["document_id", "summary_span", "summary"])


def migrate_collection(db, collection_id: str) -> MigrationStats:
    chunks = db.open_table(collection_id).to_pandas()
    new_chunks, summaries = migrate_chunk_dataframe(chunks, collection_id)

    summary_table = f"{collection_id}_summary"
    meta_table = f"{collection_id}_meta"
    
# Handle meta table
    meta_df = db.open_table(meta_table).to_pandas()
    meta_df['title_strength'] = 0
    meta_df["document_id"] = meta_df["document_id"].apply(lambda x: strip_collection_prefix(str(x), collection_id))
    db.drop_table(meta_table)
    db.create_table(meta_table, data=meta_df, schema=Document)
    
    db.drop_table(collection_id)
    db.drop_table(summary_table, ignore_missing=True)
    
    
    db.create_table(collection_id, data=new_chunks, schema=DocumentChunk)
    db.create_table(summary_table, data=summaries, schema=ChunkSummary)

    return MigrationStats(
        collection_id=collection_id,
        chunks=len(new_chunks),
        summaries=len(summaries),
    )


def migrate_configured_collections(config_path: str, collections: Iterable[str] | None = None) -> list[MigrationStats]:
    config = ConfigManager(config_path)
    server_config = config.get_server_config()
    db = lancedb.connect(os.path.expanduser(server_config.db_uri))

    collection_ids = list(collections) if collections is not None else config.list_collections()
    configured = set(config.list_collections())
    unknown = sorted(set(collection_ids) - configured)
    if unknown:
        raise ValueError(f"collection(s) are not in config: {', '.join(unknown)}")

    return [migrate_collection(db, collection_id) for collection_id in collection_ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the config TOML file")
    parser.add_argument(
        "--collection",
        action="append",
        dest="collections",
        help="Collection to migrate. May be passed more than once. Defaults to all configured collections.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive replacement of chunk and _summary tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.yes:
        raise SystemExit("Refusing destructive migration without --yes")

    stats = migrate_configured_collections(args.config, args.collections)
    for item in stats:
        print(f"{item.collection_id}: migrated {item.chunks} chunks and {item.summaries} summaries")


if __name__ == "__main__":
    main()
