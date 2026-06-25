#!/usr/bin/env python3
"""Dump LanceDB data to JSON files for migration to PostgreSQL.

This script reads from a LanceDB database and outputs JSON files that can be
loaded into PostgreSQL using the new SQLAlchemy models.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel


def parse_args():
    parser = argparse.ArgumentParser(description="Dump LanceDB data to JSON")
    parser.add_argument(
        "--db-uri",
        required=True,
        help="Path to LanceDB database directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for JSON files",
    )
    return parser.parse_args()


def serialize_value(value):
    """Serialize a value to JSON-compatible format."""
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return str(value)
    return value


def dump_table(db, table_name, output_path):
    """Dump a LanceDB table to JSON lines format."""
    print(f"Dumping {table_name}...")
    table = db.open_table(table_name)
    
    # Convert to pandas and then to records
    df = table.to_pandas()
    
    # Convert to list of dicts with JSON-serializable values
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            value = row[col]
            record[col] = serialize_value(value)
        records.append(record)
    
    # Write as JSON lines
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    print(f"  Wrote {len(records)} records to {output_path}")
    return len(records)


def main():
    args = parse_args()
    
    db_path = os.path.expanduser(args.db_uri)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to LanceDB at {db_path}...")
    db = lancedb.connect(db_path)
    
    total_docs = 0
    total_chunks = 0
    total_summaries = 0
    
    for table_name in db.table_names():
        if table_name.endswith("_meta"):
            # Document table
            output_path = output_dir / f"documents.jsonl"
            count = dump_table(db, table_name, output_path)
            total_docs += count
        elif table_name.endswith("_summary"):
            # ChunkSummary table
            output_path = output_dir / f"summaries.jsonl"
            count = dump_table(db, table_name, output_path)
            total_summaries += count
        else:
            # DocumentChunk table
            output_path = output_dir / f"chunks.jsonl"
            count = dump_table(db, table_name, output_path)
            total_chunks += count
    
    print(f"\nSummary:")
    print(f"  Documents: {total_docs}")
    print(f"  Chunks: {total_chunks}")
    print(f"  Summaries: {total_summaries}")
    print(f"\nOutput files in {output_dir}/")


if __name__ == "__main__":
    main()