#!/usr/bin/env python3
"""Dump LanceDB tables to JSONL files for migration to SQLAlchemy+PostgreSQL.

This script extracts all data from the three table types used by each collection:
- {collection_id}: DocumentChunk records
- {collection_id}_meta: Document records  
- {collection_id}_summary: ChunkSummary records

Output format is JSONL with the following files:
- documents.jsonl - Document records
- chunks.jsonl - DocumentChunk records
- summaries.jsonl - ChunkSummary records

The metadata field is output as a dict (not JSON string) so the loader can
convert it to metadata_str.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import lancedb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump LanceDB tables to JSONL format"
    )
    parser.add_argument(
        "--db-uri",
        required=True,
        help="LanceDB URI (e.g., ~/.lancedb or s3://bucket/path)"
    )
    parser.add_argument(
        "--collection-id",
        required=True,
        help="Collection ID to dump"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for JSONL files"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    db_uri = os.path.expanduser(args.db_uri)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Connecting to LanceDB at {db_uri}...")
    db = lancedb.connect(db_uri)
    
    collection_id = args.collection_id
    chunks_table_name = collection_id
    meta_table_name = f"{collection_id}_meta"
    summary_table_name = f"{collection_id}_summary"
    
    # Check if tables exist
    tables = db.table_names()
    
    def dump_table(table_name, output_path):
        if table_name not in tables:
            print(f"  Table {table_name} not found, skipping")
            return 0
        
        print(f"  Dumping {table_name}...")
        table = db.open_table(table_name)
        count = table.count_rows()
        
        if count == 0:
            print(f"    Empty table")
            return 0
        
        # Use pandas to_json with orient='records' and lines=True for JSONL
        df = table.to_pandas()
        # Convert to JSONL - pandas to_json handles this efficiently
        df.to_json(output_path, orient='records', lines=True, date_format='iso')
        
        print(f"    Wrote {count} records to {output_path}")
        return count
    
    total_docs = 0
    total_chunks = 0
    total_summaries = 0
    
    # Dump meta (document) table first
    if meta_table_name in tables:
        total_docs = dump_table(meta_table_name, output_dir / "documents.jsonl")
    
    # Dump chunks table
    if chunks_table_name in tables:
        total_chunks = dump_table(chunks_table_name, output_dir / "chunks.jsonl")
    
    # Dump summary table
    if summary_table_name in tables:
        total_summaries = dump_table(summary_table_name, output_dir / "summaries.jsonl")
    
    print(f"\nDumped collection '{collection_id}':")
    print(f"  Documents: {total_docs}")
    print(f"  Chunks: {total_chunks}")
    print(f"  Summaries: {total_summaries}")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()