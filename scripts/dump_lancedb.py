#!/usr/bin/env python
"""Dump LanceDB tables from a DocumentSource to JSON files for migration to SQLAlchemy+PostgreSQL.

This script extracts all data from the three table types used by each collection:
- {collection}_chunk: DocumentChunk records
- {collection}_meta: Document records  
- {collection}_summary: ChunkSummary records

The output format is designed for easy loading into PostgreSQL with pgvector extension.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import lancedb
import pandas as pd


def normalize_embedding(embedding: Any) -> List[float] | None:
    """Convert an embedding (list, numpy array, etc.) to a plain Python list."""
    if embedding is None:
        return None
    if hasattr(embedding, 'tolist'):
        return embedding.tolist()
    if hasattr(embedding, '__iter__'):
        return list(embedding)
    # Handle string representations
    try:
        return [float(x) for x in str(embedding).split(',')]
    except (ValueError, AttributeError):
        return [0.0] * 768  # Fallback


def normalize_metadata(metadata: str | dict | None | Any) -> dict[str, Any]:
    """Parse metadata to dict, handling edge cases."""
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    # Handle numpy arrays (empty arrays should become empty dict)
    if hasattr(metadata, '__len__') and hasattr(metadata, 'item'):
        # It's likely a numpy array or similar
        try:
            if len(metadata) == 0:
                return {}
        except Exception:
            return {}
    try:
        result = json.loads(metadata)
        if not isinstance(result, dict):
            return {}
        return result
    except (json.JSONDecodeError, TypeError):
        return {}


def dump_table_to_dict(table: Any) -> List[Dict[str, Any]]:
    """Convert a LanceDB table to a list of dicts with normalized types."""
    # Check row count directly to avoid Arrow array issues with len()/empty
    row_count = table.count_rows()
    if row_count == 0:
        return []
    
    try:
        df = table.to_pandas()
    except Exception as e:
        print(f"  Warning: Could not read table: {e}", file=sys.stderr)
        return []
    
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            
            try:
                # Normalize embeddings - handle both array and regular iterables
                if col == 'embedding':
                    if val is None or (hasattr(val, '__len__') and len(val) == 0):
                        record[col] = None
                    elif hasattr(val, 'tolist'):
                        record[col] = val.tolist()
                    elif hasattr(val, '__iter__') and not isinstance(val, str):
                        record[col] = list(val)
                    else:
                        record[col] = [float(x) for x in str(val).split(',')]
                # Normalize metadata
                elif col == 'metadata_str':
                    record[col] = normalize_metadata(val)
                # Handle datetime
                elif col == 'last_modified' or 'datetime' in str(type(val)):
                    if pd.notna(val):
                        record[col] = val.isoformat() if hasattr(val, 'isoformat') else str(val)
                    else:
                        record[col] = None
                # Handle other types - skip array type check for numpy arrays
                elif hasattr(val, 'size') and val.size == 0:
                    # Empty numpy array
                    record[col] = None
                elif pd.isna(val):
                    record[col] = None
                else:
                    # Convert numpy types to Python native types
                    if hasattr(val, 'item'):
                        record[col] = val.item()
                    else:
                        record[col] = val
            except Exception as e:
                print(f"  Warning: Error processing column '{col}' (type {type(val)}): {e}", file=sys.stderr)
                record[col] = None
        
        records.append(record)
    
    return records


def dump_collection(db: Any, collection_id: str, output_dir: Path) -> dict:
    """Dump all tables for a single collection to files. Returns stats."""
    chunks_table_name = collection_id
    meta_table_name = f"{collection_id}_meta"
    summary_table_name = f"{collection_id}_summary"
    
    stats = {
        'collection_id': collection_id,
        'chunks': 0,
        'meta': 0,
        'summary': 0,
    }
    
    # Dump chunks table
    chunks_path = output_dir / f"{collection_id}_chunks.json"
    try:
        chunks_table = db.open_table(chunks_table_name)
        chunks_data = dump_table_to_dict(chunks_table)
        stats['chunks'] = len(chunks_data)
        with open(chunks_path, 'w') as f:
            json.dump(chunks_data, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not dump chunks table: {e}", file=sys.stderr)
    
    # Dump meta (document) table
    meta_path = output_dir / f"{collection_id}_meta.json"
    try:
        meta_table = db.open_table(meta_table_name)
        meta_data = dump_table_to_dict(meta_table)
        stats['meta'] = len(meta_data)
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not dump meta table: {e}", file=sys.stderr)
    
    # Dump summary table
    summary_path = output_dir / f"{collection_id}_summary.json"
    try:
        summary_table = db.open_table(summary_table_name)
        summary_data = dump_table_to_dict(summary_table)
        stats['summary'] = len(summary_data)
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not dump summary table: {e}", file=sys.stderr)
    
    return stats


def get_all_collections(db: Any) -> List[str]:
    """Get all collection table names from the database."""
    tables = db.table_names()
    # Filter to get only collection tables (not _meta or _summary tables)
    collection_ids = set()
    for table_name in tables:
        if table_name.endswith('_meta') or table_name.endswith('_summary'):
            continue
        # Check if the corresponding meta table exists
        if f"{table_name}_meta" in tables:
            collection_ids.add(table_name)
    return sorted(collection_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        help="Path to the config TOML file. If provided, all collections in the config will be dumped."
    )
    parser.add_argument(
        "--db-uri",
        help="LanceDB URI (e.g., ~/.lancedb or s3://bucket/path). Required if no config."
    )
    parser.add_argument(
        "--collection",
        action="append",
        dest="collections",
        help="Specific collection ID to dump. Can be specified multiple times. Requires --db-uri if no config."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for dumped JSON files"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.config and not (args.db_uri and args.collections):
        parser.error("Either --config or both --db-uri and --collection must be specified")
    
    return args


def load_collections_from_config(config_path: str) -> List[str]:
    """Load collection IDs from a config file."""
    import tomllib
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    return list(config.get('collections', {}).keys())


def main() -> None:
    args = parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Connect to database
    db_uri = os.path.expanduser(args.db_uri) if args.db_uri else None
    db = lancedb.connect(db_uri)
    
    # Get collection IDs
    if args.config:
        collection_ids = load_collections_from_config(args.config)
    else:
        collection_ids = args.collections
    
    # Dump each collection
    stats = []
    for collection_id in collection_ids:
        print(f"Dumping collection: {collection_id}")
        table_stats = dump_collection(db, collection_id, output_path)
        stats.append(table_stats)
        
        # Print individual stats
        total = table_stats['chunks'] + table_stats['meta'] + table_stats['summary']
        print(f"  Chunks: {table_stats['chunks']}, Meta: {table_stats['meta']}, Summary: {table_stats['summary']}")
    
    # Print summary
    print(f"\nDumped {len(stats)} collection(s) to {args.output}")
    if stats:
        total_chunks = sum(s['chunks'] for s in stats)
        total_meta = sum(s['meta'] for s in stats)
        total_summary = sum(s['summary'] for s in stats)
        print(f"Total: {total_chunks} chunks, {total_meta} meta, {total_summary} summaries")


if __name__ == "__main__":
    main()