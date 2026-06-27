#!/usr/bin/env python3
"""Load data from dump files into PostgreSQL.

This script loads JSONL files exported by dump_lancedb.py into the PostgreSQL
database using SQLAlchemy models. It handles the correct order for foreign keys.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from mcp_indexer.context import Context
from mcp_indexer.models import ChunkSummary, Document, DocumentChunk
from mcp_indexer.config import ConfigManager, ServerConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load dump files into PostgreSQL"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.toml file",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing JSONL dump files",
    )
    parser.add_argument(
        "--collection-id",
        required=True,
        help="Collection ID to load",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be loaded without making changes",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    """Load records from a JSONL file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def convert_record(record: dict, collection_id: str) -> dict:
    """Convert a dump record to the format expected by SQLAlchemy models.
    
    Handles:
    - collection_id override
    - metadata -> metadata_str conversion and JSON serialization
    - datetime string parsing
    """
    converted = dict(record)
    converted["collection_id"] = collection_id
    
    # Convert metadata dict to metadata_str JSON string
    if "metadata" in converted:
        converted["metadata_str"] = json.dumps(converted.pop("metadata"), sort_keys=True)
    
    # Parse datetime strings back to datetime objects
    if "last_modified" in converted and isinstance(converted["last_modified"], str):
        converted["last_modified"] = datetime.fromisoformat(converted["last_modified"])
    
    return converted


def load_documents(records: list, collection_id: str, session, dry_run: bool = False) -> int:
    """Load document records."""
    count = 0
    for record in records:
        converted = convert_record(record, collection_id)
        
        if dry_run:
            print(f"  Document: {converted.get('document_id')}")
            continue
        
        doc = Document(**converted)
        session.add(doc)
        count += 1
    
    return count


def load_chunks(records: list, collection_id: str, session, dry_run: bool = False) -> int:
    """Load chunk records."""
    count = 0
    for record in records:
        converted = convert_record(record, collection_id)
        
        if dry_run:
            print(f"  Chunk: {converted.get('chunk_id')}")
            continue
        
        chunk = DocumentChunk(**converted)
        session.add(chunk)
        count += 1
    
    return count


def load_summaries(records: list, collection_id: str, session, dry_run: bool = False) -> int:
    """Load summary records."""
    count = 0
    for record in records:
        converted = convert_record(record, collection_id)
        
        if dry_run:
            print(f"  Summary: {converted.get('document_id')}::{converted.get('summary_span')}")
            continue
        
        summary = ChunkSummary(**converted)
        session.add(summary)
        count += 1
    
    return count


def validate_database(engine):
    """Validate that the database is properly set up for vector operations."""
    from sqlalchemy import text
    
    with engine.connect() as conn:
        # Check pgvector extension is enabled
        result = conn.execute(text(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        )).fetchone()
        if not result:
            raise RuntimeError(
                "pgvector extension not enabled. Run: CREATE EXTENSION vector;"
            )
    
    print("Database validation passed: pgvector extension is enabled")


def main():
    args = parse_args()
    
    # Load config
    config_manager = ConfigManager(args.config)
    server_config = config_manager.get_server_config()
    
    # Get the collection config for embedding configuration
    collection_config = config_manager.get_collection_config(args.collection_id)
    
    # Validate database BEFORE loading files
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from mcp_indexer.models import Base
    
    engine = create_engine(server_config.db_uri)
    validate_database(engine)
    
    # Create tables
    print("Creating tables...")
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    
    input_dir = Path(args.input_dir)
    
    # Verify input files exist
    files = {
        "documents": input_dir / "documents.jsonl",
        "chunks": input_dir / "chunks.jsonl",
        "summaries": input_dir / "summaries.jsonl",
    }
    
    for name, path in files.items():
        if not path.exists():
            print(f"Warning: {path} not found")
            files[name] = None
    
    # Load records from files
    documents = files["documents"] and load_jsonl(files["documents"]) or []
    chunks = files["chunks"] and load_jsonl(files["chunks"]) or []
    summaries = files["summaries"] and load_jsonl(files["summaries"]) or []
    
    print(f"Loaded {len(documents)} documents, {len(chunks)} chunks, {len(summaries)} summaries")
    
    if args.dry_run:
        print("\nDry run - no changes will be made")
        print(f"Collection: {args.collection_id}")
        print(f"Server DB URI: {server_config.db_uri}")
        
        print("\nDocuments to load:")
        for doc in documents[:5]:
            print(f"  {doc.get('document_id')}")
        if len(documents) > 5:
            print(f"  ... and {len(documents) - 5} more")
        
        print("\nChunks to load:")
        for chunk in chunks[:5]:
            print(f"  {chunk.get('chunk_id')}")
        if len(chunks) > 5:
            print(f"  ... and {len(chunks) - 5} more")
        
        print("\nSummaries to load:")
        for summary in summaries[:5]:
            print(f"  {summary.get('document_id')}::{summary.get('summary_span')}")
        if len(summaries) > 5:
            print(f"  ... and {len(summaries) - 5} more")
        
        print(f"\nTotal: {len(documents)} docs, {len(chunks)} chunks, {len(summaries)} summaries")
        return 0
    
    # Load data in correct order (documents first, then chunks, then summaries)
    total_docs = 0
    total_chunks = 0
    total_summaries = 0
    
    with Session() as session:
        if documents:
            print(f"\nLoading {len(documents)} documents...")
            total_docs = load_documents(documents, args.collection_id, session)
            session.commit()
        
        if chunks:
            print(f"Loading {len(chunks)} chunks...")
            total_chunks = load_chunks(chunks, args.collection_id, session)
            session.commit()
        
        if summaries:
            print(f"Loading {len(summaries)} summaries...")
            total_summaries = load_summaries(summaries, args.collection_id, session)
            session.commit()
    
    print(f"\nLoaded successfully:")
    print(f"  Documents: {total_docs}")
    print(f"  Chunks: {total_chunks}")
    print(f"  Summaries: {total_summaries}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
