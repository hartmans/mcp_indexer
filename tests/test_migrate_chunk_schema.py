import lancedb
import pandas as pd
import pytest

from mcp_indexer.llm import VECTOR_DIMENSIONS
from scripts.migrate_chunk_schema import (
    chunk_order,
    migrate_chunk_dataframe,
    migrate_collection,
    migrate_configured_collections,
)


def embedding(value: float = 0.0) -> list[float]:
    return [value] * VECTOR_DIMENSIONS


def test_chunk_order_uses_c_query_component():
    assert chunk_order("doc?c=12") == 12
    assert chunk_order("doc?x=1&c=7&z=2") == 7

    with pytest.raises(ValueError):
        chunk_order("doc")


def test_migrate_chunk_dataframe_orders_and_coalesces_summaries_by_document():
    old_chunks = pd.DataFrame([
        {
            "document_id": "text:test_col:doc-a",
            "chunk_id": "text:test_col:doc-a?c=2",
            "summary": "middle",
            "embedding": embedding(2.0),
            "metadata_str": '{"o": 10, "s": 5}',
        },
        {
            "document_id": "text:test_col:doc-a",
            "chunk_id": "text:test_col:doc-a?c=0",
            "summary": "intro",
            "embedding": embedding(0.0),
            "metadata_str": '{"o": 0, "s": 5}',
        },
        {
            "document_id": "text:test_col:doc-a",
            "chunk_id": "text:test_col:doc-a?c=1",
            "summary": "intro",
            "embedding": embedding(1.0),
            "metadata_str": '{"o": 5, "s": 5}',
        },
        {
            "document_id": "text:test_col:doc-b",
            "chunk_id": "text:test_col:doc-b?c=0",
            "summary": "intro",
            "embedding": embedding(3.0),
            "metadata_str": '{"o": 0, "s": 5}',
        },
    ])

    chunks, summaries = migrate_chunk_dataframe(old_chunks, "test_col")

    assert chunks["document_id"].tolist() == ["doc-a", "doc-a", "doc-a", "doc-b"]
    assert chunks["chunk_id"].tolist() == ["doc-a?c=0", "doc-a?c=1", "doc-a?c=2", "doc-b?c=0"]
    assert chunks["order"].tolist() == [0, 1, 2, 0]
    assert chunks["summary_span"].tolist() == [0, 0, 1, 0]
    assert summaries.to_dict("records") == [
        {"document_id": "doc-a", "summary_span": 0, "summary": "intro"},
        {"document_id": "doc-a", "summary_span": 1, "summary": "middle"},
        {"document_id": "doc-b", "summary_span": 0, "summary": "intro"},
    ]


def test_migrate_collection_replaces_chunk_table_and_writes_summary_table(tmp_path):
    db = lancedb.connect(str(tmp_path / "db"))
    db.create_table("notes", data=pd.DataFrame([
        {
            "document_id": "text:notes:doc",
            "chunk_id": "text:notes:doc?c=1",
            "summary": "later",
            "embedding": embedding(1.0),
            "metadata_str": '{"o": 5, "s": 5}',
        },
        {
            "document_id": "text:notes:doc",
            "chunk_id": "text:notes:doc?c=0",
            "summary": "first",
            "embedding": embedding(0.0),
            "metadata_str": '{"o": 0, "s": 5}',
        },
    ]))
    db.create_table("notes_meta", data=pd.DataFrame([{
        "document_id": "text:notes:doc",
        "title": "doc",
        "title_strength": 9,
        "embedding": embedding(0.0),
        "summary": "first",
        "last_modified": "2024-01-01T00:00:00Z",
    }]))

    stats = migrate_collection(db, "notes")

    assert stats.chunks == 2
    assert stats.summaries == 2
    assert sorted(db.table_names()) == ["notes", "notes_meta", "notes_summary"]

    chunk_rows = db.open_table("notes").to_pandas().sort_values("order").to_dict("records")
    meta_rows = db.open_table("notes_meta").to_pandas().to_dict("records")
    summary_rows = db.open_table("notes_summary").to_pandas().sort_values("summary_span").to_dict("records")

    assert meta_rows[0]["document_id"] == "doc"
    assert meta_rows[0]["title"] == "doc"
    assert meta_rows[0]["title_strength"] == 9
    assert meta_rows[0]["summary"] == "first"
    assert len(meta_rows[0]["keywords"]) == 0
    assert str(meta_rows[0]["last_modified"]).startswith("2024-01-01")
    assert [row["order"] for row in chunk_rows] == [0, 1]
    assert [row["document_id"] for row in chunk_rows] == ["doc", "doc"]
    assert [row["chunk_id"] for row in chunk_rows] == ["doc?c=0", "doc?c=1"]
    assert [row["summary_span"] for row in chunk_rows] == [0, 1]
    assert [row["summary"] for row in summary_rows] == ["first", "later"]


def test_migrate_configured_collections_uses_configured_db_and_collections(tmp_path):
    db_path = tmp_path / "db"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[server]
db_uri = "{db_path}"

[collections.notes]
tool_prefix = "notes"
source_config = {{ type = "text", directory = "." }}
""",
        encoding="utf-8",
    )

    db = lancedb.connect(str(db_path))
    db.create_table("notes", data=pd.DataFrame([{
        "document_id": "text:notes:doc",
        "chunk_id": "text:notes:doc?c=0",
        "summary": "first",
        "embedding": embedding(0.0),
        "metadata_str": "{}",
    }]))

    stats = migrate_configured_collections(str(config_path))

    assert stats[0].collection_id == "notes"
    assert db.open_table("notes_summary").to_pandas().to_dict("records") == [{
        "document_id": "doc",
        "summary_span": 0,
        "summary": "first",
    }]
