from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mcp.server.fastmcp import FastMCP

from .models import Document, DocumentChunk
from .plugins.base import DocumentSource

# Initialize the MCP server
mcp = FastMCP("LanceDB Indexer")

# Initialize the Indexer and add the example plugin


def result_info(
    results: Iterable[tuple[Document, list[DocumentChunk]]],
    collection_sources: dict[str, DocumentSource] | None = None,
) -> list[dict[str, Any]]:
    formatted_results: list[dict[str, Any]] = []
    for document, relevant_chunks in results:
        source = None
        if collection_sources is not None:
            source = collection_sources.get(document.collection_id)
        formatted_results.append(
            {
                "title": document.title,
                "keywords": list(document.keywords),
                "document_id": f"{document.collection_id}:{document.document_id}",
                "summary": document.summary,
                "relevant_chunks": [
                    _relevant_chunk_info(document, chunk, source)
                    for chunk in relevant_chunks
                ],
            }
        )
    return formatted_results


def _relevant_chunk_info(
    document: Document,
    chunk: DocumentChunk,
    source: DocumentSource | None,
) -> dict[str, Any]:
    chunk_metadata = dict(chunk.metadata_dict)
    if source is not None:
        semantic_length, embedding_length = source.chunk_lengths(chunk_metadata)
    else:
        semantic_length = None
        if "e" in chunk_metadata:
            semantic_length = int(chunk_metadata["e"]) - int(chunk_metadata.get("b", 0))
        embedding_length = int(chunk_metadata["s"]) if "s" in chunk_metadata else None
        if semantic_length is None:
            semantic_length = embedding_length

    return {
        "chunk_id": (
            f"{document.collection_id}:"
            f"{document.document_id}:"
            f"{chunk.chunk_id}"
        ),
        "summary": chunk.summary.summary if chunk.summary else None,
        "semantic_length": semantic_length,
        "embedding_length": embedding_length,
    }


if __name__ == "__main__":
    mcp.run()
