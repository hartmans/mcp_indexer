from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mcp.server.fastmcp import FastMCP

from .models import Document, DocumentChunk

# Initialize the MCP server
mcp = FastMCP("LanceDB Indexer")

# Initialize the Indexer and add the example plugin


def result_info(
    results: Iterable[tuple[Document, list[DocumentChunk]]],
) -> list[dict[str, Any]]:
    formatted_results: list[dict[str, Any]] = []
    for document, relevant_chunks in results:
        formatted_results.append(
            {
                "title": document.title,
                "keywords": list(document.keywords),
                "document_id": f"{document.collection_id}:{document.document_id}",
                "summary": document.summary,
                "relevant_chunks": [
                    {
                        "chunk_id": (
                            f"{document.collection_id}:"
                            f"{document.document_id}:"
                            f"{chunk.chunk_id}"
                        ),
                        "summary": chunk.summary.summary if chunk.summary else None,
                    }
                    for chunk in relevant_chunks
                ],
            }
        )
    return formatted_results


if __name__ == "__main__":
    mcp.run()
