from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Iterable
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .context import Context
from .indexer import Indexer
from .search import search
from .models import Document, DocumentChunk
from .plugins.base import DocumentSource

logger = logging.getLogger(__name__)

# If a document's total semantic length is below this threshold we return the full text.
FULL_DOCUMENT_LENGTH = 10_000


def _create_server(*, host: str = "localhost", port: int | None = None) -> FastMCP:
    server_kwargs: dict[str, Any] = {}
    if port is not None:
        server_kwargs["host"] = host
        server_kwargs["port"] = port
    return FastMCP("LanceDB Indexer", **server_kwargs)


def result_info(
    results: Iterable[tuple[Document, list[DocumentChunk]]],
    collection_sources: dict[str, DocumentSource],
) -> list[dict[str, Any]]:
    formatted_results: list[dict[str, Any]] = []
    for document, relevant_chunks in results:
        source = collection_sources[document.collection_id]
        formatted_results.append(
            {
                "title": document.title,
                "keywords": list(document.keywords),
                "document_id": _document_ref(document.collection_id, document.document_id),
                "summary": document.summary,
                "relevant_chunks": [
                    _relevant_chunk_info(document, chunk, source)
                    for chunk in relevant_chunks
                ],
            }
        )
    return formatted_results


def _document_ref(collection_id: str, ref_id: str) -> str:
    return f"{collection_id}:{ref_id}"


def _tool_name(collection_id: str, suffix: str) -> str:
    return f"{collection_id}_{suffix}"


def _parse_prefixed_document_id(prefixed_document_id: str) -> tuple[str, str]:
    collection_id, document_id = prefixed_document_id.split(":", 1)
    return collection_id, document_id


def _parse_prefixed_chunk_id(prefixed_chunk_id: str) -> tuple[str, str]:
    collection_id, chunk_id = prefixed_chunk_id.split(":", 1)
    return collection_id, chunk_id


def _get_collection_source(context: Context, collection_id: str) -> DocumentSource:
    try:
        return context.collections[collection_id]
    except KeyError as exc:
        raise KeyError(f"Unknown collection_id: {collection_id}") from exc


async def _load_document_record(
    context: Context,
    *,
    collection_id: str,
    document_id: str,
) -> Document | None:
    async with context.get_session() as session:
        result = await session.execute(
            select(Document)
            .where(
                Document.collection_id == collection_id,
                Document.document_id == document_id,
            )
            .options(
                selectinload(Document.chunks).selectinload(DocumentChunk.summary)
            )
        )
        return result.scalars().unique().one_or_none()


async def _load_chunk_record(
    context: Context,
    *,
    collection_id: str,
    document_id: str,
    chunk_id: str,
) -> DocumentChunk | None:
    async with context.get_session() as session:
        result = await session.execute(
            select(DocumentChunk)
            .where(
                DocumentChunk.collection_id == collection_id,
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_id == chunk_id,
            )
            .options(selectinload(DocumentChunk.summary))
        )
        return result.scalars().one_or_none()


def _relevant_chunk_info(
    document: Document,
    chunk: DocumentChunk,
    source: DocumentSource,
) -> dict[str, Any]:
    chunk_metadata = dict(chunk.metadata_dict)
    semantic_length, embedding_length = source.chunk_lengths(chunk_metadata)

    return {
        "chunk_id": _document_ref(
            document.collection_id,
            chunk.chunk_id,
        ),
        "summary": chunk.summary.summary if chunk.summary else None,
        "semantic_length": semantic_length,
        "embedding_length": embedding_length,
    }


async def core_search(
    *,
    context: Context,
    indexer: Indexer,
    collection_id: str,
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    _get_collection_source(context, collection_id)
    results = await search(
        indexer,
        collection_id,
        query,
        limit=limit,
    )
    return result_info(results, context.collections)


async def core_fetch_document(
    *,
    context: Context,
    indexer: Indexer,
    prefixed_document_id: str,
) -> dict[str, Any]:
    collection_id, document_id = _parse_prefixed_document_id(prefixed_document_id)
    source = _get_collection_source(context, collection_id)

    # Ensure the document's summaries are up‑to‑date before we hydrate it.
    await indexer.summarize_results(collection_id, [document_id])

    document = await _load_document_record(
        context,
        collection_id=collection_id,
        document_id=document_id,
    )
    if document is None:
        raise LookupError(
            f"Document not found in indexed collection {collection_id!r}: {document_id!r}"
        )

    # Build the base response with fields we always include.
    result: dict[str, Any] = {
        "title": document.title,
        "keywords": list(document.keywords),
        "document_id": _document_ref(document.collection_id, document.document_id),
        "summary": document.summary,
    }

    # Compute total semantic length across all chunks.
    total_semantic_length = 0
    for chunk in document.chunks:
        meta = dict(chunk.metadata_dict)
        semantic_len, _ = source.chunk_lengths(meta)
        if semantic_len is not None:
            total_semantic_length += semantic_len

    # If the document fits within FULL_DOCUMENT_LENGTH, return the full concatenated text.
    if total_semantic_length < FULL_DOCUMENT_LENGTH:
        full_text_parts: list[str] = []
        for chunk in document.chunks:
            meta = dict(chunk.metadata_dict)
            part = await source.fetch_chunk(
                document_id=document.document_id,
                chunk_metadata=meta,
                scope="semantic",
            )
            full_text_parts.append(part)
        result["full_text"] = "".join(full_text_parts)
    else:
        # Otherwise include per‑chunk info as before.
        result["chunks"] = [
            _relevant_chunk_info(document, chunk, source) for chunk in document.chunks
        ]

    return result


async def core_fetch_chunk(
    *,
    context: Context,
    prefixed_chunk_id: str,
    scope: Literal["semantic", "embedding"] = "semantic",
) -> dict[str, Any]:
    collection_id, chunk_id = _parse_prefixed_chunk_id(prefixed_chunk_id)
    source = _get_collection_source(context, collection_id)
    document_id = source.document_id(chunk_id)

    document = await _load_document_record(
        context,
        collection_id=collection_id,
        document_id=document_id,
    )
    if document is None:
        raise LookupError(
            f"Document not found for chunk {prefixed_chunk_id!r} in collection {collection_id!r}"
        )

    chunk = await _load_chunk_record(
        context,
        collection_id=collection_id,
        document_id=document_id,
        chunk_id=chunk_id,
    )
    if chunk is None:
        raise LookupError(
            f"Chunk not found in indexed collection {collection_id!r}: {chunk_id!r}"
        )

    chunk_metadata = dict(chunk.metadata_dict)
    text = await source.fetch_chunk(document_id=document_id, chunk_metadata=chunk_metadata, scope=scope)
    semantic_length, embedding_length = source.chunk_lengths(chunk_metadata)

    return {
        "document_id": _document_ref(collection_id, document_id),
        "chunk_id": _document_ref(collection_id, chunk_id),
        "scope": scope,
        "text": text,
        "summary": chunk.summary.summary if chunk.summary else None,
        "semantic_length": semantic_length,
        "embedding_length": embedding_length,
        "metadata": chunk_metadata,
    }


def build_tools(
    server: FastMCP,
    *,
    context: Context,
    collection_id: str,
    indexer: Indexer,
) -> None:
    source = _get_collection_source(context, collection_id)
    collection_description = source.config.description

    @server.tool(
        name=_tool_name(collection_id, "search"),
        title=f"Search {collection_id}",
        description=(
            f"Search the {collection_id!r} collection ({collection_description}). "
            "Returns ranked documents with summaries and relevant chunks. "
            "Each result includes stable collection-prefixed identifiers that can be "
            "passed directly to the companion fetch_document and fetch_chunk tools for deeper inspection."
        ),
    )
    async def collection_search(
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the collection and return matching documents.

        Args:
            query: The search query string.
            limit: Number of matching documents to return (default: 5).

        Returns:
            A list of documents with summaries and relevant chunks.
        """
        return await core_search(
            context=context,
            indexer=indexer,
            collection_id=collection_id,
            query=query,
            limit=limit,
        )

    @server.tool(
        name=_tool_name(collection_id, "fetch_document"),
        title=f"Fetch document from {collection_id}",
        description=(
            f"Fetch a single indexed document from the {collection_id!r} collection. "
            "Use this after search when you want the document-level summary plus the full "
            "list of indexed chunks for that document. The returned chunk entries use the same "
            "identifier and length conventions as search results. Pass the fully-prefixed "
            "document_id returned by search."
        ),
    )
    async def collection_fetch_document(document_id: str) -> dict[str, Any]:
        """Fetch a document by its identifier.

        Args:
            document_id: The document identifier, starting with the collection prefix
                (e.g., 'collection:document_id'). This is the same format used in
                search and fetch_chunk results.

        Returns:
            A document with its title, keywords, summary, and either a full_text field
            (for small documents) or a chunks array with per-chunk metadata.
        """
        return await core_fetch_document(
            context=context,
            indexer=indexer,
            prefixed_document_id=document_id,
        )

    @server.tool(
        name=_tool_name(collection_id, "fetch_chunk"),
        title=f"Fetch chunk from {collection_id}",
        description=(
            f"Fetch the text for a single indexed chunk in the {collection_id!r} collection. "
            "Pass the fully-prefixed chunk_id returned by search or fetch_document. The tool "
            "derives document identity from the chunk_id, so no separate document_id is needed. "
            "Supports scope='semantic' to return the full semantic chunk and scope='embedding' "
            "to return just the embedding-sized slice used for retrieval. Returns the chunk text, "
            "summary, metadata, and the same semantic_length/embedding_length fields used elsewhere."
        ),
    )
    async def collection_fetch_chunk(
        chunk_id: str,
        scope: Literal["semantic", "embedding"] = "semantic",
    ) -> dict[str, Any]:
        """Fetch a chunk by its identifier.

        Args:
            chunk_id: The chunk identifier, starting with the collection prefix
                (e.g., 'collection:chunk_id'). This is the same format used in
                search and fetch_document results. Pass the fully-prefixed identifier
                returned by those tools to ensure the correct collection is targeted.
            scope: Either 'semantic' for the full semantic chunk or 'embedding' for
                just the embedding-sized slice. Use 'embedding' for narrowly scoped
                queries where a small text fragment will answer the query; use
                'semantic' for understanding the broader document context.

        Returns:
            A chunk with its text, summary, metadata, and semantic_length/embedding_length
            fields that indicate the response size.
        """
        return await core_fetch_chunk(
            context=context,
            prefixed_chunk_id=chunk_id,
            scope=scope,
        )


def build_all_tools(server: FastMCP, *, context: Context) -> None:
    indexer = Indexer(context)
    for collection_id in sorted(context.collections):
        build_tools(
            server,
            context=context,
            collection_id=collection_id,
            indexer=indexer,
        )


async def _build_server(
    config_path: str | list[str], *, host: str = "localhost", port: int | None = None
) -> FastMCP:
    context = Context.build_context(config_path)
    await context.build_collections()
    mcp = _create_server(host=host, port=port)
    build_all_tools(mcp, context=context)
    return mcp


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MCP document indexer server")
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=[],
        metavar="PATH",
        help="Config TOML file (repeatable; last wins)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind the HTTP MCP server to (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Run as a streamable HTTP MCP server on this port instead of stdio",
    )
    return parser


def _transport_for_args(args: argparse.Namespace) -> Literal["stdio", "streamable-http"]:
    return "streamable-http" if args.port is not None else "stdio"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    mcp = asyncio.run(_build_server(args.config, host=args.host, port=args.port))
    mcp.run(transport=_transport_for_args(args))


if __name__ == "__main__":
    main()
