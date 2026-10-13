from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Iterable
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .context import Context
from .indexer import Indexer
from .search import SearchResult, search
from .models import Document, DocumentChunk
from .plugins.base import DocumentSource
from .retrieval import fetch_chunk

logger = logging.getLogger(__name__)

# If a document's total semantic length is below this threshold we return the full text.
FULL_DOCUMENT_LENGTH = 10_000


def _create_server() -> MCPServer:
    return MCPServer("LanceDB Indexer")


def result_info(
    results: Iterable[tuple[Document, list[DocumentChunk]]],
) -> list[dict[str, Any]]:
    formatted_results: list[dict[str, Any]] = []
    for document, relevant_chunks in results:
        entry: dict[str, Any] = {
            "title": document.title,
            "keywords": list(document.keywords),
            "document_id": _document_ref(document.collection_id, document.document_id),
            "summary": document.summary,
            "relevant_embedding_chunks": [
                _relevant_chunk_info(document, chunk)
                for chunk in relevant_chunks
            ],
        }
        _expose_metadata_fields(document, entry)
        formatted_results.append(entry)
    return formatted_results


def _document_ref(collection_id: str, ref_id: str) -> str:
    return f"{collection_id}:{ref_id}"


def _expose_metadata_fields(document: Document, response: dict[str, Any]) -> None:
    """Expose persisted metadata values on a response.

    Tool-owned response fields always win; metadata never overrides them.
    """
    for key, value in (document.metadata_dict or {}).items():
        response.setdefault(key, value)


def search_result_info(page: SearchResult) -> dict[str, Any]:
    return {
        "results": result_info(page.results),
        "resume_cursor": page.resume_cursor,
        "resume_cursor_usage": (
            "End of documents reached" if page.resume_cursor is None else
            "Call this tool again and pass in the cursor in the cursor argument to resume."
        ),
        "warnings": list(page.warnings),
    }


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
    # Load the owning document and its full chunk set alongside the chunk so
    # callers can navigate to the chunk's semantic root without extra queries.
    async with context.get_session() as session:
        result = await session.execute(
            select(DocumentChunk)
            .where(
                DocumentChunk.collection_id == collection_id,
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_id == chunk_id,
            )
            .options(
                selectinload(DocumentChunk.summary),
                selectinload(DocumentChunk.document)
                .selectinload(Document.chunks)
                .selectinload(DocumentChunk.summary),
            )
        )
        return result.scalars().one_or_none()


def _relevant_chunk_info(
    document: Document,
    chunk: DocumentChunk,
) -> dict[str, Any]:
    return {
        "chunk_id": _document_ref(
            document.collection_id,
            chunk.chunk_id,
        ),
        "summary": chunk.summary.summary if chunk.summary else None,
        "semantic_length": chunk.semantic_length,
        "embedding_length": chunk.embedding_length,
    }


async def core_search(
    *,
    context: Context,
    indexer: Indexer,
    collection_id: str,
    query: str | None = None,
    limit: int = 5,
    rerank_query: str | None = None,
    cursor: str | None = None,
) -> dict[str, Any]:
    _get_collection_source(context, collection_id)
    results = await search(
        indexer,
        collection_id,
        query,
        limit=limit,
        rerank_query=rerank_query,
        cursor=cursor,
    )
    return search_result_info(results)


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

    # Compute total semantic length across the document's semantic chunks.
    # semantic_length is denormalized across every row of a semantic chunk,
    # so count only the root row (order == semantic_chunk_id) of each.
    total_semantic_length = 0
    for chunk in document.chunks:
        if chunk.is_semantic_chunk:
            total_semantic_length += chunk.semantic_length or 0

    # If the document fits within FULL_DOCUMENT_LENGTH, return the full
    # concatenated text: fetch each semantic chunk once, in order.
    if total_semantic_length < FULL_DOCUMENT_LENGTH:
        full_text_parts: list[str] = []
        current_semantic_id: int | None = None
        for chunk in document.chunks:
            if chunk.semantic_chunk_id != current_semantic_id:
                current_semantic_id = chunk.semantic_chunk_id
                part = await fetch_chunk(source, chunk.chunk_id, scope="semantic")
                full_text_parts.append(part)
        result["full_text"] = "".join(full_text_parts)
    else:
        # Otherwise group the chunks into semantic chunks. document.chunks is
        # ordered by `order`, so each semantic chunk's root row (the one with
        # order == semantic_chunk_id) is its first row. The semantic chunk's
        # summary is the root row's span summary; when the summary span
        # shifts within the chunk, the first row of the new span carries that
        # span's summary in its embedding_chunks entry.
        semantic_chunks: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        current_semantic_id: int | None = None
        last_summary_span: int | None = None
        for chunk in document.chunks:
            entry: dict[str, Any] = {
                "chunk_id": _document_ref(collection_id, chunk.chunk_id),
                "length": chunk.embedding_length,
            }
            if chunk.semantic_chunk_id != current_semantic_id:
                current = {
                    "chunk_id": _document_ref(collection_id, chunk.chunk_id),
                    "summary": chunk.summary.summary if chunk.summary else None,
                    "embedding_chunks": [entry],
                    "length": chunk.semantic_length,
                }
                semantic_chunks.append(current)
                current_semantic_id = chunk.semantic_chunk_id
                last_summary_span = chunk.summary_span
            else:
                if (
                    chunk.summary_span is not None
                    and chunk.summary_span != last_summary_span
                ):
                    entry["summary"] = (
                        chunk.summary.summary if chunk.summary else None
                    )
                current["embedding_chunks"].append(entry)
            if chunk.summary_span is not None:
                last_summary_span = chunk.summary_span
        result["semantic_chunks"] = semantic_chunks

    # Direct child visibility: enumerate the immediate subdocuments from the
    # source so children are discoverable, not merely addressable.
    handle = await source.fetch_document(document_id)
    result["sub_documents"] = [
        _document_ref(collection_id, child.document_id)
        async for child in handle.get_sub_documents()
    ]
    _expose_metadata_fields(document, result)
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

    warnings: list[str] = []
    # A semantic fetch always returns the whole semantic chunk, so if the
    # requested chunk is not the chunk's semantic root, fetch from the root
    # instead and report the substitution.
    if scope == "semantic" and not chunk.is_semantic_chunk:
        root = next(c for c in chunk.document.chunks if c.order == chunk.semantic_chunk_id)
        chunk = root
        warnings.append(
            f"Requested chunk {prefixed_chunk_id!r} is not the first chunk in its "
            "semantic chunk; returning the full semantic chunk from its first chunk "
            f"({_document_ref(collection_id, root.chunk_id)!r}) instead."
        )

    text = await fetch_chunk(source, chunk.chunk_id, scope=scope)

    return {
        "document_id": _document_ref(collection_id, document_id),
        "chunk_id": _document_ref(collection_id, chunk.chunk_id),
        "scope": scope,
        "text": text,
        "summary": chunk.summary.summary if chunk.summary else None,
        "semantic_length": chunk.semantic_length,
        "embedding_length": chunk.embedding_length,
        "warnings": warnings,
    }


def build_tools(
    server: MCPServer,
    *,
    context: Context,
    collection_id: str,
    indexer: Indexer,
) -> None:
    source = _get_collection_source(context, collection_id)
    collection_description = source.config.description

    async def collection_search(
        query: str | None = None,
        limit: int = 5,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Search the collection and return matching documents.

        Args:
            query: Required for a new search; saved query is reused on resume.
            limit: Maximum distinct matching documents to return (default: 5).
            cursor: Resume discovery, or retry the latest page-producing cursor.

        Returns:
            Documents, continuation cursor, usage instructions, and warnings.
        """
        return await core_search(
            context=context,
            indexer=indexer,
            collection_id=collection_id,
            query=query,
            limit=limit,
            cursor=cursor,
        )

    async def structured_collection_search(
        query: str | None = None, limit: int = 5, rerank_query: str | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Search with native syntax and an optional natural-language query.

        Args:
            query: Source-native expression, including supported fields and operators.
            limit: Maximum distinct matching documents to return.
            rerank_query: Natural-language information need for vector search and reranking.
                If omitted, query is used for both.
            cursor: Resume using saved queries, or replay the latest page.
        """
        return await core_search(
            context=context, indexer=indexer, collection_id=collection_id,
            query=query, limit=limit, rerank_query=rerank_query,
            cursor=cursor,
        )

    server.tool(
        name=_tool_name(collection_id, "search"),
        title=f"Search {collection_id}",
        description=(
            f"Search the {collection_id!r} collection ({collection_description}). "
            "Returns ranked documents with summaries and relevant chunks. "
            "Each result includes collection-prefixed identifiers for the companion fetch tools."
            " Pass resume_cursor as cursor to continue; query may be omitted on resume."
            " Cursors expire after 60 minutes of inactivity. The latest used cursor"
            " replays its page until the next cursor is successfully used."
            + (" Use query for native search syntax and rerank_query for the natural-language "
               "information need used by vector search and reranking." if source.separate_rerank else "")
        ),
    )(structured_collection_search if source.separate_rerank else collection_search)

    @server.tool(
        name=_tool_name(collection_id, "fetch_document"),
        title=f"Fetch document from {collection_id}",
        description=(
            f"Fetch a single indexed document from the {collection_id!r} collection. "
            "Use this after search when you want the document-level summary plus the full "
            "list of indexed chunks for that document. The returned entries use the same "
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
            (for small documents) or a semantic_chunks array. Each semantic chunk lists
            its chunk_id (the first embedding chunk), summary, length, and an
            embedding_chunks list of chunk_id/length entries.
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
            f"Fetch the text for a chunk in the {collection_id!r} collection. "
            "Pass the fully-prefixed chunk_id returned by search or fetch_document. The tool "
            "derives document identity from the chunk_id, so no separate document_id is needed. "
            "Pass an embedding chunk_id with scope='embedding' to fetch that embedding-sized "
            "slice, or a semantic chunk_id (the chunk_id of a semantic chunk's first embedding "
            "chunk) with scope='semantic' to fetch the whole semantic chunk. Returns the chunk "
            "text, summary, and the same semantic_length/embedding_length fields used elsewhere."
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
            scope: Either 'semantic' (with a semantic chunk's id) for the full semantic
                chunk or 'embedding' (with an embedding chunk's id) for just the
                embedding-sized slice. Use 'embedding' for narrowly scoped queries
                where a small text fragment will answer the query; use 'semantic' for
                understanding the broader document context.

        Returns:
            A chunk with its text, summary, and semantic_length/embedding_length
            fields that indicate the response size, plus a warnings list.
        """
        return await core_fetch_chunk(
            context=context,
            prefixed_chunk_id=chunk_id,
            scope=scope,
        )


def build_all_tools(server: MCPServer, *, context: Context) -> None:
    indexer = Indexer(context)
    for collection_id in sorted(context.collections):
        build_tools(
            server,
            context=context,
            collection_id=collection_id,
            indexer=indexer,
        )


async def _build_server(
    config_path: str | list[str],
) -> MCPServer:
    context = Context.build_context(config_path)
    await context.build_collections()
    mcp = _create_server()
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
    mcp = asyncio.run(_build_server(args.config))
    transport = _transport_for_args(args)
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
