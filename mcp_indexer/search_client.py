from __future__ import annotations

import argparse
import asyncio
import logging

import readline  # noqa: F401
import yaml

from .context import Context
from .indexer import Indexer
from .search import search, set_debug_search
from .search_state import InvalidSearchCursor, SearchAdvanceError
from .server import search_result_info


class _SuppressHttpLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not any(
            record.name == namespace or record.name.startswith(namespace + ".")
            for namespace in ("httpx", "httpcore", "httpx2", "httpcore2")
        )


class _ReadableDumper(yaml.SafeDumper):
    pass


def _represent_string(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_ReadableDumper.add_representer(str, _represent_string)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    suppress_http = _SuppressHttpLogs()
    for handler in logging.getLogger().handlers:
        handler.addFilter(suppress_http)


def format_results(results) -> str:
    return yaml.dump(
        results,
        Dumper=_ReadableDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )


def split_query(line: str) -> tuple[str, str | None]:
    """Split native discovery syntax from an optional natural-language query."""
    query, separator, rerank_query = line.partition(" => ")
    if not separator:
        return line.strip(), None
    return query.strip(), rerank_query.strip() or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive search client")
    parser.add_argument("-c", "--config", action="append", default=[],
                        metavar="PATH", help="Config TOML file (repeatable; last wins)")
    parser.add_argument("collection_id", help="Collection to search")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    parser.add_argument(
        "--document-candidate-limit",
        type=int,
        default=None,
        help="Candidate document hit limit before global ranking",
    )
    parser.add_argument(
        "--chunk-candidate-limit",
        type=int,
        default=None,
        help="Candidate chunk hit limit before global ranking",
    )
    parser.add_argument(
        "--debug-search",
        action="store_true",
        help="Log per-hit native, cosine, and reranker scores",
    )
    return parser


async def run_repl(args: argparse.Namespace) -> None:
    context = Context.build_context(args.config)
    await context.build_collections()
    if args.collection_id not in context.collections:
        raise KeyError(f"Unknown collection_id: {args.collection_id}")

    indexer = Indexer(context)
    if args.debug_search:
        set_debug_search()
    readline.parse_and_bind("tab: complete")
    cursor = None
    print("Enter a query, :more to resume, or quit. Cursors expire after 60 minutes of inactivity.")

    while True:
        try:
            query_input = input("query> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue

        if not query_input.strip():
            continue
        if query_input.strip() in {"quit", "exit"}:
            break
        resume = query_input.strip() == ":more"
        if resume:
            if cursor is None:
                print("No continuation is available. Enter a new query.")
                continue
            query, rerank_query = None, None
        else:
            query, rerank_query = split_query(query_input)
            if not query:
                continue
            cursor = None
        try:
            results = await search(
                indexer, args.collection_id, query, limit=args.limit,
                document_candidate_limit=args.document_candidate_limit,
                chunk_candidate_limit=args.chunk_candidate_limit,
                rerank_query=rerank_query, cursor=cursor,
            )
        except (InvalidSearchCursor, SearchAdvanceError, ValueError) as exc:
            print(f"Search failed: {exc}")
            continue
        cursor = results.resume_cursor
        print(format_results(search_result_info(results, context.collections)))


async def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    await run_repl(args)


if __name__ == "__main__":
    asyncio.run(main())
