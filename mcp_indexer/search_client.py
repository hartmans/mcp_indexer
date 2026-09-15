from __future__ import annotations

import argparse
import asyncio
import logging

import readline  # noqa: F401
import yaml

from .context import Context
from .indexer import Indexer
from .search import search
from .server import result_info


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
    return parser


async def run_repl(args: argparse.Namespace) -> None:
    context = Context.build_context(args.config)
    await context.build_collections()
    if args.collection_id not in context.collections:
        raise KeyError(f"Unknown collection_id: {args.collection_id}")

    indexer = Indexer(context)
    readline.parse_and_bind("tab: complete")

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
        query, rerank_query = split_query(query_input)
        if not query:
            continue

        results = await search(
            indexer,
            args.collection_id,
            query,
            limit=args.limit,
            document_candidate_limit=args.document_candidate_limit,
            chunk_candidate_limit=args.chunk_candidate_limit,
            rerank_query=rerank_query,
        )
        print(
            yaml.dump(
                result_info(results, context.collections),
                default_flow_style=False,
                sort_keys=False,
            )
        )


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()
    await run_repl(args)


if __name__ == "__main__":
    asyncio.run(main())
