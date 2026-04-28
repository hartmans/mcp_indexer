#!/usr/bin/env python3
import argparse
import asyncio
import math
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lancedb
from mcp_indexer.context import Context
from mcp_indexer.config import ConfigManager
from mcp_indexer.llm import EmbeddingCall


def build_debug_context(config_path: Path, dimensions: int | None = None) -> Context:
    config = ConfigManager(str(config_path))
    server_config = config.get_server_config()

    embedding_config = server_config.embedding.copy()
    if "model_provider" in embedding_config and "provider" not in embedding_config:
        embedding_config["provider"] = embedding_config.pop("model_provider")

    db = lancedb.connect(os.path.expanduser(server_config.db_uri))
    embedding_kwargs = {
        "batch_size": server_config.embedding_batch_size,
        "request_timeout": server_config.embedding_request_timeout,
        "timeout_retries": server_config.embedding_timeout_retries,
        **embedding_config,
    }
    if dimensions is not None:
        embedding_kwargs["dimensions"] = dimensions

    return Context(
        db=db,
        embedding=EmbeddingCall(**embedding_kwargs),
        llm=None,
        config=config,
    )


def read_documents(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = list(left)
    right_values = list(right)
    size = min(len(left_values), len(right_values))
    if size == 0:
        return 0.0

    dot = sum(left_values[i] * right_values[i] for i in range(size))
    left_norm = math.sqrt(sum(left_values[i] * left_values[i] for i in range(size)))
    right_norm = math.sqrt(sum(right_values[i] * right_values[i] for i in range(size)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def top_matches(
    query_embedding: list[float],
    document_embeddings: list[list[float]],
    documents: list[str],
    limit: int,
) -> list[tuple[float, int, str]]:
    scored = [
        (cosine_similarity(query_embedding, embedding), idx, documents[idx])
        for idx, embedding in enumerate(document_embeddings)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]


async def run(args: argparse.Namespace) -> int:
    documents = read_documents(args.input_file)
    if not documents:
        print(f"No non-empty documents found in {args.input_file}", file=sys.stderr)
        return 1

    context = build_debug_context(args.config, args.dimensions)
    print(f"Embedding {len(documents)} documents...")
    document_embeddings = await context.embedding(documents)
    print(f"Embedding dimension: {len(document_embeddings[0])}")
    print("Enter prompts to query the document embeddings. Empty input exits.")

    while True:
        try:
            prompt = input("prompt> ").strip()
        except EOFError:
            print()
            break

        if not prompt:
            break

        query_text = prompt
        if args.query_instruction:
            query_text = f"Instruct: {args.query_instruction}\n Query: {prompt}"

        query_embedding = await context.embedding.query(query_text)
        for rank, (score, idx, document) in enumerate(
            top_matches(query_embedding, document_embeddings, documents, args.top),
            start=1,
        ):
            print(f"{rank}. score={score:.6f} document={idx + 1}")
            print(document)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed newline-separated documents and interactively inspect manual "
            "cosine search results without using LanceDB."
        )
    )
    parser.add_argument("input_file", type=Path, help="Text file with one document per non-empty line.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("sample_config.toml"),
        help="Indexer TOML config used to build Context and EmbeddingCall.",
    )
    parser.add_argument("--top", type=int, default=3, help="Number of matches to print per prompt.")
    parser.add_argument(
        "--dimensions",
        type=int,
        help="Override EmbeddingCall output dimensions for this debug run.",
    )
    parser.add_argument(
        "--query-instruction",
        help="Wrap prompts as Qwen3 instruction-aware retrieval queries.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
