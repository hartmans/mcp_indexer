#!/usr/bin/env python3
"""Convert one file-source document ID per line into an excludes file."""

import argparse
import sys
from pathlib import Path
from urllib.parse import unquote

from mcp_indexer.plugins.file_source import FileSource


def document_ids_to_paths(lines: list[str], collection_id: str | None = None) -> list[str]:
    """Decode and normalize local file-source IDs, returning unique paths."""
    paths = set()
    prefix = f"{collection_id}:" if collection_id else None
    for line_number, line in enumerate(lines, start=1):
        document_id = line.strip()
        if not document_id or document_id.startswith("#"):
            continue
        if prefix and document_id.startswith(prefix):
            document_id = document_id[len(prefix):]
        path = FileSource._normalize_excluded_path(
            unquote(document_id), source=f"document ID line {line_number}"
        )
        paths.add(path)
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert file-source document IDs to relative paths in an excludes file."
    )
    parser.add_argument(
        "ids_file", nargs="?", default="-",
        help="text file with one document ID per line (default: stdin)",
    )
    parser.add_argument(
        "-o", "--output", default=".excludes",
        help="output excludes file (default: .excludes; use - for stdout)",
    )
    parser.add_argument(
        "--collection-id",
        help="strip this collection ID prefix from IDs formatted as collection:path",
    )
    args = parser.parse_args()

    if args.ids_file == "-":
        lines = sys.stdin.read().splitlines()
    else:
        lines = Path(args.ids_file).read_text(encoding="utf-8").splitlines()
    result = "".join(f"{path}\n" for path in document_ids_to_paths(lines, args.collection_id))

    if args.output == "-":
        sys.stdout.write(result)
    else:
        Path(args.output).write_text(result, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
