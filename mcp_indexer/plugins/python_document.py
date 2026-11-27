"""AST-aware interpretation of Python source files.

Python files are decoded using the source encoding declaration, then parsed
without importing or executing them. Classes and functions form semantic
chunks. A class larger than the configured embedding maximum is split into a
class preamble and method chunks. Chunk metadata contains byte ranges and the
encoding, so fetching a chunk does not parse the file again.
"""
from __future__ import annotations

import ast
import codecs
import io
import re
import tokenize
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from mcp_indexer.plugins.base import ChunkInfo
from mcp_indexer.plugins.plugin import DocumentPlugin
from mcp_indexer.plugins.text_document import TextDocument


_PARAGRAPH = re.compile(r"(?:\r\n|\r|\n)[ \t]*(?:\r\n|\r|\n)+")
_DOCSTRING_SUMMARY_THRESHOLD = 200

# Syntax-aware fallback boundaries for valid text that cannot be parsed as
# Python. The declaration patterns are intentionally line-oriented; this is a
# fallback path, not a substitute for AST boundaries.
PYTHON_TEXT_POLICY: dict[str, Any] = {
    "handler": "text",
    "semantic_boundary_regexps": (
        r"(?m)^[ \t]*\#[ \t]*[-=*+_]{3,}[ \t]*\r?$",
        r"(?m)^(?:(?:async[ \t]+)?def[ \t]+|class[ \t]+)",
    ),
    "embedding_boundary_regexps": (r"\n\s*\n",),
}


@dataclass(frozen=True)
class _SemanticSpan:
    start: int
    end: int
    node: ast.AST | None


@dataclass(frozen=True)
class _ParsedSource:
    text: str
    tree: ast.Module
    encoding: str
    byte_offsets: list[int]
    line_starts: list[int]
    separator_comments: list[tuple[int, int]]


class _PythonTextFallback(TextDocument):
    """TextDocument fallback that retains a per-file detected encoding."""

    def __init__(self, *args, encoding: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._python_encoding = encoding

    @property
    def encoding(self) -> str:
        return self._python_encoding


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer(r"\r\n|\r|\n", text):
        starts.append(match.end())
    return starts


def _byte_offsets(text: str, encoding: str, raw: bytes) -> list[int]:
    """Map decoded character boundaries to original byte boundaries.

    Incremental encoding handles a BOM-producing codec such as ``utf-8-sig``
    without inserting a BOM at every prefix. Reject codecs that do not
    round-trip this source exactly; their range semantics are unsuitable for
    independent byte-range retrieval.
    """
    encoder_type = codecs.getincrementalencoder(encoding)
    encoder = encoder_type()
    offsets = [0]
    byte_position = 0
    for character in text:
        byte_position += len(encoder.encode(character, final=False))
        offsets.append(byte_position)
    byte_position += len(encoder.encode("", final=True))
    offsets[-1] = byte_position
    if text.encode(encoding) != raw:
        raise UnicodeError(f"Source does not round-trip with encoding {encoding!r}")
    return offsets


def _ast_position(
    node: ast.AST,
    parsed: _ParsedSource,
    *,
    end: bool = False,
) -> int:
    """Convert an AST UTF-8 byte-column location to a decoded char offset."""
    line_number = getattr(node, "end_lineno" if end else "lineno", None)
    column = getattr(node, "end_col_offset" if end else "col_offset", None)
    if line_number is None or column is None:
        raise ValueError("AST node has incomplete source positions")
    line_index = line_number - 1
    line_start = parsed.line_starts[line_index]
    line_end = (
        parsed.line_starts[line_index + 1]
        if line_index + 1 < len(parsed.line_starts)
        else len(parsed.text)
    )
    line = parsed.text[line_start:line_end]
    utf8_prefix = line.encode("utf-8")[:column]
    char_column = len(utf8_prefix.decode("utf-8"))
    return line_start + char_column


def _decorated_start(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
                     parsed: _ParsedSource) -> int:
    """Return the line start of a definition or its first decorator."""
    first_line = node.lineno
    if node.decorator_list:
        first_line = min(
            first_line,
            *(decorator.lineno for decorator in node.decorator_list),
        )
    return parsed.line_starts[first_line - 1]


def _statement_start(node: ast.stmt, parsed: _ParsedSource) -> int:
    return parsed.line_starts[node.lineno - 1]


def _separator_comments(text: str) -> list[tuple[int, int]]:
    """Return source ranges for separator comments, excluding string text."""
    line_starts = _line_starts(text)
    result: list[tuple[int, int]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type != tokenize.COMMENT:
                continue
            if not re.fullmatch(r"\#[ \t]*[-=*+_]{3,}[ \t]*", token.string):
                continue
            start = line_starts[token.start[0] - 1]
            end_line = token.end[0]
            end = line_starts[end_line] if end_line < len(line_starts) else len(text)
            result.append((start, end))
    except (IndentationError, tokenize.TokenError):
        # Parsing already succeeded, but the tokenizer may not understand a
        # newer syntax version. Keep AST chunking without comment boundaries.
        return []
    return result


def _definition_span(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    parsed: _ParsedSource,
) -> tuple[int, int]:
    return _decorated_start(node, parsed), _ast_position(node, parsed, end=True)


def _class_spans(
    node: ast.ClassDef,
    parsed: _ParsedSource,
    max_size: int,
) -> list[_SemanticSpan]:
    start, end = _definition_span(node, parsed)
    if end - start <= max_size:
        return [_SemanticSpan(start, end, node)]

    methods = sorted(
        (member for member in node.body
         if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))),
        key=lambda member: _decorated_start(member, parsed),
    )
    method_starts = [
        _decorated_start(method, parsed)
        for method in methods
        if start < _decorated_start(method, parsed) < end
    ]
    boundaries = [start, *sorted(set(method_starts)), end]
    if len(boundaries) == 2:
        return [_SemanticSpan(start, end, node)]
    return [
        _SemanticSpan(left, right, node)
        for left, right in zip(boundaries, boundaries[1:])
        if left < right
    ]


def _module_gap_spans(
    start: int,
    end: int,
    text: str,
    separator_comments: list[tuple[int, int]],
) -> list[_SemanticSpan]:
    """Split module-level non-definition text at separator-comment lines."""
    if start >= end or not text[start:end].strip():
        return []
    boundaries = [start]
    boundaries.extend(comment_start for comment_start, _ in separator_comments
                      if start < comment_start < end)
    boundaries.append(end)
    return [
        _SemanticSpan(left, right, None)
        for left, right in zip(boundaries, boundaries[1:])
        if left < right and text[left:right].strip()
    ]


def _semantic_spans(parsed: _ParsedSource, max_size: int) -> list[_SemanticSpan]:
    spans: list[_SemanticSpan] = []
    cursor = 0
    for node in parsed.tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = _definition_span(node, parsed)
        spans.extend(_module_gap_spans(
            cursor, start, parsed.text, parsed.separator_comments,
        ))
        if isinstance(node, ast.ClassDef):
            spans.extend(_class_spans(node, parsed, max_size))
        else:
            spans.append(_SemanticSpan(start, end, node))
        cursor = end
    spans.extend(_module_gap_spans(
        cursor, len(parsed.text), parsed.text, parsed.separator_comments,
    ))
    if not spans and parsed.text.strip():
        spans.append(_SemanticSpan(0, len(parsed.text), None))
    return spans


def _candidate_parts(
    span: _SemanticSpan,
    parsed: _ParsedSource,
) -> list[str]:
    text = parsed.text[span.start:span.end]
    if not text:
        return []

    absolute_boundaries: set[int] = set()
    for match in _PARAGRAPH.finditer(text):
        absolute_boundaries.add(span.start + match.end())
    for _, comment_end in parsed.separator_comments:
        if span.start < comment_end < span.end:
            absolute_boundaries.add(comment_end)

    # AST statement boundaries make long functions and methods split at
    # meaningful syntax points. Include nested statements as candidates;
    # the semantic chunk remains the enclosing function/class span.
    if span.node is not None:
        for node in ast.walk(span.node):
            if isinstance(node, ast.stmt) and node is not span.node:
                if hasattr(node, "lineno"):
                    position = _statement_start(node, parsed)
                    if span.start < position < span.end:
                        absolute_boundaries.add(position)

    boundaries = [span.start]
    boundaries.extend(sorted(
        position for position in absolute_boundaries
        if span.start < position < span.end
    ))
    boundaries.append(span.end)
    return [
        parsed.text[left:right]
        for left, right in zip(boundaries, boundaries[1:])
        if left < right
    ]


def _outline(tree: ast.Module) -> list[str]:
    entries: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            entries.append(f"{prefix} {node.name}(…)")
        elif isinstance(node, ast.ClassDef):
            entries.append(f"class {node.name}")
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    prefix = "async def" if isinstance(member, ast.AsyncFunctionDef) else "def"
                    entries.append(f"  {prefix} {member.name}(…)")
                elif isinstance(member, ast.ClassDef):
                    entries.append(f"  class {member.name}")
    return entries


class PythonDocument(DocumentPlugin):
    """A Python file interpreted with the standard-library AST and tokenizer."""

    handler_name = "python"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parse_attempted = False
        self._parsed: _ParsedSource | None = None
        self._fallback: _PythonTextFallback | None = None

    def _make_fallback(self, encoding: str) -> _PythonTextFallback:
        if self._fallback is None or self._fallback.encoding != encoding:
            self._fallback = _PythonTextFallback(
                self.document_id,
                source=self.source,
                bridge=self.bridge,
                policy=PYTHON_TEXT_POLICY,
                last_modified=self.last_modified,
                metadata=self.metadata,
                encoding=encoding,
            )
        return self._fallback

    async def _ensure_parsed(self) -> _ParsedSource | None:
        if self._parse_attempted:
            return self._parsed
        self._parse_attempted = True
        raw = await self.bridge.read_bytes()
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            text = raw.decode(encoding)
            byte_offsets = _byte_offsets(text, encoding, raw)
            tree = ast.parse(text, filename=self.document_id)
        except (LookupError, SyntaxError, UnicodeError, ValueError):
            # Invalid declarations, undecodable bytes, unsupported range
            # codecs, and syntax errors all retain the ordinary text path.
            self._make_fallback(self.source.source_config.encoding)
            return None
        self._parsed = _ParsedSource(
            text=text,
            tree=tree,
            encoding=encoding,
            byte_offsets=byte_offsets,
            line_starts=_line_starts(text),
            separator_comments=_separator_comments(text),
        )
        return self._parsed

    def _max_size(self) -> int:
        return self.source.context.config.get_server_config().max_size

    async def get_chunks(self, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        parsed = await self._ensure_parsed()
        if parsed is None:
            fallback = self._make_fallback(self.source.source_config.encoding)
            async for metadata, candidates in fallback.get_chunks(min_size, max_size):
                metadata = dict(metadata)
                metadata["encoding"] = fallback.encoding
                yield metadata, candidates
            return

        for span in _semantic_spans(parsed, max_size):
            candidates = _candidate_parts(span, parsed)
            if not candidates:
                continue
            metadata = {
                "b": parsed.byte_offsets[span.start],
                "e": parsed.byte_offsets[span.end],
                "encoding": parsed.encoding,
            }
            yield metadata, candidates

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> str:
        """Read and decode the stored source byte range; never invoke the AST."""
        start = chunk_metadata.get("b")
        end = chunk_metadata.get("e")
        encoding = chunk_metadata.get("encoding")
        if type(start) is not int or type(end) is not int or end < start:
            raise ValueError("Python chunk metadata must contain a valid 'b'/'e' byte range")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("Python chunk metadata must contain its source encoding")
        data = await self.bridge.read_byte_range(start, end)
        return data.decode(encoding, errors="replace")

    async def get_document_summary(self) -> str:
        parsed = await self._ensure_parsed()
        max_size = self._max_size()
        if parsed is None:
            fallback = self._make_fallback(self.source.source_config.encoding)
            return (await fallback.get_document_summary())[:max_size]

        module_docstring = ast.get_docstring(parsed.tree) or ""
        if len(module_docstring) > _DOCSTRING_SUMMARY_THRESHOLD:
            summary = module_docstring
            outline = _outline(parsed.tree)
            if outline:
                summary += "\n\nDefinitions:\n"
                for entry in outline:
                    addition = entry + "\n"
                    if len(summary) + len(addition) > max_size:
                        break
                    summary += addition
            return summary[:max_size]

        fallback = self._make_fallback(parsed.encoding)
        return (await fallback.get_document_summary())[:max_size]
