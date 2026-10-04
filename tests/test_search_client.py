import logging

import pytest

import mcp_indexer.search_client as search_client_module
from mcp_indexer.search_client import (
    _SuppressHttpLogs,
    build_parser,
    format_results,
    split_query,
)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("cat:Physics", ("cat:Physics", None)),
        (
            "cat:Physics AND title:Quantum => Explain quantum physics => simply",
            ("cat:Physics AND title:Quantum", "Explain quantum physics => simply"),
        ),
        ("cat:Physics => ", ("cat:Physics", None)),
    ],
)
def test_split_query(line, expected):
    assert split_query(line) == expected


def test_format_results_uses_unicode_and_literal_blocks():
    output = format_results([{"title": "GNU variants", "summary": "GNU’s tools\n\nSecond paragraph"}])

    assert "GNU’s tools" in output
    assert "summary: |-" in output
    assert "\\u" not in output
    assert "\\n" not in output


@pytest.mark.parametrize(
    "name", ["httpx", "httpx.client", "httpcore.http11", "httpx2", "httpcore2.http11"],
)
def test_http_log_filter_suppresses_client_namespaces(name):
    assert not _SuppressHttpLogs().filter(logging.LogRecord(
        name, logging.INFO, __file__, 1, "request", (), None,
    ))


def test_build_parser_exposes_debug_search_defaulting_off():
    args = build_parser().parse_args(["wiki"])
    assert args.debug_search is False


def test_build_parser_accepts_debug_search_flag():
    args = build_parser().parse_args(["wiki", "--debug-search"])
    assert args.debug_search is True


async def test_run_repl_enables_debug_search_only_when_flagged(monkeypatch):
    """run_repl toggles the search logger level only for --debug-search."""
    from mcp_indexer import search as searching

    class FakeContext:
        collections = {"wiki": object()}

        async def build_collections(self):
            pass

    class Context:
        @staticmethod
        def build_context(paths):
            return FakeContext()

    monkeypatch.setattr(search_client_module, "Context", Context)
    monkeypatch.setattr(search_client_module, "Indexer", lambda context: object())

    async def search(indexer, collection_id, query, limit=5, **kwargs):
        return []

    monkeypatch.setattr(search_client_module, "search", search)
    monkeypatch.setattr(search_client_module, "search_result_info",
                        lambda results, collections: [])
    monkeypatch.setattr("builtins.input", lambda prompt="": "quit")

    searching.set_debug_search(False)
    await search_client_module.run_repl(build_parser().parse_args(["wiki"]))
    assert searching.logger.level != logging.DEBUG

    searching.set_debug_search(False)
    await search_client_module.run_repl(build_parser().parse_args(["wiki", "--debug-search"]))
    assert searching.logger.level == logging.DEBUG
    searching.set_debug_search(False)


async def test_more_resumes_saved_search_and_retries_errors(monkeypatch, capsys):
    from types import SimpleNamespace
    from mcp_indexer.search import SearchResult
    from mcp_indexer.search_state import SearchAdvanceError

    context = SimpleNamespace(collections={"wiki": object()})
    async def build():
        pass
    context.build_collections = build
    monkeypatch.setattr(search_client_module, "Context", SimpleNamespace(build_context=lambda paths: context))
    monkeypatch.setattr(search_client_module, "Indexer", lambda context: object())
    calls = []
    async def search(indexer, collection_id, query, **kwargs):
        calls.append((query, kwargs))
        if len(calls) == 2:
            raise SearchAdvanceError("temporarily unavailable")
        return SearchResult([], "next" if len(calls) == 1 else None)
    monkeypatch.setattr(search_client_module, "search", search)
    lines = iter(["cat:Physics => natural", ":more", ":more", ":more", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    await search_client_module.run_repl(build_parser().parse_args(["wiki"]))
    assert calls[0][0] == "cat:Physics" and calls[0][1]["rerank_query"] == "natural"
    assert calls[1][0] is None and calls[1][1]["cursor"] == "next"
    assert calls[2][0] is None and calls[2][1]["cursor"] == "next"
    output = capsys.readouterr().out
    assert "temporarily unavailable" in output
    assert "End of documents reached" in output and "No continuation" in output
