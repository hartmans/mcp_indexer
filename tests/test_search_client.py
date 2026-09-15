import pytest

from mcp_indexer.search_client import _SuppressHttpLogs, format_results, split_query


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
    import logging

    assert not _SuppressHttpLogs().filter(logging.LogRecord(
        name, logging.INFO, __file__, 1, "request", (), None,
    ))
