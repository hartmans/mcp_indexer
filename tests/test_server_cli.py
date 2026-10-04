import argparse
import inspect
from types import SimpleNamespace

import pytest

from mcp_indexer.server import _build_parser, _create_server, _transport_for_args


@pytest.mark.parametrize("separate", [False, True])
async def test_search_tool_only_exposes_rerank_query_when_requested(monkeypatch, separate):
    from mcp_indexer import server as module
    registered = {}
    class Server:
        def tool(self, *, name, **kwargs):
            def register(function):
                registered[name] = function
                return function
            return register
    source = SimpleNamespace(separate_rerank=separate, config=SimpleNamespace(description="wiki"))
    context = SimpleNamespace(collections={"wiki": source})
    calls = []
    async def core(**kwargs):
        calls.append(kwargs)
        return []
    monkeypatch.setattr(module, "core_search", core)
    module.build_tools(Server(), context=context, indexer=object(), collection_id="wiki")
    tool = registered["wiki_search"]
    assert ("rerank_query" in inspect.signature(tool).parameters) == separate
    assert "cursor" in inspect.signature(tool).parameters
    assert inspect.signature(tool).parameters["query"].default is None
    kwargs = {"rerank_query": "natural language"} if separate else {}
    await tool("cat:Physics", **kwargs)
    assert calls[0]["query"] == "cat:Physics"
    assert calls[0].get("rerank_query") == ("natural language" if separate else None)
    await tool(cursor="continuation")
    assert calls[-1]["cursor"] == "continuation" and calls[-1]["query"] is None


def test_build_parser_defaults_host_to_localhost_and_stdio():
    args = _build_parser().parse_args(["--config", "config.toml"])

    assert args.config == ["config.toml"]
    assert args.host == "localhost"
    assert args.port is None


def test_build_parser_accepts_host_and_port():
    args = _build_parser().parse_args(
        ["--config", "config.toml", "--host", "0.0.0.0", "--port", "8123"]
    )

    assert args.host == "0.0.0.0"
    assert args.port == 8123


async def test_registered_search_tool_smoke(monkeypatch):
    from mcp_indexer import server as module

    source = SimpleNamespace(separate_rerank=True, config=SimpleNamespace(description="Wiki"))
    context = SimpleNamespace(collections={"wiki": source})
    indexer = object()
    document = SimpleNamespace(
        collection_id="wiki", document_id="article", title="Article",
        keywords=["physics"], summary="An article",
    )

    async def search(actual_indexer, collection_id, query, *, limit, rerank_query, cursor):
        assert actual_indexer is indexer
        assert (collection_id, query, limit, rerank_query) == (
            "wiki", "cat:Physics", 2, "Explain physics",
        )
        assert cursor is None
        return module.SearchResult([(document, [])], "next")

    monkeypatch.setattr(module, "search", search)
    server = _create_server()
    module.build_tools(server, context=context, indexer=indexer, collection_id="wiki")
    result = await server.call_tool("wiki_search", {
        "query": "cat:Physics", "limit": 2, "rerank_query": "Explain physics",
    })
    assert not result.is_error
    assert result.structured_content == {"results": [{
        "title": "Article", "keywords": ["physics"],
        "document_id": "wiki:article", "summary": "An article", "relevant_chunks": [],
    }], "resume_cursor": "next", "warnings": [],
        "resume_cursor_usage": "Call this tool again and pass in the cursor in the cursor argument to resume."}


def test_transport_switches_to_streamable_http_when_port_is_specified():
    assert _transport_for_args(argparse.Namespace(port=None)) == "stdio"
    assert _transport_for_args(argparse.Namespace(port=8123)) == "streamable-http"
