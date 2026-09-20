import asyncio
import gc
import json

import pytest

from mcp_indexer.rerank.llama_cpp import LlamaCppReranker


async def _http_response(writer, payload, status="200 OK"):
    body = json.dumps(payload).encode()
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def test_tcp_setup_checks_health_and_scores():
    requests = []

    async def handler(reader, writer):
        request_line = (await reader.readline()).decode().strip()
        headers = {}
        while True:
            line = await reader.readline()
            if line == b"\r\n":
                break
            name, value = line.decode().split(":", 1)
            headers[name.lower()] = value.strip()
        body = await reader.readexactly(int(headers.get("content-length", 0)))
        requests.append((request_line, json.loads(body) if body else None))
        if request_line.startswith("GET /v1/models"):
            await _http_response(writer, {"data": [{"id": "reranker"}]})
        else:
            await _http_response(
                writer,
                {"results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]},
            )

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        reranker = LlamaCppReranker(
            url=f"http://127.0.0.1:{port}/v1", model="reranker"
        )
        await reranker.setup()
        result = await reranker.score("query", ["zero", "one"], top_n=2)
    finally:
        server.close()
        await server.wait_closed()

    assert result == [(1, 0.9), (0, 0.2)]
    assert requests == [
        ("GET /v1/models HTTP/1.1", None),
        ("POST /v1/rerank HTTP/1.1", {
            "query": "query", "documents": ["zero", "one"], "top_n": 2,
            "model": "reranker",
        }),
    ]


async def test_managed_server_uses_unix_socket_and_is_terminated(monkeypatch):
    class Process:
        returncode = None

        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

    process = Process()
    launched = []

    async def create_subprocess_exec(*args, **kwargs):
        launched.append((args, kwargs))
        return process

    calls = []

    async def request(self, method, path, payload=None, **kwargs):
        calls.append((method, path, kwargs))
        return {"data": [{"id": "reranker", "meta": {}}]}

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(LlamaCppReranker, "_request", request)

    reranker = LlamaCppReranker(
        command_line=["llama-server", "-m", "reranker.gguf", "--rerank"]
    )
    await reranker.setup()
    await reranker.setup()

    args, kwargs = launched[0]
    assert args[:4] == ("llama-server", "-m", "reranker.gguf", "--rerank")
    assert args[-2] == "--host"
    assert args[-1].endswith("/llama.sock")
    assert kwargs["stdout"] == asyncio.subprocess.DEVNULL
    assert calls == [("GET", "/models", {"use_base_path": True})]

    del reranker
    gc.collect()
    assert process.terminated


async def test_score_requires_setup():
    with pytest.raises(RuntimeError, match="setup"):
        await LlamaCppReranker().score("query", ["document"])
