"""Client for llama.cpp's reranking HTTP API."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from .base import AbstractReranker, RerankResult


class LlamaCppReranker(AbstractReranker):
    """Use a llama.cpp server over TCP, or launch one on a Unix socket.

    ``command_line`` selects the managed Unix-socket form. The configured
    command should contain the model and reranker options; this class appends
    ``--host <temporary-path>.sock``. Without a command, ``url`` selects an
    already-running TCP server.
    """

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:8080/v1",
        model: str | None = None,
        command_line: Sequence[str] | None = None,
        startup_timeout: float = 60.0,
        request_timeout: float = 400.0,
    ) -> None:
        if command_line is not None and not command_line:
            raise ValueError("command_line cannot be empty")
        if startup_timeout <= 0 or request_timeout <= 0:
            raise ValueError("timeouts must be positive")
        parsed = urlsplit(url)
        if command_line is None and (
            parsed.scheme != "http" or parsed.hostname is None
        ):
            raise ValueError("url must be an http TCP URL")

        self.url = url.rstrip("/")
        self.model = model
        self.command_line = tuple(command_line) if command_line is not None else None
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self._setup_lock = asyncio.Lock()
        self._ready = False
        self._process: asyncio.subprocess.Process | None = None
        self._socket_path: Path | None = None
        self._socket_directory: tempfile.TemporaryDirectory[str] | None = None

    def __del__(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.returncode is None:
            process.terminate()

    async def setup(self) -> None:
        async with self._setup_lock:
            if self._ready:
                return
            if self.command_line is not None and self._process is None:
                self._socket_directory = tempfile.TemporaryDirectory(
                    prefix="mcp-indexer-rerank-"
                )
                self._socket_path = Path(self._socket_directory.name) / "llama.sock"
                self._process = await asyncio.create_subprocess_exec(
                    *self.command_line,
                    "--host",
                    str(self._socket_path),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            await self._wait_until_ready()
            self._ready = True

    async def score(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if not self._ready:
            raise RuntimeError("LlamaCppReranker.setup() must complete before score()")
        if top_n is not None and top_n < 0:
            raise ValueError("top_n cannot be negative")
        if not documents or top_n == 0:
            return []

        payload: dict[str, object] = {
            "query": query,
            "documents": list(documents),
        }
        if self.model is not None:
            payload["model"] = self.model
        if top_n is not None:
            payload["top_n"] = top_n
        response = await self._request("POST", "/rerank", payload, use_base_path=True)
        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError("llama.cpp rerank response has no results list")
        return [
            (int(result["index"]), float(result["relevance_score"]))
            for result in results
        ]

    async def _wait_until_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout
        last_error: Exception | None = None
        while loop.time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError(
                    f"llama.cpp exited during startup with status "
                    f"{self._process.returncode}"
                )
            try:
                response = await self._request(
                    "GET", "/models", use_base_path=True
                )
                models = response.get("data")
                if not isinstance(models, list) or not models:
                    raise RuntimeError("reranker returned no models")
                matching_models = [
                    item for item in models
                    if isinstance(item, dict)
                    and (self.model is None or item.get("id") == self.model)
                ]
                if not matching_models:
                    raise RuntimeError("configured reranker model is not available")
                # llama.cpp exposes the model entry before loading finishes,
                # with an explicit null meta value. vLLM entries omit meta.
                if any(
                    "meta" not in item or item["meta"] is not None
                    for item in matching_models
                ):
                    return
                raise RuntimeError("reranker model is still loading")
            except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
                last_error = exc
                await asyncio.sleep(0.1)
        raise TimeoutError("llama.cpp did not become ready") from last_error

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        use_base_path: bool = False,
    ) -> dict[str, object]:
        async def request() -> dict[str, object]:
            if self._socket_path is not None:
                reader, writer = await asyncio.open_unix_connection(self._socket_path)
                host = "localhost"
            else:
                parsed = urlsplit(self.url)
                port = parsed.port or 80
                reader, writer = await asyncio.open_connection(parsed.hostname, port)
                host = parsed.netloc

            body = b"" if payload is None else json.dumps(payload).encode()
            target = path
            parsed_base = urlsplit(self.url)
            if use_base_path and parsed_base.path:
                target = f"{parsed_base.path.rstrip('/')}{path}"
            request_headers = [
                f"{method} {target} HTTP/1.1",
                f"Host: {host}",
                "Connection: close",
            ]
            if body:
                request_headers.extend(
                    ["Content-Type: application/json", f"Content-Length: {len(body)}"]
                )
            writer.write(("\r\n".join(request_headers) + "\r\n\r\n").encode() + body)
            await writer.drain()
            try:
                status_line = await reader.readline()
                parts = status_line.decode(errors="replace").split(maxsplit=2)
                if len(parts) < 2:
                    raise RuntimeError("invalid HTTP response from llama.cpp")
                status = int(parts[1])
                headers: dict[str, str] = {}
                while True:
                    line = await reader.readline()
                    if line in (b"\r\n", b"\n", b""):
                        break
                    name, value = line.decode(errors="replace").split(":", 1)
                    headers[name.lower()] = value.strip()
                if "content-length" in headers:
                    response_body = await reader.readexactly(int(headers["content-length"]))
                else:
                    response_body = await reader.read()
            finally:
                writer.close()
                await writer.wait_closed()

            if not 200 <= status < 300:
                raise RuntimeError(
                    f"llama.cpp returned HTTP {status}: "
                    f"{response_body.decode(errors='replace')}"
                )
            if not response_body:
                return {}
            value = json.loads(response_body)
            if not isinstance(value, dict):
                raise RuntimeError("llama.cpp returned a non-object JSON response")
            return value

        return await asyncio.wait_for(request(), timeout=self.request_timeout)
