import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from mcp_indexer.plugins.base import ContentBridge


class ByteBridge(ContentBridge):
    """Content access to bytes held in memory.

    For content produced by a conversion (for example an exported Markdown
    document or an extracted figure) that has no existing file: the bytes
    are retained, range reads are slices, and ``path`` materializes a
    temporary file for the duration of its context.
    """

    def __init__(self, data: bytes, *, suffix: str = ""):
        self._data = data
        self._suffix = suffix

    async def read_bytes(self) -> bytes:
        return self._data

    async def read_byte_range(self, start: int, end: int) -> bytes:
        start = max(0, start)
        end = min(len(self._data), end)
        if end < start:
            return b""
        return self._data[start:end]

    @asynccontextmanager
    async def path(self):
        path = await asyncio.to_thread(_materialize, self._data, self._suffix)
        try:
            yield path
        finally:
            await asyncio.to_thread(_remove, path)


def _materialize(data: bytes, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return Path(name)


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
