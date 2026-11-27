from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_indexer.config import ServerConfig
from mcp_indexer.plugins.base import ContentBridge
from mcp_indexer.plugins.file_source import PYTHON_POLICY, extension_policy
from mcp_indexer.plugins.plugin import HANDLER_REGISTRY
from mcp_indexer.plugins.python_document import PythonDocument


class FakeBridge(ContentBridge):
    def __init__(self, content: bytes):
        self.content = content
        self.reads = 0

    async def read_bytes(self) -> bytes:
        self.reads += 1
        return self.content

    async def read_byte_range(self, start: int, end: int) -> bytes:
        return self.content[start:end]

    def path(self):
        raise NotImplementedError


class FakeLlm:
    def __init__(self):
        self.calls = []

    async def __call__(self, prompts):
        self.calls.extend(prompts)
        return ["generated summary"] * len(prompts)


def make_document(content: bytes, *, encoding="utf-8", max_size=5000, llm=None):
    context = SimpleNamespace(
        config=SimpleNamespace(
            get_server_config=lambda: ServerConfig(max_size=max_size),
        ),
        llm=llm,
    )
    source = SimpleNamespace(
        context=context,
        source_config=SimpleNamespace(encoding=encoding, summary_length=65536),
        config=SimpleNamespace(doc_summary_prompt="Summarize the document"),
    )
    bridge = FakeBridge(content)
    return PythonDocument(
        "sample.py",
        source=source,
        bridge=bridge,
        policy=PYTHON_POLICY,
        last_modified=datetime(2026, 1, 1),
        metadata={"title": "sample.py"},
    ), bridge


@pytest.mark.asyncio
async def test_python_is_registered_by_default_and_uses_definition_chunks():
    assert extension_policy(Path("sample.py")) is PYTHON_POLICY
    assert HANDLER_REGISTRY["python"] is PythonDocument

    text = (
        "@decorate\n"
        "def first():\n"
        "    return 'one'\n\n"
        "@decorate\n"
        "class Example:\n"
        "    def method(self):\n"
        "        return 'two'\n"
    )
    document, _ = make_document(text.encode())
    chunks = [chunk async for chunk in document.get_chunks(1, 5000)]
    fetched = [await document.fetch_chunk(metadata) for metadata, _ in chunks]

    assert len(chunks) == 2
    assert "@decorate\ndef first()" in fetched[0]
    assert "@decorate\nclass Example:" in fetched[1]
    assert all(metadata["encoding"] == "utf-8" for metadata, _ in chunks)


@pytest.mark.asyncio
async def test_oversized_class_splits_preamble_and_decorated_methods_with_byte_ranges():
    raw = (
        b"# coding: latin-1\n"
        b"@register\n"
        b"class Example:\n"
        b"    label = 'caf\xe9'\n\n"
        b"    @property\n"
        b"    def first(self):\n"
        b"        return 'caf\xe9'\n\n"
        b"    @classmethod\n"
        b"    def second(cls):\n"
        b"        return 2\n"
    )
    document, bridge = make_document(raw, max_size=60)
    chunks = [chunk async for chunk in document.get_chunks(1, 60)]
    reads_after_chunking = bridge.reads
    fetched = [await document.fetch_chunk(metadata) for metadata, _ in chunks]

    assert len(chunks) == 4  # encoding-cookie preamble plus three class spans
    assert "@register\nclass Example:" in fetched[1]
    assert "@property\n    def first" in fetched[2]
    assert "@classmethod\n    def second" in fetched[3]
    assert "café" in fetched[1] + fetched[2]
    assert bridge.reads == reads_after_chunking
    for (metadata, candidates), source_text in zip(chunks, fetched):
        assert raw[metadata["b"]:metadata["e"]].decode(metadata["encoding"]) == source_text
        assert "".join(candidates) == source_text


@pytest.mark.asyncio
async def test_separator_comments_split_module_level_text_but_not_strings():
    text = (
        '"""A string containing a fake separator:\\n# ---"""\n'
        "value = 1\n"
        "# ---\n"
        "other = 2\n"
    )
    document, _ = make_document(text.encode())
    chunks = [chunk async for chunk in document.get_chunks(1, 5000)]
    fetched = ["".join(candidates) for _, candidates in chunks]

    assert len(chunks) == 2
    assert "value = 1" in fetched[0]
    assert "# ---" in fetched[1]
    assert "other = 2" in fetched[1]


@pytest.mark.asyncio
async def test_parse_and_encoding_errors_fall_back_to_text_document():
    syntax_error, _ = make_document(b"def broken(:\n    pass\n")
    chunks = [chunk async for chunk in syntax_error.get_chunks(1, 1000)]
    assert len(chunks) == 1
    assert chunks[0][0]["encoding"] == "utf-8"
    assert "def broken(:" in await syntax_error.fetch_chunk(chunks[0][0])

    bad_cookie, _ = make_document(
        b"# coding: not-a-real-codec\nvalue = 'caf\xe9'\n",
        encoding="latin-1",
    )
    fallback_chunks = [chunk async for chunk in bad_cookie.get_chunks(1, 1000)]
    assert len(fallback_chunks) == 1
    assert fallback_chunks[0][0]["encoding"] == "latin-1"
    assert "café" in await bad_cookie.fetch_chunk(fallback_chunks[0][0])


@pytest.mark.asyncio
async def test_long_module_docstring_outline_summary_is_capped_at_max_size():
    llm = FakeLlm()
    text = '"""' + ("A useful module description. " * 7) + '"""\n\nclass Example:\n    def method(self):\n        pass\n\ndef helper():\n    pass\n'
    document, _ = make_document(text.encode(), max_size=280, llm=llm)

    summary = await document.get_document_summary()

    assert len(summary) <= 280
    assert "A useful module description." in summary
    assert "class Example" in summary
    assert "def method" in summary
    assert llm.calls == []
