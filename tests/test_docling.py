"""Docling plugin tests.

Generic infrastructure (extension mapping, handler registration, bridge
behavior, reference mapping, instantiation failure without the optional
dependency) runs without Docling. Real conversion tests use the checked-in
short fixtures under ``tests/fixtures`` and are skipped only when Docling
itself is unavailable.
"""
import asyncio
import base64
import re
from datetime import datetime
from pathlib import Path

import pytest

from mcp_indexer.context import Context
from mcp_indexer.plugins.bridges import ByteBridge
from mcp_indexer.plugins.docling import (
    DoclingDocument,
    _map_image_references,
    docling_available,
)
from mcp_indexer.plugins.file_source import (
    DOCLING_POLICY,
    FileSource,
    extension_policy,
)
from mcp_indexer.plugins.images import ImageDocument
from mcp_indexer.plugins.plugin import HANDLER_REGISTRY

FIXTURES = Path(__file__).parent / "fixtures"

requires_docling = pytest.mark.skipif(
    not docling_available(),
    reason="Docling (optional dependency) is not installed",
)

DOC_LING_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".odt", ".ods", ".ppt", ".pptx", ".rtf",
)


class SimpleCfg:
    doc_summary_prompt = "Summarize documents."

    def __init__(self, directory, **kwargs):
        self.source_blob = {"directory": str(directory), **kwargs}
        self.plugin = {}

    def resolve_source_config(self, model):
        return model.model_validate(self.source_blob)


class FakeLlm:
    """Records prompts and returns a canned summary per prompt."""

    def __init__(self, response="A fake image summary."):
        self.response = response
        self.prompts = []

    async def __call__(self, prompts):
        self.prompts.extend(prompts)
        return [self.response] * len(prompts)


def make_source(tmp_path):
    for name in ("sample.pdf", "sample.docx", "sample.doc"):
        (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
    context = Context(
        engine=None, session_factory=None, embedding=None, llm=None, config=None,
    )
    return FileSource("col", context=context, collection_config=SimpleCfg(tmp_path))


async def full_text(handle) -> str:
    parts = []
    async for metadata, _ in handle.get_chunks(50, 4000):
        parts.append(await handle.fetch_chunk(metadata))
    return "".join(parts)


def figure_refs(text: str) -> list[str]:
    return re.findall(r"!\[Figure \d+\]\((fig-[^)]+)\)", text)


# ── generic infrastructure (no Docling required) ───────────────────────────


def test_extension_mappings_select_docling():
    for extension in DOC_LING_EXTENSIONS:
        policy = extension_policy(Path(f"document{extension}"))
        assert policy is DOCLING_POLICY
        assert policy["handler"] == "docling"


def test_handlers_registered():
    assert HANDLER_REGISTRY["docling"] is DoclingDocument
    assert HANDLER_REGISTRY["image"] is ImageDocument


def test_instantiation_without_docling_raises_informatively(monkeypatch):
    import mcp_indexer.plugins.docling as docling_module

    monkeypatch.setattr(docling_module, "docling_available", lambda: False)
    with pytest.raises(ImportError, match="mcp-indexer\\[docling\\]"):
        DoclingDocument(
            "sample.pdf",
            source=None,
            bridge=None,
            policy=DOCLING_POLICY,
            last_modified=datetime(2026, 1, 1),
        )


@pytest.mark.asyncio
async def test_byte_bridge_reads_and_paths():
    data = b"abc" * 100
    bridge = ByteBridge(data)
    assert await bridge.read_bytes() == data
    assert await bridge.read_byte_range(3, 9) == data[3:9]
    # Out-of-range reads clamp instead of wrapping or raising.
    assert await bridge.read_byte_range(-5, 4) == data[:4]
    assert await bridge.read_byte_range(500, 1000) == b""
    assert await bridge.read_byte_range(10, 5) == b""

    async with bridge.path() as path:
        assert path.is_file()
        assert path.read_bytes() == data
        gone = path
    assert not gone.exists()


def test_map_image_references_maps_in_order():
    figures = [
        {"id": "fig-0-aaaa", "data": b"", "mime": "image/png"},
        {"id": "fig-1-bbbb", "data": b"", "mime": "image/png"},
    ]
    markdown = (
        "intro\n"
        "![Image](data:image/png;base64,AAAA)\n"
        "middle\n"
        "![Image](data:image/png;base64,BBBB)\n"
    )
    mapped = _map_image_references(markdown, figures)
    assert "![Figure 1](fig-0-aaaa)" in mapped
    assert "![Figure 2](fig-1-bbbb)" in mapped
    assert "data:image/png;base64," not in mapped


def test_map_image_references_refuses_ambiguous_counts():
    markdown = "![Image](data:image/png;base64,AAAA)\n"
    # Two figures for one reference: the mapping is ambiguous, so the
    # Markdown is retained exactly as produced.
    mapped = _map_image_references(
        markdown,
        [{"id": "fig-0-aaaa"}, {"id": "fig-1-bbbb"}],
    )
    assert mapped == markdown
    # Zero figures for one reference is also ambiguous.
    assert _map_image_references(markdown, []) == markdown


@pytest.mark.asyncio
async def test_image_document_summarizes_without_chunks(tmp_path):
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-png-payload"
    bridge = ByteBridge(image_bytes, suffix=".png")
    llm = FakeLlm("The figure shows a bar chart.")

    class FakeSourceConfig:
        doc_summary_prompt = "Summarize documents."

    class FakeSource:
        def __init__(self):
            self.context = type("Ctx", (), {"llm": llm})()
            self.config = FakeSourceConfig()

    image = ImageDocument(
        "parent.pdf:fig-0-aaaa",
        source=FakeSource(),
        bridge=bridge,
        policy={"handler": "image", "mime": "image/png"},
        last_modified=datetime(2026, 1, 1),
        metadata={"title": "sample.pdf figure"},
    )
    assert image.save_text is True

    summary = await image.get_document_summary()
    assert summary == "The figure shows a bar chart."
    # The image content went through the LLM path as a multimodal part.
    assert len(llm.prompts) == 1
    user_content = llm.prompts[0][1][1][1]
    assert user_content["type"] == "image_url"
    assert base64.b64encode(image_bytes).decode("ascii") in user_content[
        "image_url"
    ]["url"]
    assert user_content["image_url"]["url"].startswith("data:image/png;base64,")

    # Summary-only leaf: no chunks, no subdocuments.
    assert [c async for c in image.get_chunks(50, 4000)] == []
    assert [c async for c in image.get_sub_documents()] == []


def test_unsupported_extension_raises_before_conversion(monkeypatch):
    import mcp_indexer.plugins.docling as docling_module

    monkeypatch.setattr(docling_module, "docling_available", lambda: True)
    doc = DoclingDocument(
        "sample.txt",
        source=None,
        bridge=None,
        policy=DOCLING_POLICY,
        last_modified=datetime(2026, 1, 1),
    )
    with pytest.raises(ValueError, match="Unsupported Docling extension"):
        doc._convert_sync(Path("sample.txt"))


# ── real conversion (requires Docling) ────────────────────────────────────


@requires_docling
@pytest.mark.long_cpu
@pytest.mark.asyncio
async def test_pdf_conversion_ocr_figures_and_retrieval(tmp_path):
    source = make_source(tmp_path)
    doc = await source.fetch_document("sample.pdf")
    assert isinstance(doc, DoclingDocument)
    assert doc.save_text is True

    metadata = await doc.get_metadata()
    assert metadata["title"] == "sample.pdf"
    assert "last_modified" in metadata

    chunks = [c async for c in doc.get_chunks(50, 4000)]
    assert chunks
    # Candidates partition the semantic text; retrieval matches.
    for chunk_metadata, candidates in chunks:
        assert await doc.fetch_chunk(chunk_metadata) == "".join(candidates)

    text = await full_text(doc)
    # The scanned page has no text layer: the OCR pass must have produced
    # its content, not merely the first page's native text.
    assert "QUICKBROWNFOX" in text
    assert "OCRMEASUREMENT" in text

    children = [c async for c in doc.get_sub_documents()]
    assert len(children) == 1
    child = children[0]
    assert isinstance(child, ImageDocument)
    assert child.document_id == f"sample.pdf:{child.document_id.rpartition(':')[2]}"

    # The Markdown reference identifies the figure subdocument.
    assert figure_refs(text) == [child.document_id.rpartition(":")[2]]

    # The figure content is a real PNG.
    data = await child.bridge.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"

    # Direct fetch of the child by full ID reuses the enumerated handle.
    assert (await source.fetch_document(child.document_id)) is child


@requires_docling
@pytest.mark.asyncio
async def test_docx_and_doc_conversion(tmp_path):
    source = make_source(tmp_path)
    for name in ("sample.docx", "sample.doc"):
        doc = await source.fetch_document(name)
        assert isinstance(doc, DoclingDocument)

        chunks = [c async for c in doc.get_chunks(50, 4000)]
        assert chunks
        for chunk_metadata, candidates in chunks:
            assert await doc.fetch_chunk(chunk_metadata) == "".join(candidates)

        children = [c async for c in doc.get_sub_documents()]
        assert len(children) == 1
        text = await full_text(doc)
        assert figure_refs(text) == [children[0].document_id.rpartition(":")[2]]


@requires_docling
@pytest.mark.long_cpu
@pytest.mark.asyncio
async def test_repeat_processing_yields_consistent_enumeration(tmp_path):
    source = make_source(tmp_path)
    first = await source.fetch_document("sample.pdf")
    first_children = [c.document_id async for c in first.get_sub_documents()]
    assert first_children

    # A fresh handle re-converts independently and must enumerate the
    # same subdocuments (best-effort ID stability for unchanged content).
    fresh = DoclingDocument(
        "sample.pdf",
        source=source,
        bridge=first.bridge,
        policy=DOCLING_POLICY,
        last_modified=first.last_modified,
        metadata={"title": "sample.pdf"},
    )
    fresh_children = [c.document_id async for c in fresh.get_sub_documents()]
    assert fresh_children == first_children


@requires_docling
@pytest.mark.long_cpu
@pytest.mark.asyncio
async def test_shared_converter_reused_across_documents(tmp_path, monkeypatch):
    import mcp_indexer.plugins.docling as docling_module

    docling_module._reset_shared_state()
    built = []
    real_build = docling_module._build_shared_converter

    def counting_build():
        converter = real_build()
        built.append(converter)
        return converter

    monkeypatch.setattr(docling_module, "_build_shared_converter", counting_build)

    source = make_source(tmp_path)
    for name in ("sample.pdf", "sample.docx"):
        doc = await source.fetch_document(name)
        assert await full_text(doc)

    # One converter served both formats; its pipelines were built once.
    assert len(built) == 1
    assert docling_module._shared_converter is built[0]
    docling_module._reset_shared_state()


@requires_docling
@pytest.mark.long_cpu
@pytest.mark.asyncio
async def test_shared_converter_released_after_idle_timeout(tmp_path, monkeypatch):
    import mcp_indexer.plugins.docling as docling_module

    docling_module._reset_shared_state()
    assert docling_module.DOCLING_IDLE_TIMEOUT == 120.0
    monkeypatch.setattr(docling_module, "DOCLING_IDLE_TIMEOUT", 0.3)

    source = make_source(tmp_path)
    doc = await source.fetch_document("sample.pdf")
    assert await full_text(doc)
    converter = docling_module._shared_converter
    assert converter is not None

    # After the idle timeout the shared converter is dropped.
    await asyncio.sleep(0.6)
    assert docling_module._shared_converter is None

    # The next conversion rebuilds a fresh converter. The source's handle
    # cache returns the same already-converted document, so construct a
    # fresh handle directly (as the repeat-processing test does) to force
    # a real second conversion.
    again = DoclingDocument(
        "sample.pdf",
        source=source,
        bridge=doc.bridge,
        policy=DOCLING_POLICY,
        last_modified=doc.last_modified,
        metadata={"title": "sample.pdf"},
    )
    assert await full_text(again)
    assert docling_module._shared_converter is not None
    assert docling_module._shared_converter is not converter
    docling_module._reset_shared_state()


@requires_docling
@pytest.mark.asyncio
async def test_conversions_bounded_by_context_semaphore(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace

    import mcp_indexer.plugins.docling as docling_module

    docling_module._reset_shared_state()
    state = {"in_flight": 0, "peak": 0, "done": 0}

    def bounded_convert(path):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        time.sleep(0.4)
        state["in_flight"] -= 1
        state["done"] += 1
        return SimpleNamespace(
            status=SimpleNamespace(value="success"),
            document=SimpleNamespace(
                export_to_markdown=lambda **kwargs: "converted text. " * 20 + "\n",
                pictures=[],
            ),
        )

    monkeypatch.setattr(docling_module, "_shared_convert", bounded_convert)

    source = make_source(tmp_path)
    source.context.complex_document_semaphore = asyncio.Semaphore(2)
    base = await source.fetch_document("sample.pdf")

    docs = [
        DoclingDocument(
            f"doc-{i}.pdf",
            source=source,
            bridge=base.bridge,
            policy=DOCLING_POLICY,
            last_modified=base.last_modified,
            metadata={"title": f"doc-{i}.pdf"},
        )
        for i in range(4)
    ]

    async def convert(doc):
        async for _ in doc.get_chunks(50, 4000):
            pass

    await asyncio.gather(*(convert(doc) for doc in docs))

    assert state["done"] == 4
    assert state["in_flight"] == 0
    # The context semaphore (2) bounded the four concurrent conversion runs;
    # without it all four would have been in flight at once.
    assert state["peak"] <= 2
    docling_module._reset_shared_state()


@requires_docling
@pytest.mark.asyncio
async def test_figure_summary_travels_the_llm_path(tmp_path):
    source = make_source(tmp_path)
    doc = await source.fetch_document("sample.docx")
    child = [c async for c in doc.get_sub_documents()][0]

    llm = FakeLlm("The figure shows quarterly results.")
    source.context.llm = llm
    summary = await child.get_document_summary()
    assert summary == "The figure shows quarterly results."
    assert len(llm.prompts) == 1
