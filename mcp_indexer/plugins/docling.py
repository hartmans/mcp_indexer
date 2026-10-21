"""Docling-based interpretation of PDF and office formats.

:class:`DoclingDocument` converts the source file with the Docling
pipeline (OCR through RapidOCR on the PyTorch backend for PDFs), exports
the result as Markdown, and interprets that Markdown through a
:class:`~mcp_indexer.plugins.text_document.TextDocument` delegate that
carries the same document ID (a delegate, not a subdocument). Extracted
figures become :class:`~mcp_indexer.plugins.images.ImageDocument`
subdocuments, and the Markdown's image references are mapped to their
subdocument IDs when the mapping is unambiguous. Page boundaries are
preserved in the Markdown: each break carries a form feed, followed by an
HTML comment naming the page that begins there, so machine consumers can
split on the form feed and LLM consumers can read the annotation.

Docling is an optional dependency: importing and registering this module
works without it, but instantiating :class:`DoclingDocument` raises an
informative :class:`ImportError`.

A single Docling converter is shared across documents so each conversion
does not pay the pipeline startup cost (OCR model loading, torch
initialization) again.  The converter — and the accelerator memory its
pipelines hold — is released :data:`DOCLING_IDLE_TIMEOUT` seconds after
the last conversion so an idle GPU does not hold memory indefinitely.

Conversion runs are additionally bounded by the context-level
``complex_document_semaphore`` (sized from the ``complex_documents``
server config, defaulting to the core count): at most that many Docling
conversions run at once, so a burst of complex documents cannot saturate
the machine (or an accelerator) with pipeline work.
"""
import asyncio
import hashlib
import io
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from mcp_indexer.plugins.base import (
    ChunkInfo,
    DocumentHandle,
    DocumentNotFoundError,
)
from mcp_indexer.plugins.bridges import ByteBridge
from mcp_indexer.plugins.file_source import MARKDOWN_POLICY
from mcp_indexer.plugins.images import ImageDocument
from mcp_indexer.plugins.plugin import DocumentPlugin
from mcp_indexer.plugins.text_document import TextDocument

# A Markdown image reference whose target is an embedded data URI, as
# produced by Docling's embedded export. The match is independent of the
# alt text.
_DATA_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(data:image/(?:png|jpe?g|webp|gif|tiff?);base64,([^)]+)\)"
)

_DOC_ERROR = (
    "Docling is required to interpret this document ({document_id}). "
    "Install the optional dependency: pip install 'mcp-indexer[docling]'. "
    "({detail})"
)


def docling_available() -> bool:
    """True when the optional Docling dependency is importable."""
    import importlib.util

    return importlib.util.find_spec("docling") is not None


def _import_docling():
    """Import the Docling conversion surface, raising informatively."""
    try:
        from docling.document_converter import (
            DocumentConverter,
            OdsFormatOption,
            OdtFormatOption,
            PdfFormatOption,
            PowerpointFormatOption,
            WordFormatOption,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            RapidOcrOptions,
        )
    except ImportError as exc:
        raise ImportError(
            "Docling is required to interpret this document. Install the "
            "optional dependency: pip install 'mcp-indexer[docling]'."
        ) from exc
    return {
        "DocumentConverter": DocumentConverter,
        "InputFormat": InputFormat,
        "PdfPipelineOptions": PdfPipelineOptions,
        "RapidOcrOptions": RapidOcrOptions,
        "PdfFormatOption": PdfFormatOption,
        "WordFormatOption": WordFormatOption,
        "PowerpointFormatOption": PowerpointFormatOption,
        "OdtFormatOption": OdtFormatOption,
        "OdsFormatOption": OdsFormatOption,
    }


# Seconds without a conversion before the shared converter (and the
# accelerator memory its pipelines hold) is released.
DOCLING_IDLE_TIMEOUT = 120.0

# Extensions this plugin can convert; validation happens before the shared
# converter is touched.
_DOC_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".rtf", ".ppt", ".pptx", ".odt", ".ods",
}

# Shared-converter state.  Conversions run in worker threads, so the lock
# is a threading lock; the idle-release timer lives on the event loop.
_shared_converter: Any | None = None
_converter_lock = threading.Lock()
_idle_release_task: asyncio.Task[None] | None = None


def _build_shared_converter() -> Any:
    """Build a converter preconfigured for every supported format.

    Docling builds pipelines lazily per format on first use, so building
    the converter is cheap; the first PDF conversion pays for the OCR
    pipeline.
    """
    imports = _import_docling()
    input_format = imports["InputFormat"]
    pdf_options = imports["PdfPipelineOptions"](
        do_ocr=True,
        ocr_options=imports["RapidOcrOptions"](backend="torch"),
        generate_page_images=False,
        generate_picture_images=True,
        do_picture_classification=False,
        do_picture_description=False,
        do_table_structure=True,
    )
    format_options: dict[Any, Any] = {
        input_format.PDF: imports["PdfFormatOption"](pipeline_options=pdf_options),
        input_format.DOC: imports["WordFormatOption"](),
        input_format.DOCX: imports["WordFormatOption"](),
        input_format.RTF: imports["WordFormatOption"](),
        input_format.PPT: imports["PowerpointFormatOption"](),
        input_format.PPTX: imports["PowerpointFormatOption"](),
        input_format.ODT: imports["OdtFormatOption"](),
        input_format.ODS: imports["OdsFormatOption"](),
    }
    return imports["DocumentConverter"](format_options=format_options)


def _shared_convert(path: str) -> Any:
    """Convert under the shared converter (blocking; call from a thread).

    Conversions are serialized: the Docling pipelines are not documented
    as thread-safe, and serializing keeps the idle-release semantics
    simple (a conversion in flight cannot race the release).
    """
    global _shared_converter
    with _converter_lock:
        converter = _shared_converter
        if converter is None:
            converter = _shared_converter = _build_shared_converter()
        return converter.convert(path)


def _free_device_cache() -> None:
    """Return accelerator memory still held after the pipeline is dropped."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cancel_idle_release() -> None:
    global _idle_release_task
    if _idle_release_task is not None and not _idle_release_task.done():
        _idle_release_task.cancel()
    _idle_release_task = None


def _schedule_idle_release() -> None:
    """Release the shared converter after an idle gap of no conversions."""
    global _idle_release_task

    async def release_after_idle() -> None:
        global _shared_converter
        await asyncio.sleep(DOCLING_IDLE_TIMEOUT)
        with _converter_lock:
            _shared_converter = None
        _free_device_cache()

    _cancel_idle_release()
    _idle_release_task = asyncio.get_running_loop().create_task(release_after_idle())


def _reset_shared_state() -> None:
    """Drop the shared converter and any pending idle release (test hook)."""
    global _shared_converter
    _cancel_idle_release()
    with _converter_lock:
        _shared_converter = None


def _map_image_references(markdown: str, figures: list[dict[str, Any]]) -> str:
    """Map embedded data-URI image references to figure subdocument IDs.

    References are matched in document order against the extracted
    figures. When the counts do not line up the mapping is ambiguous and
    the Markdown is retained as Docling produced it: no reference is
    rewritten and no placeholder is guessed at.
    """
    refs = list(_DATA_IMAGE_RE.finditer(markdown))
    if len(refs) != len(figures):
        return markdown
    parts: list[str] = []
    pos = 0
    for index, match in enumerate(refs):
        parts.append(markdown[pos : match.start()])
        parts.append(f"![Figure {index + 1}]({figures[index]['id']})")
        pos = match.end()
    parts.append(markdown[pos:])
    return "".join(parts)


def _number_page_breaks(markdown: str, doc: Any) -> str:
    """Annotate each form-feed page break with the page it begins.

    Docling's serializer walked the items in reading order and emitted a
    break wherever the page number rose past the previous item's. This
    replay records the page each break starts at; the Nth recorded page
    annotates the Nth form feed as ``\\f<!-- Page N -->``. When the
    recorded breaks and the form feeds do not line up the pairing is
    ambiguous and the Markdown is retained exactly as produced.
    """
    breaks = markdown.count("\f")
    if breaks == 0:
        return markdown
    pages: list[int] = []
    prev: int | None = None
    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page = prov[0].page_no
        if page is None:
            continue
        if prev is not None and page > prev:
            pages.append(page)
        prev = page
    if len(pages) != breaks:
        return markdown
    parts: list[str] = []
    pos = 0
    for page in pages:
        start = markdown.index("\f", pos)
        parts.append(markdown[pos:start])
        parts.append(f"\f<!-- Page {page} -->")
        pos = start + 1
    parts.append(markdown[pos:])
    return "".join(parts)


class DoclingDocument(DocumentPlugin):
    """A document interpreted through Docling conversion.

    The input bridge supplies the source file (a real filesystem path in
    the file source); the processed representation (Markdown and figure
    bytes) is retained in memory on the handle. Chunk enumeration,
    retrieval, and the bounded text excerpt are delegated to a
    TextDocument over the exported Markdown; extracted figures are
    subdocuments.
    """

    handler_name = "docling"

    def __init__(
        self,
        document_id: str,
        source: Any,
        bridge: Any,
        policy: dict[str, Any],
        last_modified: datetime,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(document_id, source, bridge, policy, last_modified, metadata)
        if not docling_available():
            raise ImportError(_DOC_ERROR.format(document_id=document_id, detail="not installed"))
        self.save_text = True
        self._markdown: str | None = None
        self._figures: list[dict[str, Any]] = []
        self._figure_handles: dict[str, ImageDocument] = {}
        self._delegate: TextDocument | None = None
        self._conversion_lock = asyncio.Lock()

    # ── conversion ────────────────────────────────────────────────────────

    async def _ensure_converted(self) -> TextDocument:
        """Convert the source file on first use; return the delegate.

        The conversion run holds a slot on the context's
        ``complex_document_semaphore`` while it takes the source path,
        runs the blocking conversion, and (re)schedules the idle release.
        The release is scheduled in a ``finally`` so a failed conversion
        cannot strand the shared converter — and the memory its
        pipelines hold — without an idle timer.
        """
        if self._delegate is not None:
            return self._delegate
        async with self._conversion_lock:
            if self._delegate is not None:
                return self._delegate
            async with self.source.context.complex_document_semaphore:
                _cancel_idle_release()
                try:
                    async with self.bridge.path() as path:
                        result = await asyncio.to_thread(self._convert_sync, path)
                finally:
                    _schedule_idle_release()
            doc = result.document
            if result.status.value not in ("success", "partial_success"):
                raise RuntimeError(
                    f"Docling conversion of {self.document_id} failed "
                    f"with status {result.status.value}"
                )
            markdown = doc.export_to_markdown(
                image_mode="embedded", page_break_placeholder="\f"
            )
            markdown = _number_page_breaks(markdown, doc)
            figures: list[dict[str, Any]] = []
            for index, picture in enumerate(doc.pictures):
                if picture.image is None:
                    continue
                buf = io.BytesIO()
                picture.image.pil_image.save(buf, format="PNG")
                data = buf.getvalue()
                digest = hashlib.sha1(data).hexdigest()[:12]
                figures.append(
                    {"id": f"fig-{index}-{digest}", "data": data, "mime": "image/png"}
                )
            markdown = _map_image_references(markdown, figures)
            self._markdown = markdown
            self._figures = figures
            self._delegate = TextDocument(
                self.document_id,
                source=self.source,
                bridge=ByteBridge(markdown.encode("utf-8")),
                policy=MARKDOWN_POLICY,
                last_modified=self.last_modified,
                metadata=self.metadata,
            )
            # Materialize the figure handles once so enumeration and
            # direct lookup return identical, durable handles.
            for figure in self._figures:
                child_id = f"{self.document_id}:{figure['id']}"
                self._figure_handles[figure["id"]] = self._make_figure_handle(
                    child_id, figure
                )
            return self._delegate

    def _convert_sync(self, path: Path) -> Any:
        """Run the Docling conversion for this file's format (blocking)."""
        suffix = path.suffix.lower()
        if suffix not in _DOC_EXTENSIONS:
            raise ValueError(f"Unsupported Docling extension: {suffix}")
        return _shared_convert(str(path))

    # ── delegated text interpretation ─────────────────────────────────────

    async def get_chunks(
        self, min_size: int, max_size: int
    ) -> AsyncGenerator[ChunkInfo, None]:
        delegate = await self._ensure_converted()
        async for chunk in delegate.get_chunks(min_size, max_size):
            yield chunk

    async def fetch_chunk(self, chunk_metadata: dict[str, Any]) -> str:
        delegate = await self._ensure_converted()
        return await delegate.fetch_chunk(chunk_metadata)

    async def get_document_summary(self) -> str:
        delegate = await self._ensure_converted()
        return await delegate.get_document_summary()

    # ── figure subdocuments ───────────────────────────────────────────────

    def _make_figure_handle(self, child_id: str, figure: dict[str, Any]) -> ImageDocument:
        return ImageDocument(
            child_id,
            source=self.source,
            bridge=ByteBridge(figure["data"], suffix=".png"),
            policy={"handler": "image", "mime": figure["mime"]},
            last_modified=self.last_modified,
            metadata={
                "title": f"{self.metadata.get('title', 'document')} figure",
                "mime": figure["mime"],
            },
        )

    async def get_sub_documents(self) -> AsyncGenerator[DocumentHandle, None]:
        await self._ensure_converted()
        for figure in self._figures:
            yield self._figure_handles[figure["id"]]

    async def fetch_sub_document(self, document_id: str) -> DocumentHandle:
        await self._ensure_converted()
        _, _, component = document_id.rpartition(":")
        handle = self._figure_handles.get(component)
        if handle is None:
            raise DocumentNotFoundError(
                document_id, f"has no figure subdocument {component}"
            )
        return handle
