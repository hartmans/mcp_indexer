"""Docling-based interpretation of PDF and office formats.

:class:`DoclingDocument` converts the source file with the Docling
pipeline (OCR through RapidOCR on the PyTorch backend for PDFs), exports
the result as Markdown, and interprets that Markdown through a
:class:`~mcp_indexer.plugins.text_document.TextDocument` delegate that
carries the same document ID (a delegate, not a subdocument). Extracted
figures become :class:`~mcp_indexer.plugins.images.ImageDocument`
subdocuments, and the Markdown's image references are mapped to their
subdocument IDs when the mapping is unambiguous.

Docling is an optional dependency: importing and registering this module
works without it, but instantiating :class:`DoclingDocument` raises an
informative :class:`ImportError`.
"""
import asyncio
import hashlib
import io
import re
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
        """Convert the source file on first use; return the delegate."""
        if self._delegate is not None:
            return self._delegate
        async with self._conversion_lock:
            if self._delegate is not None:
                return self._delegate
            async with self.bridge.path() as path:
                result = await asyncio.to_thread(self._convert_sync, path)
            doc = result.document
            if result.status.value not in ("success", "partial_success"):
                raise RuntimeError(
                    f"Docling conversion of {self.document_id} failed "
                    f"with status {result.status.value}"
                )
            markdown = doc.export_to_markdown(image_mode="embedded")
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
        imports = _import_docling()
        converter_cls = imports["DocumentConverter"]
        input_format = imports["InputFormat"]
        suffix = path.suffix.lower()
        format_options: dict[Any, Any] = {}
        if suffix == ".pdf":
            options = imports["PdfPipelineOptions"](
                do_ocr=True,
                ocr_options=imports["RapidOcrOptions"](backend="torch"),
                generate_page_images=False,
                generate_picture_images=True,
                do_picture_classification=False,
                do_picture_description=False,
                do_table_structure=True,
            )
            format_options[input_format.PDF] = imports["PdfFormatOption"](
                pipeline_options=options
            )
        elif suffix in {".doc", ".docx", ".rtf"}:
            format_options[input_format.DOC] = imports["WordFormatOption"]()
            format_options[input_format.DOCX] = imports["WordFormatOption"]()
            format_options[input_format.RTF] = imports["WordFormatOption"]()
        elif suffix in {".ppt", ".pptx"}:
            format_options[input_format.PPT] = imports["PowerpointFormatOption"]()
            format_options[input_format.PPTX] = imports["PowerpointFormatOption"]()
        elif suffix == ".odt":
            format_options[input_format.ODT] = imports["OdtFormatOption"]()
        elif suffix == ".ods":
            format_options[input_format.ODS] = imports["OdsFormatOption"]()
        else:
            raise ValueError(f"Unsupported Docling extension: {suffix}")
        return converter_cls(format_options=format_options).convert(str(path))

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
