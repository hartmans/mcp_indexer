import base64
from typing import Any, AsyncGenerator

from mcp_indexer.plugins.base import ChunkInfo
from mcp_indexer.plugins.plugin import DocumentPlugin


class ImageDocument(DocumentPlugin):
    """A leaf document whose content is an image.

    The content arrives through the construction bridge (typically the
    bytes of an extracted figure). It yields no chunks and no
    subdocuments; its document summary is produced by sending the image
    itself through the source's (image-capable) LLM path, and child
    summaries like this one feed the generic bottom-up parent
    summarization. ``save_text`` is set for uniformity although it has no
    effect without chunk rows.
    """

    handler_name = "image"

    def __init__(
        self,
        document_id: str,
        source: Any,
        bridge: Any,
        policy: dict[str, Any],
        last_modified: Any,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(document_id, source, bridge, policy, last_modified, metadata)
        # No chunk rows exist, so retention is a no-op; set it so the
        # handle shape matches other document plugins.
        self.save_text = True

    async def get_chunks(self, min_size: int, max_size: int) -> AsyncGenerator[ChunkInfo, None]:
        # Images are summary-only leaves: no chunks.
        return
        yield

    async def get_document_summary(self) -> str:
        """Summarize the image with the source's LLM.

        The image bytes are sent as a multimodal content part through the
        existing LLM path; no separate image-model routing is introduced.
        """
        llm = self.source.context.llm
        if llm is None:
            return ""
        data = await self.bridge.read_bytes()
        mime = self.policy.get("mime", "application/octet-stream")
        prompt = [
            ("system", self.source.config.doc_summary_prompt),
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": (
                            "Write no more than two paragraphs to summarize "
                            "the following image:"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
                        },
                    },
                ],
            ),
        ]
        summaries = await llm([prompt])
        return summaries[0]
