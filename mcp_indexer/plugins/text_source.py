from mcp_indexer.plugins.file_source import FileSource, FileSourceConfig, FileSourcePointer

class TextFileSourceConfig(FileSourceConfig):
    """Configuration for the TextFileSource plugin."""
    # We can add specific text-plugin settings here if needed, 
    # e.g., custom encoding or specific parsing flags.
    encoding: str = "utf-8"
    summary_length: int = 65536

class TextFilePointer(FileSourcePointer):
    """
    A DocumentPointer for simple text files.
    """
    async def get_metadata(self) -> dict[str, str]:
        """
        Extracts metadata from a text file. 
        For a simple implementation, we use the filename as the title.
        """
        return await super().get_metadata()

class TextFileSource(FileSource[TextFilePointer]):
    """
    A concrete FileSource for handling simple text files (Markdown, RST, TXT).
    """
    
    source_prefix = "text"

    # Semantic boundaries:
    # 1. RST style headers (underline of = or -)
    # 2. Markdown style headers (# Header)
    # 3. Large gaps of whitespace (though handled by embedding boundaries, 
    #    we might want them as semantic breaks)
    semantic_boundary_regexps = (
        rb"\n={3,}\n",           # RST Level 1
        rb"\n-{3,}\n",           # RST Level 2
        rb"\n\s*#{1,6}\s+.*",    # Markdown Headers
    )

    # Embedding boundaries:
    # 1. Double newlines (paragraphs)
    # 2. Horizontal rules
    embedding_boundary_regexps = (
        rb"\n\s*\n",             # Paragraphs
        rb"\n\s*-{3,}\s*\n",     # MD/RST Horizontal rules
    )

    def __init__(self, collection_id: str, *, context, collection_config, reranker=None):
        super().__init__(
            collection_id,
            context=context,
            collection_config=collection_config,
            reranker=reranker,
        )
        # Resolve the specific TextFileSourceConfig
        self.source_config = collection_config.resolve_source_config(TextFileSourceConfig)

    async def get_document_summary(self, document_id: str) -> str:
        pointer = self.fetch_document(document_id)
        text = pointer.path.read_text(
            encoding=self.source_config.encoding,
            errors="replace",
        )
        if len(text) <= self.source_config.summary_length and self.context.llm is not None:
            prompt = [
                    ('system',self.config.doc_summary_prompt),
                    ('user', f"Write no more than two paragraphs to summarize the following document:\n\n{text}")]
            summaries = await self.context.llm([prompt])
            return summaries[0]
        return ""

    def fetch_document(self, document_id: str) -> TextFilePointer:
        """Resolves a document_id into a TextFilePointer."""
        relative_path_str = self.decode_document_path(document_id)
        absolute_path = self.source_config.directory / relative_path_str
        return TextFilePointer(self, document_id, absolute_path)
