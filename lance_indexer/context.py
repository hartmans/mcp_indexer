from dataclasses import dataclass
import lancedb
from lancedb.embeddings import EmbeddingFunction
from langchain_core.language_models.chat_models import BaseChatModel
from lance_indexer.config import ConfigManager

@dataclass
class Context:
    """
    Provides shared resources for DocumentSource plugins and the indexer.
    """
    db: lancedb.connect
    embedding_fn: EmbeddingFunction
    llm: BaseChatModel
    config: ConfigManager

    def get_table(self, table_name: str):
        """Helper to get a LanceDB table."""
        return self.db.open_table(table_name)
