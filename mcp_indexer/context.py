from dataclasses import dataclass, field
import lancedb
import os
from typing import Dict, Any, TYPE_CHECKING
from mcp_indexer.config import ConfigManager
from mcp_indexer.llm import LlmCall, EmbeddingCall

# Global registry for DocumentSource plugins
SOURCE_REGISTRY: Dict[str, type] = {}


@dataclass
class Context:
    """
    Provides shared resources for DocumentSource plugins and the indexer.
    """
    db: lancedb.connect
    embedding: EmbeddingCall
    llm: LlmCall
    config: ConfigManager
    collections: Dict[str, "DocumentSource"] = field(default_factory=dict)

    def get_table(self, table_name: str):
        """Helper to get a LanceDB table."""
        return self.db.open_table(table_name)

    @staticmethod
    def build_context(config_path: str) -> 'Context':
        """
        Builds a Context object from a configuration file.
        """
        config_manager = ConfigManager(config_path)
        server_config = config_manager.get_server_config()
        
        llm_call = LlmCall(
            batch_size=server_config.llm_batch_size,
            request_timeout=server_config.llm_request_timeout,
            timeout_retries=server_config.llm_timeout_retries,
            **server_config.llm,
        )
        embedding_config = server_config.embedding.copy()
        if "model_provider" in embedding_config and "provider" not in embedding_config:
            embedding_config["provider"] = embedding_config.pop("model_provider")
        embedding_call = EmbeddingCall(
            batch_size=server_config.embedding_batch_size,
            request_timeout=server_config.embedding_request_timeout,
            timeout_retries=server_config.embedding_timeout_retries,
            **embedding_config,
        )
        
        # Expand tilde in db_uri
        db_uri = os.path.expanduser(server_config.db_uri)
        db = lancedb.connect(db_uri)
        
        ctx = Context(
            db=db,
            embedding=embedding_call,
            llm=llm_call,
            config=config_manager
        )
        
        ctx.build_collections()
        return ctx

    def build_collections(self):
        """
        Builds the collections mapping based on the configuration.
        Raises ValueError if a collection cannot be initialized.
        """
        import mcp_indexer.plugins  # noqa: F401

        for collection_id in self.config.list_collections():
            col_config = self.config.get_collection_config(collection_id)
            
            prefix = col_config.source_blob.get("type")
            if not prefix:
                raise ValueError(f"Collection '{collection_id}' is missing 'type' in source_config.")
                
            cls = SOURCE_REGISTRY.get(prefix)
            if not cls:
                raise ValueError(f"No DocumentSource plugin registered for prefix '{prefix}' (collection '{collection_id}').")
                
            self.collections[collection_id] = cls(
                collection_id=collection_id,
                context=self,
                collection_config=col_config
            )

if TYPE_CHECKING:
    from .plugins.base import DocumentSource
