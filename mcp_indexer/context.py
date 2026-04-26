from dataclasses import dataclass
import lancedb
from mcp_indexer.config import ConfigManager
from mcp_indexer.llm import LlmCall, EmbeddingCall

@dataclass
class Context:
    """
    Provides shared resources for DocumentSource plugins and the indexer.
    """
    db: lancedb.connect
    embedding: EmbeddingCall
    llm: LlmCall
    config: ConfigManager

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
        
        # Initialize LLM and Embedding calls using the server config
        llm_call = LlmCall(**server_config.llm)
        embedding_call = EmbeddingCall(**server_config.embedding)
        
        # Initialize LanceDB connection
        # Note: in a real scenario, we might need to handle the path expansion
        db = lancedb.connect(server_config.db_uri)
        
        return Context(
            db=db,
            embedding=embedding_call,
            llm=llm_call,
            config=config_manager
        )
