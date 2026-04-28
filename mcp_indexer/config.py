import tomllib
from typing import Any, Dict, Type, TypeVar
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)

class ServerConfig(BaseModel):
    """Truly global server settings."""
    model_config = ConfigDict(extra="ignore")
    db_uri: str = "~/.lancedb"
    log_level: str = "INFO"
    llm: Dict[str, Any] = Field(default_factory=dict)
    embedding: Dict[str, Any] = Field(default_factory=dict)
    min_size: int = 500
    max_size: int = 5000
    max_to_summarize: int = 65536
    llm_batch_size: int = 10
    embedding_batch_size: int = 10
    llm_request_timeout: float | None = 400.0
    embedding_request_timeout: float | None = 400.0
    llm_timeout_retries: int | None = None
    embedding_timeout_retries: int | None = None

class CollectionInfraConfig(BaseModel):
    """
    Shared infrastructure configuration for collections.
    Fields here can be provided as global defaults.
    """
    model_config = ConfigDict(extra="ignore")
    doc_summary_prompt: str = "Summarize the following document concisely..."
    chunk_summary_prompt: str = ""
    
class CollectionConfig(CollectionInfraConfig):
    """
    The resolved configuration for a specific collection.
    Inherits global defaults but can define fields that MUST be set per collection.
    """
    model_config = ConfigDict(extra="ignore")
    collection_id: str
    tool_prefix: str  # Required per collection, no global default
    description: str = "Document collection"
    
    # The raw blob for the DocumentSource plugin
    source_blob: Dict[str, Any] = Field(default_factory=dict)

    def resolve_source_config(self, model: Type[T]) -> T:
        """
        Casts the raw source_config blob into a specific Pydantic model.
        This allows DocumentSource plugins to define their own validation schemas.
        """
        return model.model_validate(self.source_blob)

class ConfigManager:
    """
    Handles loading and resolving configuration from a TOML file.
    """

    def __init__(self, config_path: str):
        with open(config_path, "rb") as f:
            self._raw_config = tomllib.load(f)

    def get_server_config(self) -> ServerConfig:
        """Returns the [server] section as a validated ServerConfig."""
        return ServerConfig.model_validate(self._raw_config.get("server", {}))

    def list_collections(self) -> list[str]:
        """Returns a list of all configured collection IDs."""
        return list(self._raw_config.get("collections", {}).keys())

    def get_collection_config(self, collection_id: str) -> CollectionConfig:
        """
        Resolves the final configuration for a collection.
        
        Merges:
        1. [defaults] (Global infrastructure defaults)
        2. [collections.<id>] (Collection-specific overrides and required fields)
        """
        collections_section = self._raw_config.get("collections", {})
        if collection_id not in collections_section:
            raise KeyError(f"Collection '{collection_id}' not found in configuration.")

        col_spec = collections_section[collection_id].copy()
        
        # Extract the source-specific configuration blob
        source_blob = col_spec.pop("source_config", {})
        
        # Merge: Global Defaults < Collection Overrides
        defaults = self._raw_config.get("defaults", {})
        merged_dict = {**defaults, **col_spec}
        
        # Add the metadata and the blob for validation
        merged_dict["collection_id"] = collection_id
        merged_dict["source_blob"] = source_blob
        
        # Validation via Pydantic model
        # This will raise a ValidationError if tool_prefix is missing
        return CollectionConfig.model_validate(merged_dict)
