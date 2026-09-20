import tomllib
from typing import Any, Dict, Literal, Type, TypeVar
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T", bound=BaseModel)

class ServerConfig(BaseModel):
    """Truly global server settings."""
    model_config = ConfigDict(extra="ignore")
    db_uri: str = "postgresql://postgres@:5432/indexer?host=/var/run/postgresql"
    log_level: str = "INFO"
    llm: Dict[str, Any] = Field(default_factory=dict)
    embedding: Dict[str, Any] = Field(default_factory=dict)
    min_size: int = 500
    max_size: int = 5000
    max_to_summarize: int = 65536
    llm_batch_size: int = 8
    embedding_batch_size: int = 10
    llm_max_batch_size: int = 8
    embedding_max_batch_size: int = 64
    llm_request_timeout: float | None = 400.0
    embedding_request_timeout: float | None = 400.0
    llm_timeout_retries: int = 20
    embedding_timeout_retries: int = 20


class RerankConfig(BaseModel):
    """Configuration for the shared reranker."""

    model_config = ConfigDict(extra="forbid")
    plugin: Literal["qwen", "llama_cpp"] = "qwen"
    command_line: list[str] | None = None
    url: str = "http://127.0.0.1:8080/v1"
    model: str | None = None
    startup_timeout: float = 60.0
    request_timeout: float = 400.0

class CollectionInfraConfig(BaseModel):
    """
    Shared infrastructure configuration for collections.
    Fields here can be provided as global defaults.
    """
    model_config = ConfigDict(extra="ignore")
    doc_summary_prompt: str = "Summarize the following document concisely..."
    chunk_summary_prompt: str = ""
    rerank: bool = False
    indexing_mode: Literal["full", "indexed"] | None = None
    
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
    Handles loading and resolving configuration from one or more TOML files.
    Files are deep-merged in order (later files win on leaf-level collisions).
    The merged result is stored as _raw_config before any semantic layer begins.
    """

    def __init__(self, config_paths: str | list[str]):
        if isinstance(config_paths, str):
            config_paths = [config_paths]
        if not config_paths:
            raise ValueError("At least one config file path is required")
        merged: dict = {}
        for path in config_paths:
            with open(path, "rb") as f:
                loaded = tomllib.load(f)
            merged = ConfigManager._deep_merge(merged, loaded)
        self._raw_config = merged

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge *override* into *base*, returning a new dict.

        When both base and override have the same key with dict values, recurse.
        Otherwise override wins.
        """
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = ConfigManager._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def get_server_config(self) -> ServerConfig:
        """Returns the [server] section as a validated ServerConfig."""
        return ServerConfig.model_validate(self._raw_config.get("server", {}))

    def get_rerank_config(self) -> RerankConfig:
        """Return the global ``[rerank]`` section."""
        return RerankConfig.model_validate(self._raw_config.get("rerank", {}))

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
