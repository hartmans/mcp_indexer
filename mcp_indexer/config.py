import os
import tomllib
from typing import Any, Dict, Literal, Type, TypeVar
from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T", bound=BaseModel)


def normalize_extension(extension: str) -> str:
    """Canonicalize a file extension for extension-map keys.

    Case-insensitive with a leading dot: ``MD``, ``md``, and ``.md`` all
    normalize to ``.md``, matching ``path.suffix.lower()`` lookups.
    """
    extension = extension.strip().lower()
    if not extension.startswith("."):
        extension = "." + extension
    return extension


class PluginPolicyConfig(BaseModel):
    """One entry of a collection's ``plugin`` list.

    ``handler`` names the document plugin and ``extensions`` lists the
    file extensions the entry covers; both are validated here. Any
    additional keys are policy keys passed through to the selected
    plugin verbatim. Sources consume ``extensions`` to key their
    extension-to-policy map; the remaining keys form the policy.
    """
    model_config = ConfigDict(extra="allow")
    handler: str = Field(min_length=1)
    extensions: list[str] = Field(min_length=1)

    @field_validator("extensions")
    @classmethod
    def _canonicalize_extensions(cls, value: list[str]) -> list[str]:
        normalized = []
        for extension in value:
            extension = extension.strip().lower()
            if not extension:
                raise ValueError("extension must be a non-empty string")
            normalized.append(normalize_extension(extension))
        return normalized


def default_complex_documents() -> int:
    """Default slot count for complex document conversion: the core count."""
    return os.cpu_count() or 1


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
    # Concurrency for complex (pipeline-heavy) document conversion.
    complex_documents: int = Field(default_factory=default_complex_documents)


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
    min_cosine_distance: float = 0.65
    min_rerank_score: float = 0.5
    # Extension-to-policy map, resolved from the [[defaults.plugin]] /
    # [[collections.<id>.plugin]] table lists: one policy per
    # extension, collection entries overriding defaults. FileSources
    # layer it over the code-defined extension map; other sources
    # ignore it.
    plugin: dict[str, dict[str, Any]] = Field(default_factory=dict)
    
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

    @staticmethod
    def _merge_plugin_entries(default_entries, collection_entries) -> dict[str, dict[str, Any]]:
        """Merge defaults and collection ``plugin`` lists into the
        extension-to-policy map.

        Both raw lists are validated first, so a malformed entry raises
        ``ValidationError`` instead of being silently dropped. Each
        entry's ``extensions`` key is consumed to key the map; the
        remaining entry keys form the policy. A collection entry for an
        extension replaces the defaults entry for that extension
        wholly, and later entries within a single list win over earlier
        ones.
        """
        merged: dict[str, dict[str, Any]] = {}
        for entries in (default_entries, collection_entries):
            for entry in entries or []:
                validated = PluginPolicyConfig.model_validate(entry)
                policy = validated.model_dump(exclude={"extensions"})
                for extension in validated.extensions:
                    merged[extension] = policy
        return merged

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

        The ``plugin`` lists merge by extension rather than wholesale:
        collection entries supplement the defaults and override them
        per extension (see :meth:`_merge_plugin_entries`).
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

        # The shallow dict merge above would replace the whole plugin
        # value, so redo the key explicitly: defaults and collection
        # entries merge per extension into the extension-to-policy map.
        merged_dict["plugin"] = ConfigManager._merge_plugin_entries(
            defaults.get("plugin", []), col_spec.get("plugin", [])
        )
        
        # Add the metadata and the blob for validation
        merged_dict["collection_id"] = collection_id
        merged_dict["source_blob"] = source_blob
        
        # Validation via Pydantic model
        # This will raise a ValidationError if tool_prefix is missing
        return CollectionConfig.model_validate(merged_dict)
