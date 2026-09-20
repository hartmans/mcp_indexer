import pytest
from pathlib import Path

from mcp_indexer.config import ConfigManager


def test_indexing_mode_defaults_and_fragment_override(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text('[collections.wiki]\ntool_prefix = "wiki"\n')
    assert ConfigManager(str(base)).get_collection_config("wiki").indexing_mode is None
    defaults = tmp_path / "defaults.toml"
    defaults.write_text('[defaults]\nindexing_mode = "indexed"\n')
    override = tmp_path / "full.toml"
    override.write_text('[collections.wiki]\nindexing_mode = "full"\n')
    assert ConfigManager([str(base), str(defaults)]).get_collection_config("wiki").indexing_mode == "indexed"
    assert ConfigManager([str(base), str(defaults), str(override)]).get_collection_config("wiki").indexing_mode == "full"


def test_indexing_mode_rejects_unknown_value(tmp_path):
    from pydantic import ValidationError
    config = tmp_path / "invalid.toml"
    config.write_text('[collections.wiki]\ntool_prefix = "wiki"\nindexing_mode = "sometimes"\n')
    with pytest.raises(ValidationError):
        ConfigManager(str(config)).get_collection_config("wiki")


def test_batch_concurrency_defaults(tmp_path):
    conf_file = Path(tmp_path) / "empty.toml"
    conf_file.write_text("")

    server = ConfigManager(str(conf_file)).get_server_config()

    assert server.llm_max_batch_size == 8
    assert server.embedding_max_batch_size == 64


def test_rerank_config_defaults_to_qwen(tmp_path):
    conf_file = tmp_path / "empty.toml"
    conf_file.write_text("")

    config = ConfigManager(str(conf_file)).get_rerank_config()

    assert config.plugin == "qwen"
    assert config.command_line is None
    assert config.url == "http://127.0.0.1:8080/v1"


def test_llama_cpp_rerank_config(tmp_path):
    conf_file = tmp_path / "llama.toml"
    conf_file.write_text(
        '[rerank]\nplugin = "llama_cpp"\n'
        'model = "Qwen/Qwen3-Reranker-0.6B"\n'
        'command_line = ["llama-server", "-m", "/models/reranker.gguf", '
        '"--embedding", "--rerank"]\n'
    )

    config = ConfigManager(str(conf_file)).get_rerank_config()

    assert config.plugin == "llama_cpp"
    assert config.command_line[0] == "llama-server"
    assert config.model == "Qwen/Qwen3-Reranker-0.6B"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("yes", True),
        ("no", False),
    ],
)
def test_collection_rerank_boolean_values(configured, expected):
    from mcp_indexer.config import CollectionConfig

    config = CollectionConfig.model_validate(
        {
            "collection_id": "docs",
            "tool_prefix": "docs",
            "rerank": configured,
        }
    )

    assert config.rerank is expected


class TestDeepMerge:
    def test_deep_merge_dicts(self):
        base = {"a": 1, "b": {"x": 10, "y": 20}, "c": [1, 2]}
        override = {"b": {"y": 99, "z": 30}, "d": "hello"}
        result = ConfigManager._deep_merge(base, override)
        assert result == {
            "a": 1,
            "b": {"x": 10, "y": 99, "z": 30},
            "c": [1, 2],
            "d": "hello",
        }

    def test_deep_merge_replaces_non_dict_value(self):
        base = {"a": {"nested": 42}}
        override = {"a": "string"}
        result = ConfigManager._deep_merge(base, override)
        assert result == {"a": "string"}

    def test_deep_merge_base_is_empty(self):
        result = ConfigManager._deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_deep_merge_override_is_empty(self):
        base = {"a": 1}
        result = ConfigManager._deep_merge(base, {})
        assert result == {"a": 1}

    def test_deep_merge_nested_dicts(self):
        base = {"a": {"b": {"c": 1}}}
        override = {"a": {"b": {"d": 2}}}
        result = ConfigManager._deep_merge(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 2}}}

    def test_deep_merge_preserves_top_level_base_keys(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = ConfigManager._deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}


class TestConfigMultiFile:
    def test_single_path_string(self, tmp_path):
        conf_file = Path(tmp_path) / "base.toml"
        conf_file.write_text('[server]\ndb_uri = "postgresql://localhost/db"\n')
        cm = ConfigManager(str(conf_file))
        server = cm.get_server_config()
        assert server.db_uri == "postgresql://localhost/db"

    def test_single_path_list(self, tmp_path):
        conf_file = Path(tmp_path) / "base.toml"
        conf_file.write_text('[server]\ndb_uri = "postgresql://localhost/db"\n')
        cm = ConfigManager([str(conf_file)])
        server = cm.get_server_config()
        assert server.db_uri == "postgresql://localhost/db"

    def test_multiple_paths_merged(self, tmp_path):
        base = Path(tmp_path) / "base.toml"
        base.write_text(
            '[server]\ndb_uri = "postgresql://localhost/base"\n'
            'llm.model = "gpt-4o-mini"\n'
            "[defaults]\ndoc_summary_prompt = \"default prompt\"\n"
        )

        override = Path(tmp_path) / "local.toml"
        local_content = (
            '[server.llm]\nprovider = "openai"\n'
            '[server]\ndb_uri = "postgresql://localhost/local"\n'
            "[collections.docs_collection]\n"
            "tool_prefix = \"docs\"\n"
            'source_config = {type = "text"}\n'
        )
        override.write_text(local_content)

        cm = ConfigManager([str(base), str(override)])
        server = cm.get_server_config()
        # Leaf-level overwrite wins
        assert server.db_uri == "postgresql://localhost/local"
        assert server.llm['provider'] == "openai"
        # Nested dict values merged - llm.model survives from base (never overwritten)
        assert server.llm['model'] == "gpt-4o-mini"

    def test_list_collections_across_files(self, tmp_path):
        base = Path(tmp_path) / "base.toml"
        base.write_text(
            "[collections.base_col]\n"
            "tool_prefix = \"base\"\n"
            'source_config = {type = "text"}\n'
        )

        override = Path(tmp_path) / "local.toml"
        override.write_text(
            "[collections.local_col]\n"
            "tool_prefix = \"local\"\n"
            'source_config = {type = "text"}\n'
        )

        cm = ConfigManager([str(base), str(override)])
        cols = cm.list_collections()
        assert "base_col" in cols
        assert "local_col" in cols

    def test_collection_config_overrides_defaults(self, tmp_path):
        base = Path(tmp_path) / "base.toml"
        base.write_text(
            "[server]\ndb_uri = \"postgresql://localhost/base\"\n"
            '[defaults]\ndoc_summary_prompt = "default prompt"\n'
            "\n[collections.docs_collection]\n"
            'tool_prefix = "docs"\nsource_config = {type = "text"}\n'
        )

        override = Path(tmp_path) / "local.toml"
        # Proper TOML syntax for nested key: declare parent table first, then child key
        override.write_text(
            "[defaults]\ndoc_summary_prompt = \"custom prompt\"\n"
            "[server.llm]\nprovider = \"anthropic\"\n"
        )

        cm = ConfigManager([str(base), str(override)])
        col = cm.get_collection_config("docs_collection")
        assert col.doc_summary_prompt == "custom prompt"

    def test_later_path_wins_on_conflict(self, tmp_path):
        first = Path(tmp_path) / "first.toml"
        first.write_text('[server]\ndb_uri = "postgresql://first"\n')

        second = Path(tmp_path) / "second.toml"
        second.write_text('[server]\ndb_uri = "postgresql://second"\n')

        cm = ConfigManager([str(first), str(second)])
        server = cm.get_server_config()
        assert server.db_uri == "postgresql://second"
        assert len(cm.list_collections()) == 0
