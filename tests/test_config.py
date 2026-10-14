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


def test_search_threshold_defaults(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[collections.wiki]\ntool_prefix = "wiki"\n')
    config = ConfigManager(str(config_file)).get_collection_config("wiki")
    assert config.min_cosine_distance == 0.65
    assert config.min_rerank_score == 0.5


def test_search_thresholds_overridable_from_defaults_and_collection(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text('[collections.wiki]\ntool_prefix = "wiki"\n')
    defaults = tmp_path / "defaults.toml"
    defaults.write_text(
        '[defaults]\n'
        'min_cosine_distance = 0.8\n'
        'min_rerank_score = 0.4\n'
    )
    assert ConfigManager([str(base), str(defaults)]).get_collection_config(
        "wiki",
    ).min_cosine_distance == 0.8

    override = tmp_path / "override.toml"
    override.write_text(
        '[collections.wiki]\n'
        'tool_prefix = "wiki"\n'
        'min_rerank_score = 0.9\n'
    )
    config = ConfigManager(
        [str(base), str(defaults), str(override)],
    ).get_collection_config("wiki")
    assert config.min_cosine_distance == 0.8
    assert config.min_rerank_score == 0.9


def test_plugin_entries_default_to_empty(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text('[collections.docs]\ntool_prefix = "docs"\n')
    config = ConfigManager(str(config_file)).get_collection_config("docs")
    assert config.plugin == {}


def test_plugin_entries_from_defaults_and_collection_supplement(tmp_path):
    defaults = tmp_path / "defaults.toml"
    defaults.write_text(
        '[defaults]\n'
        '[[defaults.plugin]]\n'
        'handler = "text"\n'
        'extensions = ["rst"]\n'
        'semantic_boundary_regexps = ["^=+"]\n'
    )
    base = tmp_path / "base.toml"
    base.write_text(
        '[collections.docs]\n'
        'tool_prefix = "docs"\n'
        '[[collections.docs.plugin]]\n'
        'handler = "text"\n'
        'extensions = ["md"]\n'
        'semantic_boundary_regexps = ["^#+"]\n'
    )
    config = ConfigManager([str(defaults), str(base)]).get_collection_config("docs")
    assert set(config.plugin) == {".md", ".rst"}
    # The defaults policy survives alongside the collection's own policy.
    assert config.plugin[".rst"] == {"handler": "text", "semantic_boundary_regexps": ["^=+"]}
    assert config.plugin[".md"] == {"handler": "text", "semantic_boundary_regexps": ["^#+"]}


def test_collection_plugin_entry_overrides_default_per_extension(tmp_path):
    defaults = tmp_path / "defaults.toml"
    defaults.write_text(
        '[defaults]\n'
        '[[defaults.plugin]]\n'
        'handler = "text"\n'
        'extensions = ["md"]\n'
        'semantic_boundary_regexps = ["^#"]\n'
        'default_marker = "default"\n'
    )
    base = tmp_path / "base.toml"
    base.write_text(
        '[collections.docs]\n'
        'tool_prefix = "docs"\n'
        '[[collections.docs.plugin]]\n'
        'handler = "text"\n'
        'extensions = ["md"]\n'
        'semantic_boundary_regexps = ["^##"]\n'
        'collection_marker = "collection"\n'
    )
    config = ConfigManager([str(defaults), str(base)]).get_collection_config("docs")
    # One effective policy for the extension: the collection entry wholly
    # replaces the defaults entry, with no key-level merge.
    policy = config.plugin.get(".md")
    assert policy is not None
    assert policy["semantic_boundary_regexps"] == ["^##"]
    assert policy["collection_marker"] == "collection"
    assert "default_marker" not in policy


def test_plugin_entry_extensions_case_insensitive_and_dot_optional(tmp_path):
    defaults = tmp_path / "defaults.toml"
    defaults.write_text(
        '[defaults]\n'
        '[[defaults.plugin]]\n'
        'handler = "text"\n'
        'extensions = ["MD"]\n'
    )
    base = tmp_path / "base.toml"
    base.write_text(
        '[collections.docs]\n'
        'tool_prefix = "docs"\n'
        '[[collections.docs.plugin]]\n'
        'handler = "text"\n'
        'extensions = [".md"]\n'
    )
    config = ConfigManager([str(defaults), str(base)]).get_collection_config("docs")
    # "MD" and ".md" are the same extension: one effective policy, keyed
    # canonically.
    assert set(config.plugin) == {".md"}
    assert config.plugin[".md"]["handler"] == "text"


def test_plugin_duplicate_extensions_within_one_list_last_wins(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text(
        '[collections.docs]\n'
        'tool_prefix = "docs"\n'
        '[[collections.docs.plugin]]\n'
        'handler = "text"\n'
        'extensions = ["md"]\n'
        'marker = "first"\n'
        '\n'
        '[[collections.docs.plugin]]\n'
        'handler = "text"\n'
        'extensions = ["md"]\n'
        'marker = "second"\n'
    )
    config = ConfigManager(str(base)).get_collection_config("docs")
    assert set(config.plugin) == {".md"}
    assert config.plugin[".md"]["marker"] == "second"


def test_plugin_entry_requires_handler(tmp_path):
    from pydantic import ValidationError
    config_file = tmp_path / "invalid.toml"
    config_file.write_text(
        '[collections.docs]\n'
        'tool_prefix = "docs"\n'
        '[[collections.docs.plugin]]\n'
        'extensions = ["md"]\n'
    )
    with pytest.raises(ValidationError, match="handler"):
        ConfigManager(str(config_file)).get_collection_config("docs")


def test_plugin_entry_requires_nonempty_extensions(tmp_path):
    from pydantic import ValidationError
    missing = tmp_path / "missing_extensions.toml"
    missing.write_text(
        '[collections.docs]\n'
        'tool_prefix = "docs"\n'
        '[[collections.docs.plugin]]\n'
        'handler = "text"\n'
    )
    with pytest.raises(ValidationError, match="extensions"):
        ConfigManager(str(missing)).get_collection_config("docs")

    empty_list = tmp_path / "empty_extensions.toml"
    empty_list.write_text(
        '[collections.docs]\n'
        'tool_prefix = "docs"\n'
        '[[collections.docs.plugin]]\n'
        'handler = "text"\n'
        'extensions = []\n'
    )
    with pytest.raises(ValidationError, match="extensions"):
        ConfigManager(str(empty_list)).get_collection_config("docs")

    blank = tmp_path / "blank_extension.toml"
    blank.write_text(
        '[collections.docs]\n'
        'tool_prefix = "docs"\n'
        '[[collections.docs.plugin]]\n'
        'handler = "text"\n'
        'extensions = [""]\n'
    )
    with pytest.raises(ValidationError, match="non-empty"):
        ConfigManager(str(blank)).get_collection_config("docs")


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
