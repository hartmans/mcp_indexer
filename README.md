# mcp_indexer

A server that indexes documents in collections and provides a set of
MCP tools for each collection to search and fetch documents.

## Configuration

Configuration is a TOML file (or several, deep-merged in the order
given) with three layers:

* `[server]` — global server settings (database URI, models, batch
  sizes, timeouts).
* `[defaults]` — defaults for collection-level settings (summary
  prompts, reranking, indexing mode, search thresholds).
* `[collections.<id>]` — per-collection settings, overriding the
  defaults. Each collection requires a `tool_prefix` and a
  `source_config` blob that its DocumentSource plugin validates.

### Plugin extension policies

FileSources resolve each file's extension to an interpretation policy.
The base map is code-defined; configuration layers plugin entries on
top. `[[defaults.plugin]]` applies to every collection, and
`[[collections.<id>.plugin]]` supplements the defaults and overrides
them per collection.

Each entry carries:

* `handler` — the document plugin name (required).
* `extensions` — the extensions the entry covers (required;
  case-insensitive, with or without the leading dot: `MD` and `.md`
  are the same extension).
* Any policy keys the handler uses, which pass through to the plugin
  verbatim.

Entries merge into the collection's extension-to-policy map: a
collection entry for an extension replaces the defaults entry for that
extension wholly, and later entries within one list win over earlier
ones. Extensions absent from the map fall back to the source's default
text policy. Non-file sources ignore the plugin entries.

Example:

```toml
[defaults]
doc_summary_prompt = "Summarize the document"

[[defaults.plugin]]
handler = "text"
extensions = ["rst"]
semantic_boundary_regexps = ["^=+"]

[collections.docs]
tool_prefix = "docs"
source_config = { type = "file", directory = "/home/user/documents" }

[[collections.docs.plugin]]
handler = "text"
extensions = ["md", "markdown"]
semantic_boundary_regexps = ["^## "]
```
