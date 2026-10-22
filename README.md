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

### Server settings

`[server]` holds the following settings. Sizes of text are characters, not bytes.

| Setting | Default | Meaning |
| --- | --- | --- |
| `db_uri` | `postgresql://postgres@:5432/indexer?host=/var/run/postgresql` | PostgreSQL connection URI; requires pgvector |
| `log_level` | `"INFO"` | Server configuration log level |
| `llm` | `{}` | Arguments passed to LangChain chat-model initialization |
| `embedding` | `{}` | Arguments passed to LangChain embedding initialization |
| `min_size`, `max_size` | `500`, `5000` | Target minimum and hard maximum embedding chunk sizes |
| `max_to_summarize` | `65536` | Text budget used to plan summary spans |
| `llm_batch_size`, `embedding_batch_size` | `8`, `10` | Model batch sizes |
| `llm_max_batch_size`, `embedding_max_batch_size` | `8`, `64` | Admission limits for concurrent calls to the batchers |
| `llm_request_timeout`, `embedding_request_timeout` | `400.0`, `400.0` | Model request timeouts in seconds |
| `llm_timeout_retries`, `embedding_timeout_retries` | `20`, `20` | Timeout retry budgets |
| `complex_documents` | CPU core count | Concurrent complex document conversions |

Supply model/provider settings and any provider connection arguments in
`[server.llm]` and `[server.embedding]`. The embedding adapter also accepts
`model_provider` as an alias for `provider`. The current database model uses
768-dimensional vectors; select a compatible embedding model. Changing embedding
models requires rebuilding compatible indexed vectors, not just changing the
connection settings.

### Collection settings

These settings may appear in `[defaults]` or `[collections.<id>]`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `doc_summary_prompt` | `"Summarize the following document concisely..."` | Document summary instruction |
| `chunk_summary_prompt` | `""` | Chunk summary instruction |
| `rerank` | `false` | Enable the shared reranker |
| `indexing_mode` | Source default | `"full"` enumerates new documents; `"indexed"` indexes on demand |
| `min_cosine_distance` | `0.65` | Maximum accepted vector cosine distance, despite the name |
| `min_rerank_score` | `0.5` | Minimum accepted reranker score |
| `plugin` | No overrides | Extension-policy entries described above |

Each collection requires `tool_prefix` and a `source_config` containing its source
`type`. Its optional `description` defaults to `"Document collection"`. The
collection ID comes from its TOML table name. An omitted indexing mode uses the
source default: `full` for ordinary file collections and `indexed` for Wikipedia.

Configuration fragments are deep-merged first, with later leaf values replacing
earlier ones; lists replace earlier lists at this stage. The resolved collection
then inherits defaults, including the per-extension policy merge described above.
Use repeated `--config` arguments to apply deployment-specific overrides.

### Sources and formats

A file source uses `type = "file"`. Its required `directory` is the corpus root.
Optional `include` and `exclude` are lists of glob patterns, both defaulting to
empty lists. `encoding` defaults to `"utf-8"`; `summary_length` defaults to `65536`
and controls the source text summary size cap.

Markdown (`.md`, `.markdown`), HTML (`.html`, `.htm`), and RST (`.rst`) have built-in
text policies. Textual markup is preserved. Markdown and HTML use major headings
as semantic boundaries; RST recognizes `=` and `-` title adornments. Other text
uses separator lines, with paragraph boundaries as embedding candidates. Unknown
extensions and extensionless files fall back to text interpretation.

Install the optional converter with `pip install '.[docling]'` from this checkout
for `.pdf`, `.docx`, `.odt`, `.ods`, `.ppt`, `.pptx`, `.doc`, and `.rtf`. These files
produce retained Markdown text and extracted figure subdocuments. PDF OCR uses
RapidOCR with the PyTorch backend; model provisioning may be needed on first use.
Figure descriptions use the configured image-capable chat model. Missing Docling
raises an informative error when a conversion plugin is instantiated; it does not
silently interpret binary files as text. Figure IDs are best-effort stable and may
change after rerendering. Fetching a document lists its direct subdocuments;
binary image delivery through MCP is not yet defined.

A Wikipedia source uses `type = "wikipedia"` and additionally requires
`xapian_directory`. Install the `wikipedia` extra and provide an extracted
MediaWiki corpus with its existing Xapian index. Articles retain raw MediaWiki,
including template names and arguments. The source reads Xapian directly and
supports native search without embedding the whole corpus first. Updating a dump
must replace changed files with changed modification times, remove absent files,
and update Xapian. Historical extraction scripts that skip existing entries do
not perform this refresh.

### Reranking

The global `[rerank]` section selects one adapter shared by collections with
`rerank = true`. `plugin = "qwen"` is the default in-process Transformers adapter;
install the `rerank` extra for it. `plugin = "llama_cpp"` uses a server with a
vLLM-compatible rerank API.

| Setting | Default | Meaning |
| --- | --- | --- |
| `plugin` | `"qwen"` | `"qwen"` or `"llama_cpp"` |
| `url` | `"http://127.0.0.1:8080/v1"` | External server base URL, including its API path |
| `model` | Unset | Optional model name included in external rerank requests |
| `command_line` | Unset | Argument list for a managed local rerank server |
| `startup_timeout` | `60.0` | Readiness timeout in seconds |
| `request_timeout` | `400.0` | Request timeout in seconds |

For an existing server, set `url`; set `model` if the service requires explicit
model selection. Alternatively, `command_line` launches a local server on a
temporary Unix-domain socket. Include model and reranking flags in the command;
the indexer appends its socket `--host` argument. Readiness uses the base URL's
`/models` endpoint and scoring uses `/rerank`, preserving the configured API path.

Vector distance filtering happens before candidate limits. Rerank score filtering
happens before the document page limit, so rejected hits do not consume result
slots. `--debug-search` in the interactive client logs native scores, vector cosine
distances, and successful reranker scores; normal logging omits these diagnostics.

## Running and maintaining the index

Install the project into your environment with `pip install .` and provision
PostgreSQL with pgvector and the configured model services. The following commands
use the repository's Python environment:

```sh
~/venv/bin/python -m mcp_indexer.indexer --config config.toml
~/venv/bin/python -m mcp_indexer.server --config config.toml
~/venv/bin/python -m mcp_indexer.search_client --config config.toml docs
```

The server defaults to stdio. Add `--port PORT` for streamable HTTP and optionally
`--host HOST` (default `localhost`). The indexing CLI runs indexing, chunk
summarization, and document summarization when no operation flags are supplied.
Use `--index`, `--summarize-chunks`, or `--summarize-documents` to select operations.
`--debug` writes per-document indexing statistics.

Maintenance is opt-in with `--maintenance`, in either indexing mode. It refreshes
changed indexed documents and removes confirmed missing ones; source access
failures preserve stored records. Search reuses existing index rows without
checking for changes. Handle caching can also retain stale source content, so
use a fresh process when a maintenance pass must observe newly replaced files.

Existing databases require the idempotent, resumable schema migration before
using the current storage model:

```sh
~/venv/bin/python scripts/migrate_schema.py --config config.toml --dry-run
~/venv/bin/python scripts/migrate_schema.py --config config.toml
```

The migration includes semantic chunks, foreign-key cleanup, subdocuments,
metadata, and pending child work; the retained-text column already exists in the
pre-migration schema. It can resume a partially migrated
database; cursor sessions themselves require no migration or persistent storage.

## Search and continuation

MCP provides collection search and companion document/chunk fetch tools. External
IDs have the form `collection_id:local_document_id`; subdocuments add colon-separated
components and chunk references append `?c=...`. Filename colons are encoded as
`%3A`. Document responses expose nonconflicting source metadata and direct children.

Start a search with `query` and a positive `limit` (default `5`, counting distinct
documents). Structured native sources additionally expose `rerank_query` for a
natural-language query used by vector discovery and reranking. An omitted or blank
rerank query uses the original query. Search responses use this envelope:

```json
{
  "results": [],
  "resume_cursor": "opaque-token",
  "resume_cursor_usage": "Call this tool again and pass in the cursor in the cursor argument to resume.",
  "warnings": []
}
```

Pass `resume_cursor` as the next call's `cursor`; `query` can then be omitted.
Saved queries remain in effect if conflicting queries are supplied, with a
warning. The page limit may change. Short or empty pages can still have a cursor;
continue until `resume_cursor` is null, when `resume_cursor_usage` reads
`End of documents reached`. Later pages can contain higher-scoring documents.
Approximate vector search does not guarantee that every relevant document is found.

Retrying the most recently consumed cursor replays the same page and next cursor.
Successfully advancing the next cursor invalidates the older replay token. Final
pages remain replayable through their input cursor. A lost initial response cannot
be recovered by token. Sessions expire after 60 minutes of inactivity; restarting
or replacing the source invalidates them. Multiple serving processes need sticky
routing. Unknown, expired, or wrong-collection tokens are errors, not end-of-results.

Warnings report partial failures. Preparation and document assembly each permit
one later retry before dropping a failed item. Discovery or reranker outages can
produce retryable errors; retry with the same cursor. Successful discovery and
scoring are retained during continuation failures.

The interactive client accepts `:more` to continue the latest search and retains
its cursor after errors. New query input starts a new session; `query => rerank
query` supplies a separate natural-language query. Tokens cannot survive client
process exit. `--document-candidate-limit` and `--chunk-candidate-limit` control
per-batch discovery sizes, defaulting to four times the initial page limit each.
Zero disables a hit type; both cannot be zero. Python callers have the equivalent
keyword arguments. These sizes, thresholds, and reranker are fixed for a session.

## Development documentation and verification

[architecture.md](architecture.md) describes component responsibilities and the
contracts to preserve when extending them. Method docstrings describe detailed
API behavior. Run fast tests with `~/venv/bin/python -m pytest -m "not long_cpu"`.
The `long_cpu` Docling conversions exercise real OCR and take longer; they require
the optional converter and its models. Database tests require
`MCP_INDEXER_TEST_DB_URL` pointing to a dedicated test database, whose tables tests
may create and drop. Tests use pytest, monkeypatch, and handwritten fakes.
