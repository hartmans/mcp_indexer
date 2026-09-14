# Search architecture: native sources and indexed documents

This plan records the architectural direction agreed for rearranging search.
It is for review before defining implementation tasks. The implementation
decisions below establish module responsibilities; exact signatures and return
types still need to be worked out within those decisions.

## Motivation

Some collections are too large to embed in full ahead of time, but already have
useful native search. Examples include hundreds of gigabytes of email searched
through notmuch and a Wikipedia dump searched through a native text index.

Native search discovers documents across the full corpus. Documents selected
for return become indexed in our database, allowing subsequent vector searches
to rediscover previously interesting material.

Native hits are not assumed to include useful excerpts. Notmuch can provide
thread IDs, subjects, participants, and matching messages. The current Wikipedia
search provides article titles. Preparing useful results therefore requires
content retrieval and, where necessary, summarization and indexing work.

## Search stages

Separate the current search into candidate discovery, optional reranking,
selection, and result preparation/assembly.

1. Discover candidates through the applicable search mechanisms. A SQL
   search provides document-level and chunk-level vector hits. Doing the sql vector search is always the responsibility of source-independent code. Sources can also
   provide candidates through their native search. At this stage we return document hits separately from chunk hits, both for native search and for vector search. It is okay if a source can only implement one of chunk or document hits.
2. When reranking is enabled, resolve the text needed to score the combined
   candidates and rerank them before final selection. As a consequence, getting document level summary texts  if not provided by native search is a reranker concern. The vector search code can presumably optimistically return summaries where present (possibly using the same mechanism the native search would use to inject summaries) since it already has access to those database rows.
3. Select results using reranker ordering, or preserve the initial search
   rankings under the no-reranker combination policy below.
4. Prepare selected native results for persistence, generate the summaries
   needed for useful responses, and assemble documents with relevant chunks.

During search, we do not reindex already persisted documents. Maintenance updates
changed documents as described in the native-source plan below. Filling missing
summaries and their associated document embeddings is still permitted.
If a selected native document does not exist in the database, assembly explicitly
indexes it. Reranking may independently need to index and summarize a candidate
earlier to prepare its scoring text, even if that candidate is not selected.
Keep these responsibilities separate: assembly only operates on selected hits,
while reranking preparation may be optimized independently in the future.

Candidate limits and the final result limit are distinct. Candidate defaults
belong in the search implementation, not in the MCP wrapper. The current
candidate multiplier is tuning, not an architectural requirement.

Source-independent vector retrieval and optional source-native retrieval feed
the search orchestration in search.py. External
reranker scores do not need to be written back into SQL for ranking or grouping;
those operations can run in Python over the bounded candidate set.

## Native search and vector search coexist

A native-search collection can use both native search over its full corpus and
vector search over its indexed subset. This is not an exclusive choice made by
source type.

Previously indexed documents have an additional route into the results. Native
search continues to provide coverage of documents that have never been indexed.
Newly indexed documents participate in later vector searches; search need not
rerun vector retrieval after indexing its selected native hits.

Native scores and vector scores are not assumed to share a numerical scale.

### Without reranking

Keep the separately limited document and chunk hits from both vector and native
search. Concatenate the lists, vector results first, preserving each list's
initial ranking. Return each document only once and merge its relevant chunks.
There is no additional common-score cutoff: with each of the four lists limited
to the requested result limit, the combined results can contain up to four times
that limit before deduplication. This is acceptable; the earlier twice-the-limit
ceiling is not a requirement. Candidate overfetching for reranking is separate
from these no-reranker result limits.

The architecture does not require score normalization or rank fusion between
the initial search lists.

### With reranking

Combine candidates from the search mechanisms, resolve their reranker text,
and use reranking to select the final results. Initial search scores govern
candidate discovery, rather than being numerically mixed with reranker scores.

Preserve document identity, merge relevant chunks, and avoid returning duplicate
documents. The exact interaction between the final limit, hit selection, and
document grouping remains an implementation detail to specify.
## Candidate information and reranker text

Conceptually, search produces document hits and chunk hits:

- A document hit identifies a document and carries its initial search score.
  It can optionally supply a document summary.
- A chunk hit identifies a document and a embedding chunk and carries its initial
  search score. The source must make the relevant original text retrievable. Long term there is support for storing chunk text in the database if a chunk undergoes transformation, BUt that is not used today.



### Document hits

Resolve a document hit's summary in this order:

1. Use a summary supplied by the source on the hit.
2. Otherwise call summarize_documents. Make sure summarize_documents does not update summaries if they already exist. Use the resulting summary, retrieved in a new session.

This fallback can be expensive, but returning a document-level hit without a
summary accepts that cost. A source can avoid it by supplying a suitable summary
or returning meaningful chunk hits and not document hits.

A supplied summary intended for reuse as the database document summary must
describe the document, not just its relevance to the current query.
Hit summaries serve reranking. Persistence uses the existing source-summary
interface through document summarization; sources providing summaries are
expected to provide them efficiently there too. No separate hit-summary
persistence path or summary_authoritative flag is needed now.

Ordinary database documents without summaries currently have zero embedding
vectors, so they are not expected to be useful document-level vector hits. We
do  proactively summarize all unsummarized database documents for search, but this task may take months to complete.
The hot-path fallback applies when a document hit actually needs a summary.

### Chunk hits

Reranking uses original embedding-chunk text for current database chunk hits,
retrieved through the source. That text is currently not stored in the database;
future implementations may obtain it there.

Use the existing source interfaces and chunk metadata to retrieve and identify
chunks. Native chunks need not have been persisted before candidate scoring.
For notmuch, matching messages are accessible semantic chunks; embedding chunks
within those messages follow the existing semantic/embedding chunk architecture.
This work does not redefine that boundary or introduce a new chunking scheme.

A summary of matching messages must be identified as such, rather than treated
as a summary of an entire thread. The same distinction applies to selected
passages within other documents.
This follows from the existing distinction between chunk summaries and document
summaries, rather than requiring a new summary category.

## Persistence and complete indexing

Selected native documents are fully chunked and embedded when they are
interesting enough to return to a client. Preserve the current invariant:
if any indexed chunk exists for a document, all its indexed chunks exist.
Reranking preparation may also index candidates before generating missing
summary fallbacks. That earlier work is distinct from assembly's obligation to
prepare selected documents and must not expand assembly to all candidates.

Embedding is currently non-null, so storing chunks requires embedding them.
Complete chunk storage does not require summarizing every chunk. Summarization
can remain selective, provided returned results have useful descriptive content
and summaries produced for reuse are persisted. This may require adjustments to the indexer code to support summarizing a single semantic chunk. Summaries belong to spans that map onto semantic chunks.


We are not adopting partial indexed-chunk storage or relying on a lightweight
"known document" record as sufficient preparation for a client result. Titles
and identities alone are too thin for the sources under discussion.

Making embeddings nullable later is a reasonable option if separating chunk
storage from embedding becomes necessary. It is not required for this design.

Document boundaries matter to the cost of complete indexing: a Wikipedia article
or email thread is a candidate unit; an entire corpus dump is not. Handling
unusually large documents and indexing failures requires implementation policy,
without silently weakening the complete-chunk invariant.

## Summary freshness and session lifetime

The current rerank path loads candidate ORM documents into a session before
calling the summarizer. Summarization updates through another session, leaving
the first session's identity map with stale, potentially empty summaries.

Result assembly must observe completed summarization. Use fresh hydration after
summary writes, or an explicit refresh strategy that guarantees current data.
Re-querying in a session that already holds those objects is not sufficient by
itself. Candidate discovery and content preparation should not accidentally
determine the lifetime or freshness of response ORM objects.

## Reranker lifecycle

Context construction must not download or load the reranker. Indexing-only
commands do not need it.

Initialize the reranker on the first search that requires it. Enabled sources
must share the same reranker instance because its semaphore coordinates scoring
concurrency across collections. Concurrent first searches must coordinate setup
so the model is loaded once. Explicit setup remains responsible for downloading
and allocating model resources; its invocation moves from context construction
to search-time use.

# Implementation Decisions.

- Move the search code out of the indexer into search.py (new file). Probably search does need an indexer, but search is getting complicated enough for its own module.

- vector_search_hits takes a collection, query,  document_limit and chunk_limit, and returns a list of DocumentHit|ChunkHit. Some of the database complexity (combining chunk and document hits) disappears.
- rerank takes an indexer, rerank_query, and hit_limit and returns reranked hits. By this point DocumentHits at least on the rerank path probably can be guaranteed to have summaries in their DocumentHit, either filled in by the reranker or by the native source.

- assemble_hits takes hits and transforms into (Document, list[Chunk]) guaranteeing that document.summary and chunk level summaries are non-null.
  It operates only on selected hits, reuses existing or supplied summaries, and
  indexes missing documents as necessary. Reranking preparation remains a
  separate responsibility even where both stages need indexing or summarization.

- search combines the above and takes a separate query and rerank_query. The
  native-source plan below exposes the latter through MCP when the source has
  structured search syntax (`separate_rerank`).

- Update architecture.md as an implementation task: document the new search
  stages, source-independent vector search alongside native search, selection
  and persistence behavior, summary freshness, and reranker lifecycle. Clean up
  outdated or unclear descriptions discovered during this work, including the
  description of the existing semantic/embedding chunk relationships. Preserve
  those relationships; this is documentation alignment, not a boundary change.
  Revision of the relevant existing architecture.md sections is authorized for
  this task.

## out of scope

- Tests for native sources and native_search implementations. We'll do that when we implement such a source.

## Remaining implementation decisions

- Exact signatures, hit representations, and ordering of document versus chunk
  lists within each mechanism, following the architecture above.
- Concurrency is handled at other layers; do not introduce search-layer
  coordination for discovery or indexing.
- Failures are logged and search returns whatever useful results it can prepare.

The architectural review issues are resolved. The remaining decisions should be
discussed as implementation tasks; they do not reopen the agreed search stages,
no-reranker combination policy, or semantic/embedding chunk boundary.

---

# Native sources and Wikipedia: proposed implementation

This section incorporates the review of the first draft. It preserves the search
stages above. Maintenance now refreshes changed documents; search preparation
still reuses persisted documents. The separate natural-language rerank query is
now exposed by MCP tools for sources that request it. These clarify the earlier
search-only freshness rule and supersede the earlier deferral of MCP exposure.
Implementation is now underway as authorized. See `implementation-notes.md` for
implemented behavior, completed checks, and the remaining deployment/testing steps.

## Collection policy and identifiers

Add `indexing_mode: Literal["full", "indexed"] | None = None` to
`CollectionInfraConfig`. `DocumentSource.default_indexing_mode` defaults to
`"full"`; Wikipedia overrides it with `"indexed"`. A source `indexing_mode`
property resolves explicit collection configuration, then `[defaults]`, then
the source class default. Configuration fragments continue to override settings.

| Mode | Existing database documents | Documents not yet indexed |
| --- | --- | --- |
| `full` | Refresh/remove when maintenance is requested | Discover through `get_documents` and index |
| `indexed` | Refresh/remove when maintenance is requested | Index when search selects them or a caller requests them |

Full coverage does not mean re-embedding unchanged documents. Both modes retain
native search over the source and vector search over persisted documents.
An `indexed` collection can still implement exhaustive enumeration, including
Wikipedia. Sources that cannot enumerate can raise `NotImplementedError` from
`get_documents`; no enumeration capability registry is needed.

Use an additional configuration fragment for overrides, passed through the
existing repeatable `-c` option. For example, a fragment containing
`[collections.wikipedia]` and `indexing_mode = "full"` requests a full run.
Do not add indexing-mode, removal-policy, retry, or collection-selection CLI
flags for this work. Keep the existing operation flags and add the independent
maintenance flag described below. No separate removal-policy setting is proposed.

Document IDs have two intentional forms:

- Collection-local IDs are used by `DocumentSource`, `DocumentPointer`, search
  hits, and indexer methods taking a source. SQL stores these with a separate
  `collection_id`. For Wikipedia, an example is `Wikipedia.mediawiki`.
- Global references used in tools and outside the system are
  `<collection_id>:<local_document_id>`, for example
  `wikipedia:Wikipedia.mediawiki`. Global chunk references similarly prefix the
  local `Wikipedia.mediawiki?c=0`. The tool boundary adds/removes the collection
  prefix once, using the existing helpers in `server.py`.

There is no source-type prefix in the current global convention. A collection
rename changes `collection_id` in all relevant tables and configuration; local
document IDs, chunk IDs, and source retrieval metadata remain unchanged. External
references thereafter use the new collection prefix. Clarify this convention
in architecture documentation when implementing; it is not an ID migration.

## DocumentSource changes

Keep calling `native_search` unconditionally. Its default implementation returns
`[]`; a native-search capability flag adds no useful information.

Add source defaults and the query-syntax attribute to `DocumentSource`:

```python
default_indexing_mode: Literal["full", "indexed"] = "full"
separate_rerank: bool = False

@property
def indexing_mode(self) -> Literal["full", "indexed"]:
    return self.config.indexing_mode or self.default_indexing_mode
```

This describes query syntax, not whether reranking is enabled. Wikipedia sets it
to true because Xapian accepts structured expressions such as `cat:` and `title:`.
For such collections, expose an optional `rerank_query: str | None = None` in
the MCP search tool, described as the natural-language information need. Other
collections retain their existing tool signature. An omitted rerank query falls
back to `query`. Pass it through the MCP/core wrapper to the existing search API.

Agreed query routing: send `query` to native discovery. Use `rerank_query` when
supplied, otherwise `query`, for both vector discovery and reranking. Treat an
empty natural-language input as omitted. Document and test this change to the
current vector-query behavior.
The natural-language input remains useful when reranking is disabled, since
vector discovery still runs. Keep the existing ranking and combination policies.

Retain the existing methods, correcting the missing `self` in the base
`fetch_document` declaration:

```python
async def native_search(
    self, query: str, *, document_limit: int, chunk_limit: int,
) -> list[SearchHit]: ...

async def get_documents(
    self, last_modified: datetime | None = None,
) -> AsyncGenerator[DocumentPointer, None]: ...

def fetch_document(self, document_id: str) -> DocumentPointer: ...
async def get_document_summary(self, document_id: str) -> str: ...
```

Use `fetch_document(id)` and the returned pointer's existing `last_modified` for
maintenance. Specify that a newly fetched pointer describes the current source
document and its current modification time. It must not return cached old
metadata. File pointers already stat the file when constructed; no separate
batch inspection API or status dataclasses are needed for this source.

Add `DocumentNotFoundError(LookupError)` with `document_id: str` (local) and a
message. It has no additional public methods. Pointer resolution or actual
content/chunk retrieval raises this when the document is missing. Translate
file-not-found errors at the source boundary; leave permission, I/O, and source
availability failures distinct. Missing collection roots raise an ordinary
source/I/O error, not a missing-document error. Do not couple file existence to
whether Xapian is available, or introduce an article sentinel configuration.

An error during a necessary fetch can be handled by search's existing per-result
failure handling or reported by a fetch tool. Search preparation does not check
presence or mtime, and does not delete documents. In particular, a cached summary
does not trigger a source probe. Maintenance is responsible for discovering
removals and updates that no client happens to fetch.

`get_documents(last_modified=...)` still cannot reveal deletions. The initial
maintenance implementation does not use a collection-wide timestamp watermark;
it compares each persisted document against its current pointer. This also avoids
missing updates whose timestamp precedes the last run's completion.

## Indexer: maintain the persisted subset in either mode

Add `maintenance: bool = False` to `index_all` and a `--maintenance` CLI operation
flag, alongside indexing and summarization. `--maintenance` alone only maintains
stored documents; combine flags to request other operations. With no operation
flags, preserve the previous index-and-summarize behavior without maintenance.
Expose `maintain_collection(source) -> None` for maintenance, independent of
`index_collection` discovery. Read effective policy from `source.indexing_mode`.
Maintenance performance on large persisted corpora is a deferred optimization.
Keep indexing and summarization running
concurrently: waiting for an entire indexing pass before summarizing is
unacceptable for collections that take months. Later summary passes can pick up
documents missed during concurrent enumeration.

`maintain_collection` maintains stored documents in either mode;
`index_collection` only performs discovery in full mode. Page through database IDs and stored
modification times in bounded batches; no corpus-sized membership set or new
`CollectionIndexingResult` is required. Use existing statistics/logging.

For each stored document:

1. Fetch a fresh source pointer when ready to process that document.
2. If that raises `DocumentNotFoundError`, delete the corresponding database
   document in a short transaction immediately thereafter. Do not collect a
   missing list for a later pass or repeat the inspection. Other errors are
   logged and leave the row alone.
3. If `pointer.last_modified == stored.last_modified`, reuse the row.
4. Otherwise, schedule complete reindexing using the pointer. Compare for
   inequality, not just a newer timestamp: restoring an older dump must also
   refresh an article.

Full-mode discovery streams `get_documents(None)` and schedules missing IDs;
existing IDs are handled when maintenance is requested. Neither a failed listing nor absence
from a timestamp-filtered listing is evidence of deletion. Indexed mode never
enumerates the unindexed corpus. It therefore updates changed Wikipedia articles
even when only a small subset of the dump has ever been embedded.

When refreshing, prepare all replacement chunks and embeddings before opening
the short write transaction. Replace the old document and dependent data
atomically, recording the new pointer's mtime, clearing its old document summary
and summary embedding, and discarding all old chunk summaries. Reuse the current
complete-document indexing routine for preparation; adjust its persistence step
so an update removes summaries as well as replacing chunks. A failed preparation
leaves the old stored version intact and its old timestamp ensures a later run
still recognizes it as changed. Do not let the failed-document skip list
permanently suppress maintenance refreshes of stored changed documents; clear
that failure on successful refresh.

Summary jobs may overlap replacement. Have summary writes lock the document row,
check the mtime they started from inside their write transaction, and abandon stale work
if the document was replaced or deleted. They must not merge an old ORM document
back into existence. Keep this guard in indexer persistence, without serializing
whole collection passes or adding search-layer coordination.

This provides normal eventual maintenance, not snapshot-isolated search over a
changing filesystem. During extraction and before maintenance completes, stored
summaries/embeddings can still describe old content and old chunk offsets can be
stale. Run maintenance after extraction; callers requiring no mixed results can
pause searches during replacement and refresh. No new collection ID, versioned
collection system, or change-feed protocol is required.

Move the existing search helper's responsibility into the approved public API:

```python
async def ensure_document_indexed(
    self, source: DocumentSource, document_id: str,
    *, retry_failed: bool = False,
) -> None: ...
```

Here `document_id` is collection-local and contains no collection prefix. If
persisted, return without source checks or refresh. Otherwise fetch the pointer
and await full indexing, applying the existing failure handling; the explicit
retry keyword permits a caller to retry a previously failed document. This method
does not summarize. Both reranker preparation and selected-hit assembly call it,
preserving their separate responsibilities. No other new generic indexer class
or public reconciliation/result API is needed.

## ORM constraints and the production migration

Fix ownership in the models rather than implementing a manual child-table
deletion workaround in every indexing path:

- Give `DocumentChunk`'s composite document foreign key `ON DELETE CASCADE`.
- Add the missing composite foreign key from `ChunkSummary` to `Document`, also
  with `ON DELETE CASCADE`, and the corresponding document/summary relationships.
- Align ORM cascades and passive deletion with database ownership. Preserve the
  chunk-to-summary rule that deleting a summary alone clears `summary_span`.
- Make the collection/document foreign-key checks, including chunk-to-summary,
  deferrable so a collection rename can update `collection_id` in each table in
  one transaction with deferred checks. No local ID rewriting is necessary.

Deleting a document then removes its chunks and summaries through the database.
Atomic replacement can delete and reinsert the document with its full prepared
chunk set in one transaction. `FailedDocument` intentionally permits failure
records without a `Document`, so it must not acquire that foreign key. Clear any
failure for a deleted/refreshed document explicitly in the same transaction.

Manual migrations are required for the single production database. Supply the
reviewable SQL to clean up any existing orphan summaries, replace/add foreign
keys and deferrability, and validate the resulting constraints. Updating Python
models or calling `create_all` will not migrate existing tables. Test the manual
migration on a fixture containing the current schema before applying it to
production; no migration framework is proposed.

## Wikipedia source

### Existing format

`/srv/wikipedia/pages/` contains UTF-8 raw MediaWiki articles, one `.mediawiki`
file per extracted title. `redirects/` is separate. The existing Xapian builder
indexes only `pages/`, putting the filename stem in document data, body terms
without a field prefix, title terms under `T`, categories under `C`, and unique
identity under `Q` (hashed for long stems). It uses English stemming. It stores
no article text or revision timestamp in document data/value slots.

The extraction filename mapping is lossy (replacement of `/`, `*`, and `?`,
leading-dot removal and long-name truncation). Use the actual filename as local
identity; do not try to recover original titles or upstream page IDs. Use the
stem as the display title. Keep redirects outside the indexed article collection
for now.

The inspected scripts skip existing files/terms rather than updating them.
Supporting a new dump means the extraction process must actually replace changed
files, remove absent articles, and update/rebuild its Xapian index. We will consume
that resulting format directly, not import or invoke those scripts. Extraction
must leave a changed mtime for changed bytes; do not preserve old mtimes on changed
content. Normalize timestamp precision/timezone consistently with the SQL column.
If upstream cannot provide this contract, a content fingerprint would be needed;
it is unnecessary for an extraction workflow that sets mtimes correctly.

A new dump remains in the same collection. The maintenance flow above refreshes
already indexed articles; full mode additionally indexes all new ones. Xapian is
only a discovery index, not the authority for current file membership or mtime.

### Reuse TextFileSource

Implement `WikipediaSource(TextFileSource)` with `source_prefix = "wikipedia"`
and `separate_rerank = True`. Override the semantic boundary expressions for
MediaWiki headings and the embedding boundary expressions for paragraphs. Reuse
file retrieval, encoding, enumeration, chunking, and the text source's existing
document-summary implementation. Changing the heading expressions is sufficient
for the initial raw-text implementation; it does not justify bypassing TextSource.

`WikipediaSourceConfig(TextFileSourceConfig)` adds `xapian_directory: Path`.
Retain inherited directory, include/exclude, encoding, and summary-length options.
The source constructor keeps the existing `DocumentSource` signature:

```python
def __init__(
    self, collection_id: str, *, context, collection_config, reranker=None,
): ...
```

Call `super()` and resolve the Wikipedia config. `WikipediaPointer(TextFilePointer)`
only overrides `get_metadata()` to use the filename stem for the title; the
source's `fetch_document` returns it. Reuse all other pointer behavior. These
classes are Wikipedia-specific; no generic Xapian wrapper, mixin, or match DTO is
needed. If another Xapian-backed source arrives, extract shared code then.

Implement `native_search` directly on `WikipediaSource`, calling Xapian's Python
bindings. A private synchronous `_native_search` worker may contain the blocking
query code for `asyncio.to_thread`; it has the same inputs and returns ordinary
`SearchHit` values. It is an execution helper, not another public search API.

Within a query, open a read-only database and keep parser/enquiry/MSet use in that
worker. Use English stemming, explicit `STEM_SOME`, `title` -> `T`, `cat` -> `C`,
and the inspected query flags `FLAG_DEFAULT | FLAG_BOOLEAN_ANY_CASE`. Validate
the deployed parser's default operator before making it explicit. Release the
handle at query completion so subsequent queries see a replaced database.
Lazy-import the binding so other sources do not need it installed. See the
[Xapian Python API](https://xapian.org/docs/bindings/python/xapian.html).

Map hit data to `<stem>.mediawiki` and the existing file-source path encoding.
The result must be the same local ID as enumeration. Decode exactly once, apply
the same include/exclude rules, and reject paths outside the configured root.
Do not use Xapian numeric document IDs as persistent IDs. No presence check is
added to discovery/preparation; a stale hit fails when content is actually fetched.

Initially return ranked `DocumentHit`s with Xapian weights, honoring
`document_limit` independently of `chunk_limit`. A zero document limit returns
no document hits. Native query failures use the existing search failure path,
allowing vector results to remain useful.

### Body-term positions and native chunk hits

Body positions are usable evidence, not inherently unmappable. In the inspected
builder, `set_termpos(1)` resets the sequence before each category, before the
title, and before the body. `index_text` then advances positions through each
input. Thus category/title positions are relative to those fields, while body
positions run through the raw article. A position greater than 1 does not identify
the field; the term prefix does. Categories are not all stored at position 1.

Xapian exposes matching terms and position lists. However, positions count tokens,
not bytes or characters, and some stemming strategies omit positions for stem
terms. Do not infer source offsets by splitting on whitespace. Establish the
actual stem/position behavior with a tiny database built using the same settings.
See the [TermGenerator API](https://xapian.org/docs/apidoc/html/classXapian_1_1TermGenerator.html)
and the Python API linked above.

Before deciding whether to include native chunk hits in the first implementation,
run a bounded mapping experiment:

1. For a matching article, read body matching terms and their available positions.
   Exclude title/category/identity terms, including their stemmed forms. If a
   matched stem has no positions, identify matching unstemmed body terms using
   the same stemmer and their position lists; do not simply strip `Z`.
2. Reproduce Xapian token ranges for the article's semantic chunks using a temporary
   `TermGenerator` and the same body settings. Carry positions across boundaries;
   verify that segmented tokenization agrees with whole-article tokenization.
   Headings/paragraph boundaries generally provide natural breaks, but arbitrary
   embedding boundaries may split a token and must not silently change numbering.
3. Associate matching positions with semantic spans, then identify the embedding
   span(s) containing the match using the same embedding-chunk construction as
   indexing. Emit existing `b/e/o/s` metadata, not a Xapian position disguised as
   an offset. Compare reconstructed term/position data with the native document
   to catch changed files or tokenizer incompatibility; fall back to document hits
   when alignment cannot be established.
4. Test repeated words, Unicode, stemming, phrases, and queries combining category
   restrictions with body terms. A passage containing a matching word need not
   independently satisfy the whole Boolean query. Use positions to discover
   relevant passages, with reranking deciding usefulness.

If the mapping is reliable with modest code, add separately limited chunk hits.
Rank them by parent match rank, then matched body-term coverage, with source order
as the tie-breaker; keep those scores separate from document scores as usual.
Otherwise ship document hits and retain this as a concrete improvement rather
than claiming the existing index cannot support it. Title/category-only queries
can always produce document hits without inventing article passage locations.

### Text, summaries, and optional expansion

Use raw MediaWiki initially. A capable model should understand it, including
named infobox parameters. Keep the inherited TextSource summary path: it can
summarize an article directly within `summary_length`, and the generic fallback
handles larger articles. Do not override it to unconditionally return an empty
summary. No query-time hit summary is needed initially.

Rendering is optional and primarily a conversion to a format other than MediaWiki.
Do not assume Wikipedia template expansion adds useful facts: template names
and arguments may be more useful to an LLM than their expansion. Evaluate embedding
and reranking quality on raw text separately from LLM comprehension. Preserve
original-file byte spans and decoded embedding character offsets; test Unicode
round trips. Any correction to byte-versus-character length reporting belongs
in the shared file implementation and should reflect its existing API contract.

The user is willing to obtain template/module definitions and use a modest
library, but not install PHP MediaWiki.
[`wikitextprocessor`](https://github.com/tatuylonen/wikitextprocessor/blob/main/README.md)
is an optional rendering candidate: it supports cached definitions, template/parser-function
expansion, and Scribunto Lua execution, including a Wikipedia project setting.
Test a small representative article sample with definitions from a matching dump
before committing to integration. The current extractor discarded non-main
namespaces, so those definitions must be obtained separately. A syntax parser
such as `mwparserfromhell` alone does not expand templates.

Expanded wikitext may be sufficient; Markdown rendering is not required. If
adopted, chunk/embed/fetch the same cached expanded representation and invalidate
it when article or template inputs change. Raw-file offsets cannot address the
expanded text. Defer new processing classes until this experiment establishes
the representation needed. No rendering dependency is required for the initial
source and no packages or dumps have been installed for the experiment.

### Example configuration

```toml
[collections.wikipedia]
tool_prefix = "wiki"
description = "Local English Wikipedia"
indexing_mode = "indexed"
doc_summary_prompt = "Summarize the article in plain language, interpreting MediaWiki markup."

[collections.wikipedia.source_config]
type = "wikipedia"
directory = "/srv/wikipedia/pages"
xapian_directory = "/srv/wikipedia/wikipedia-xapian"
include = ["*.mediawiki"]
```

## Implementation and verification

1. The authorized base changes (`default_indexing_mode`, `indexing_mode`,
   `separate_rerank`, missing-document exception and fetch/mtime contract) are
   implemented in `mcp_indexer/plugins/base.py`. The user's existing `self`
   correction was preserved.
2. Add collection policy and maintenance refresh/deletion using existing indexer
   entry points. Move the approved ensure-indexed helper into the indexer.
   Preserve concurrent summaries and guard their writes against replacement.
3. Fix ORM ownership and provide the manual production SQL migration, including
   a collection-rename check. No production migration occurs as part of planning.
4. Add the TextSource subclass, Xapian discovery, plugin registration, config
   example, and conditional MCP rerank-query parameter. Perform the term-position
   experiment before finalizing document-only versus document-and-chunk discovery.
5. Validate using pytest, monkeypatch and handwritten fakes, plus a tiny Xapian
   fixture and a few read-only queries over the actual corpus. Use the requested
   `~/ai/venv/bin/python`, as subsequently specified by the user. Xapian is
   available there, but several application dependencies are absent. No full Wikipedia
   embedding job is needed to validate the source.

Required checks cover both modes refreshing changed mtimes (including older
timestamps), prompt deletion after a missing fetch, retention on I/O errors,
atomic replacement/failure, summary-write races, cascade deletion and manual
migration, configuration-fragment overrides, same-collection dump replacement,
local/global ID round trips and collection rename, structured/native versus
natural-language queries, unchanged search ranking, and raw-text chunk retrieval.
Verify indexed mode never enumerates the corpus and search preparation never
performs maintenance presence checks. Preserve the complete-chunk invariant.

Resolve the existing architecture wording for IDs and document the new maintenance
and query-input behavior during implementation, subject to the repository's
approval rule for revising existing architecture sections. Unreviewed implementation
details in this revised proposal remain proposals, not assumed approvals.

Implementation results and remaining checks are recorded in
`implementation-notes.md`. Native discovery currently returns document hits;
semantic-position reconstruction was verified on three real articles, but exact
embedding-span mapping remains unimplemented. Rendering/expansion is not enabled.
