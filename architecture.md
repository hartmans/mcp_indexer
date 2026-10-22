# Architecture

## Purpose and component boundaries

The indexer makes heterogeneous document collections searchable and retrievable
through MCP. It combines a persistent index of document summaries and text chunks
with optional discovery in a source's own search engine. A collection can therefore
be indexed in advance or acquire indexed coverage as users search it.

A collection is one configured `DocumentSource` instance. The source owns discovery
and document acquisition; bound `DocumentHandle` objects own document interpretation.
The `Indexer` owns persistence, child-work scheduling, embeddings, and summary
composition. Search orchestrates discovery, scoring, and result assembly through
these interfaces. MCP and the interactive client adapt the shared application
operations rather than implementing independent search pipelines.

`Context` supplies database sessions, collections, batched chat and embedding model
access, and shared resource controls. Configuration resolution separates server
resources, collection infrastructure, and source-specific validation. Sources own
the schema of their configuration; format plugins receive interpretation policies.
The configuration schema and operational instructions live in [README.md](README.md).
Exact method arguments and storage mechanics belong in the corresponding code.

## Collection identity and document trees

Sources, handles, and database rows use collection-local document IDs. Database
identity always includes the collection separately. External references add
`collection_id:` at the tool boundary; this prefix names a collection, not a source
type. Changing a collection's name does not change its local document identities.

A literal colon within a local ID separates subdocument levels. For example,
`library:reports/report.pdf:figure` identifies a child of `reports/report.pdf`
inside `library`. Literal colons in filename components are percent-encoded;
hierarchy separators must be split before decoding. Chunk IDs append `?c=...` to
the complete local document ID. Child resolution receives that complete local ID,
not just the final component, so nested handlers retain enough context to resolve
their own descendants.

Documents form trees within a collection. A child's parent must exist in the same
collection, and deleting a document deletes its subtree. Reparenting and
cross-collection movement are not supported workflows. Direct addressability does
not imply cheap or independent access: resolving a figure may require converting
its parent document. Direct indexing of a descendant establishes missing ancestor
rows first without requiring every sibling subtree to finish.

## Acquisition, interpretation, and content

`DocumentSource` enumerates roots, resolves roots and descendants, and optionally
provides native search. `DocumentHandle` binds a source, full local identity, and
modification time to operations for metadata, semantic chunks, document summaries,
and immediate children. Acquisition is asynchronous and returns a durable handle,
not an acquisition-scoped context manager. Handles yielded by enumeration remain
usable after enumeration ends. Missing documents raise `DocumentNotFoundError`;
an unavailable source is a different failure and must not establish absence.

Document shapes are combinations of operations. A document may have chunks,
children, both, or only a summary. A summary-only image still participates in
indexing and document-vector search. A container can have children without its
own text. Generic orchestration must support these shapes without branching on
format names.

Each source owns a process-local handle cache shared by direct acquisition, root
enumeration, and child enumeration. Enumerators register acquired handles through
the source. Idle eviction releases cache references without invalidating handles
held by ongoing work. This cache manages reuse and memory, not freshness: a cache
hit does not probe the source. Process-lifetime staleness is accepted, and a handle
is not a snapshot guarantee for mutable underlying bytes.

`DocumentPlugin` is a discoverable handle implementation that interprets a
`ContentBridge`. Its handler-name registry is separate from source registration
and from the extension-to-policy map. `FileSource` selects an interpretation
policy for acquired files; other sources may construct a known handler directly.
Policy describes interpretation and is read-only by convention. Metadata describes
the particular document and can contain structured JSON-compatible values.

A content bridge provides asynchronous whole-content and half-open byte-range
reads, plus scoped access to a filesystem path. Every bridge supports all three
operations, adapting its content internally. Real files can expose their paths;
in-memory conversion output can materialize a temporary file for a path consumer.
Path lifetime belongs to that context, not to the lifetime of the document handle.
Bridge reads preserve original bytes, including newline sequences. This boundary
lets format interpretation work with files, extracted figures, or future mail
attachments without teaching the indexer how their bytes were acquired.

Delegation is distinct from hierarchy. A converter can interpret its output
through another plugin with the same document ID; that delegate introduces no
new document or summary level. Docling uses a Markdown `TextDocument` delegate,
while extracted figures become separately addressable `ImageDocument` children.
Images obtain summaries through the shared image-capable LLM path. Format plugins
own conversion and figure identities; generic indexing owns their persistence and
failure handling. Optional format dependencies must not prevent unrelated sources
or plugins from loading.

## Chunking, retained text, and retrieval

Semantic chunks are the handle-facing unit for contextual retrieval and
summarization. Embedding chunks partition each semantic chunk into model-sized
text. A handle supplies semantic retrieval metadata and candidate embedding
boundaries; generic sizing enforces the maximum while treating the minimum and
candidate boundaries as hints. Concatenating candidates, or the resulting stored
embedding chunks, must reproduce exactly the semantic text returned by the handle.
This invariant survives changes to sizing algorithms.

Source metadata locates semantic content. Embedding offsets and lengths locate
character spans within that decoded content and are separate from bridge byte
offsets. Consumers must not interchange these coordinate systems. Every stored
embedding chunk belongs to exactly one semantic chunk and has a stable order
within its document. Summary spans group chunks for summarization without changing
their semantic or embedding boundaries.

A handle's `save_text` policy requests retention of each embedding chunk's exact
text. Retention is independent of input storage: converted file content can need
retention to preserve the indexed representation across later converter changes.
Ordinary ORM loads defer this text because most consumers need references and
summaries, not the content itself.

The shared `retrieval.fetch_chunk(source, chunk_id, scope=...)` helper is the
application boundary for stored chunk text. Embedding scope returns one stored
span; semantic scope reconstructs its whole semantic group even when the supplied
ID refers to a non-root embedding chunk. Complete retained text is returned without
acquiring a handle. Otherwise the helper acquires a cached handle and retrieves
the semantic text, applying a character slice for embedding scope. Missing rows,
NULL text, and empty text are distinct. Retrieval must not concatenate partial
retained text with content from another source version.

Search hits without stored chunk IDs instead carry enough semantic metadata and
span information for direct handle retrieval. They need not be indexed merely to
score their text. This distinction is based on stored identity, not on whether
native or vector discovery produced the hit.

## Persistence and indexing lifecycle

PostgreSQL and pgvector hold durable index state. `Document` is the unit of
indexing and stores identity, parentage, modification time, metadata, summary, and
summary embedding. `DocumentChunk` stores embedding units and their semantic
membership; `ChunkSummary` stores summaries of spans. `IndexingWork` records
pending child acquisition, and `FailedDocument` records handled indexing failures,
including failures for documents with no stored row. Composite collection-local
keys and database cascades enforce ownership throughout these relationships.

Document metadata is persisted separately from the dedicated title and keyword
fields. Tool responses expose nonconflicting metadata while reserving their own
identity, summary, chunk, and child fields. Metadata cannot overwrite protocol
fields. Fetching a document exposes its immediate children so hierarchy is
discoverable as well as directly addressable.

Indexing prepares a document's complete chunk set and enumerates its immediate
children. The document, its chunks, and pending work for discovered children are
published atomically. A parent commit promises recoverable child work, not a
fully indexed subtree. Successful complete enumeration permits removal of absent
children and their work; partial or failed enumeration never establishes absence.
Child modification times inherit ancestor changes.

Work entries identify targets and parents, not serialized handles or frozen
source revisions. A child commit publishes its own descendant work and retires
its completed work atomically. Restarted indexing processes pending work before
new source discovery. After a parent commit, already enumerated children can be
scheduled immediately through the shared indexing task machinery, reusing cached
handles. Handled failures are recorded and excluded from ordinary scheduling,
even if a parent later recreates work for the same ID; they do not cause an
unbounded automatic retry loop.

Parents must not hold a concurrency slot while waiting to acquire child slots or
for children to finish. Descendants participate in the same task and idle tracking
as root work. Recovery does not require a whole-subtree transaction, generation
tokens, or a second child-worker pool.

Full indexing discovers new roots by enumeration. Indexed-only coverage adds
roots through search or explicit requests. Durable child work remains independent of root discovery policy.
`ensure_document_indexed` reuses persisted documents without source probes and
fully indexes missing ones. Search does not refresh an existing indexed document
or persist only its matching chunks.

Maintenance is a separate opt-in pass over stored documents. It compares acquired
modification times, prepares changed content before atomic replacement, and deletes
confirmed missing documents while retaining records on other access failures.
Refresh invalidates old summaries, and summary writes guard against changed or
deleted rows. Maintenance uses the same acquisition path and inherits its cache
staleness; immediate freshness and coordination across source revisions are not
promised. Redesigning that policy is separate from adding formats or search modes.

## Summaries and shared model resources

Chunk summaries describe spans of a document's own text. Document summaries
combine available own-text inputs with immediate-child summaries. The composition
keeps text or excerpts, summaries of parts, and identified child summaries distinct.
Descendant information flows upward through immediate children rather than being
flattened into every ancestor's prompt. A handle-provided summary cannot bypass
available child summaries, while a summary-only leaf can use its supplied summary
directly. Documents without chunks still receive summary embeddings.

Document summarization proceeds bottom-up once descendant indexing work has
completed or failed and available child summaries are ready. Readiness waits must
not occupy slots needed by descendants. Completion and failure events trigger
newly ready parent work; a catch-up pass handles work completed in an earlier
process. Search result summarization can converge the required tree bottom-up.
Failed descendants follow the general failure policy rather than blocking parent
readiness indefinitely or creating provisional summaries after every completion.

Concurrent preparation feeds shared model batchers. Expensive conversion uses
context-level concurrency control, and reranking uses one shared configured
adapter for enabled collections. Reranker setup completes before a new search
issues embedding or summary requests, reserving local accelerator resources
before other model services can claim them. Adapters may run in-process or call
an external service, but search requires a finite score for every prepared input;
scoring and page selection are separate operations. Database sessions must not
remain open across source conversion or model calls.

## Discovery, scoring, and result assembly

`search.py` is the common search application interface. Vector discovery searches
the indexed subset; optional native discovery searches the source corpus, including
documents not yet persisted. They run concurrently and return detached
`DocumentHit` and `ChunkHit` values rather than ORM objects. Native scores and
vector distances are not comparable across discovery lists.

Search deduplicates before text preparation or scoring. Document summaries and
chunk spans are separate evidence for a document. Equivalent chunk references
must share canonical semantic metadata and consistent span coordinates; a stored
chunk ID is a retrieval reference rather than a separate discovery identity.
Each distinct accepted hit is successfully scored at most once per session.
The first accepted scoring input is retained as the choice for that hit rather
than being refreshed and rescored during continuation.

Without reranking, selection follows encounter order: vector documents, vector
chunks, native documents, then native chunks within a batch, with older pending
hits first. With reranking, all new prepared hits receive scores, including those
not selected for this page. Documents compete by their best qualifying discovered
hit; ties preserve encounter order and then document identity. Thresholds apply
before page selection. A later strong chunk can qualify a document whose summary
was rejected, but a returned document is never updated or returned again along
the continuation chain.

The page limit counts distinct successfully assembled documents. Selected
documents include their qualifying discovered chunks with duplicate references
removed. Document scoring uses a supplied hit summary when available, otherwise
the stored/source summary and generic summarization path. Preparation may index a
candidate that is never selected. Assembly ensures selected documents are fully
indexed, fills missing summaries while preserving existing ones, and loads response
objects after writes finish. Independent sessions do not gain a new global indexing
or summarization lock through search.

## Incremental discovery and cursor sessions

A `SearchResult` carries results, a continuation cursor, and warnings. Each source
owns an in-memory session store. A session freezes the query, effective natural-
language query, candidate batch sizes, thresholds, and reranker, while allowing the
page limit to change on advancement. Sources with structured native syntax can
expose a separate natural-language rerank query; it drives both vector discovery
and reranking while the original query drives native discovery.

Each advancement performs at most one bounded batch per unfinished discovery
stream, even when retained candidates could fill the page. New evidence competes
with pending evidence; pages are ordered only by what has been discovered so far.
Later pages can score better, and short or empty pages can still have continuation.
A session ends only when discovery is exhausted and no eligible pending or
retryable item work remains. Bounded candidate counts do not bound elapsed source
or model time.

Vector discovery keeps separate document and chunk positions and reuses the query
embedding. Positions advance by raw candidate counts before duplicate suppression
or rejection. The existing ANN/HNSW path remains approximate; exhausting its
exposed candidates does not prove exhaustive recall. Native discovery returns
`NativeSearchBatch` with source-owned continuation positions, passed back unchanged.
An empty batch still advances at least one enabled position; only `None` means
native exhaustion. Limits can disable a hit type, whose position remains unchanged.
Unchanged native inputs require deterministic traversal and stable ties. Wikipedia
uses raw Xapian match positions so filtered matches consume scan budget without
requiring a rescan from the start.

Duplicate and returned-document suppression belongs to orchestration. Discovery
must not receive seen sets or shrink its offset-based candidate universe around
them, which could skip unrelated results. Corpus changes can still reorder or omit
candidates during a session; cursors are not database or source snapshots.

Cursor tokens are opaque, collection-local references to process-local state.
The current token advances; the most recently consumed token replays its cached
page and next token. Successful advancement invalidates the older replay token.
A final page remains replayable until expiration. Replay returns isolated copies
of loaded response data without database, source, or model work, so caller mutation
and corpus changes cannot alter the cached page. Collection resolution and any
applicable access checks still precede replay.

Concurrent requests for the same advancing token share one shielded operation.
Cancelling a waiter must not cancel that operation or cause completed scoring to
repeat. Page snapshot publication, returned-document tracking, pending-hit removal,
and token rotation commit together in memory. Saved phase progress allows retries
after a page-production failure without repeating successful discovery or scoring.
The initial request has no client-visible retry token until it returns.

Sessions expire after inactivity, including cleanup when no new requests arrive;
in-flight advancement is protected from eviction. Restarting or replacing a
source invalidates its tokens. Multiple serving processes require sticky routing.
The replay window bounds response history, not active seen-set memory; hard memory
caps would require an explicit admission policy. Terminal sessions release candidate
state while retaining the final replay page.

## Failure boundaries and extension direction

Discovery failure is never exhaustion. Successful streams and pending work may
produce a page with warnings while a failed stream remains retryable. If all
attempted discovery fails with no usable work, search returns a retryable error.
Reranker setup or batch failure also returns an error instead of mixing scored
results with an unscored fallback. Successful discovery and scoring survive a
failed continuation attempt.

Individual preparation and assembly failures receive bounded later retries and
then explicit warnings when discarded. Failed assembly never marks a document
successfully returned. Persistent discovery outages can keep a session unfinished
until expiration; unreadable documents prevent a guarantee of exhaustive results.
Invalid or expired cursors are errors, never implicit new searches or empty
terminal pages.

New corpus access belongs in a source, new byte acquisition in a bridge, and new
format interpretation in a handle or plugin. Converters should reuse text
interpretation through delegation and retain converted text when reproducible
retrieval requires it. EPUB conversion and mail attachment interpretation fit
these boundaries but are not implemented features. Whether a chapter or message
part is a subdocument depends on its identity, lifecycle, and retrieval needs;
conversion alone does not require a new hierarchy level. Binary tool delivery,
generalized streaming, and stronger freshness or subtree-version coordination
remain separate design work rather than implied requirements of these interfaces.
