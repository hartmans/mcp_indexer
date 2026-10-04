# Resumable search implementation design

This is an implementation handoff for incremental discovery with collection-local,
in-memory cursors. The user approved the design choices on 2026-10-03, including
the defaults, failure policy, and base-interface changes specified below, and
explicitly directed that implementation require no further approval for base.py.
This satisfies the AGENTS.md approval requirement for these scoped changes.
Implementation may proceed without another design approval step. This handoff
itself changes no application code. Code inspected is under `mcp_indexer`, despite
the older `lance_indexer` path in AGENTS.md.

## Agreed behavior

- Resume discovery, not just an initial pool of results. Repeated continuation
  should eventually traverse the candidates exposed by discovery, subject to
  thresholds, successful retrieval, and the qualifications below. Vector discovery
  remains approximate: missing documents due to ANN/HNSW is explicitly acceptable;
  do not introduce exhaustive vector scans.
- Score each distinct hit at most once per search. A document summary and each
  of its embedding chunks are separate hits. Never return a document twice along
  a chain of successive cursors. Replaying a page deliberately repeats that page.
- A new chunk may qualify a previously unreturned document whose summary scored
  poorly. A new chunk of an already returned document never produces an update.
- Select up to `limit` distinct documents, rather than limiting hits before
  grouping. This intentionally changes both reranked and unreranked behavior.
  Short or empty pages are permitted, including pages with a continuation cursor.
- Rank using discovered evidence. Later pages may contain better results than
  earlier pages. Do not run all discovery before the first response.
- Reuse saved `query` and `rerank_query` on resume; warn when supplied values
  disagree. Return a named result containing documents, continuation, and warnings.
- Keep the cursor that produced the latest page for replay and the cursor that
  advances to the next page. Advancing successfully invalidates the older replay
  cursor. Store sessions on each `DocumentSource` instance.
- Expire sessions after 60 minutes of inactivity. No database persistence is
  required. Restarting or replacing a source instance invalidates its sessions.
- Discovery uses offsets and deterministic ordering for unchanged inputs and
  corpus. Data changes may cause missed documents or surprising order, but must
  not defeat collection isolation or applicable access checks.
- Native discovery returns `NativeSearchBatch | None`. Only `None` establishes
  exhaustion. A batch with `hits=[]` means no usable hits in the scanned range;
  its returned offsets advance discovery. Exceptions do not establish exhaustion.
- Duplicate suppression belongs to search orchestration, after discovery and
  before text preparation or scoring, for both native and vector hits.
- Tool output includes `resume_cursor` and `resume_cursor_usage`. Use the corrected
  spelling of the latter consistently. Its exact values are:
  `End of documents reached` when the cursor is null; otherwise
  `Call this tool again and pass in the cursor in the cursor argument to resume.`

## Approved implementation defaults

The user approved the following defaults with the rest of this design.

1. Each advancing call performs at most one batch from each unfinished discovery
   mechanism, even when pending candidates could fill the page. This gives all
   streams progress and lets newly discovered hits compete with retained ones.
   It bounds candidate counts, not elapsed source or model time.
2. Save candidate batch sizes at creation: explicit candidate limits, otherwise
   initial page limit times four, as today. Permit changing the document result
   limit on advancement. Reject conflicting explicit candidate limits on resume;
   do not silently restart. Snapshot thresholds and reranker choice per session.
3. Require positive page limits and nonnegative candidate limits. A zero candidate
   limit explicitly disables that hit type; disabled streams do not prevent
   exhaustion. This replaces today's empty result for a nonpositive page limit
   with an input error. Reject disabling both hit types.
4. Failures are reported in warnings or explicit errors and never masquerade as
   exhaustion. Successful hit scores are never recomputed; unsuccessful attempts
   may be retried. Detailed policy appears below.
5. Without a reranker, keep discovery order with a monotonically increasing
   encounter sequence. Within each batch visit vector documents, vector chunks,
   native documents, native chunks. Older pending hits precede new hits; do not
   compare scores across those lists.
6. Permit one preparation retry and one assembly retry on a later advancing call;
   then discard that failed hit/document with a warning. This prevents a permanently
   broken source item keeping a search alive forever. Discovery and reranker-wide
   outages remain retryable errors rather than silently declaring completion.

## Public API and result type

Prefer a frozen dataclass over a NamedTuple so callers use field names and adding
fields does not encourage positional unpacking. Keep current hit classes and
document/chunk assembly values; no new public document DTO is needed for this work.

```python
@dataclass(frozen=True)
class SearchResult:
    results: list[tuple[Document, list[DocumentChunk]]]
    resume_cursor: str | None
    warnings: tuple[str, ...] = ()

async def search(
    indexer: Indexer,
    collection_id: str,
    query: str | None = None,
    limit: int = 5,
    *,
    document_candidate_limit: int | None = None,
    chunk_candidate_limit: int | None = None,
    rerank_query: str | None = None,
    cursor: str | None = None,
) -> SearchResult: ...
```

With no cursor, require a query value and preserve existing blank-query behavior.
With a cursor, omitted queries produce no mismatch warning. Compare a supplied
native query exactly against the original string. Normalize an omitted/blank
rerank query to no override both when saving and comparing; preserve nonblank
strings exactly. Compare an explicitly supplied rerank query against that saved
override, not against a newly derived effective query. Always use the saved
effective natural-language query (nonblank override, otherwise native query).
Never include the saved private query text in a warning.

Example warnings: `query differs from this search; using the saved query` and
`rerank_query differs from this search; using the saved rerank query`.
For replay, cache page results and operational warnings; add request-specific
query mismatch warnings afresh. Replay page contents and next cursor are fixed
even if the request supplies a different page limit.

Use explicit errors such as `InvalidSearchCursor` (unknown, old, expired, or wrong
collection, with one generic public message) and `SearchAdvanceError` (retryable
operation failure). No invalid cursor may start a new search automatically.

## Approved base changes

The user has approved the changes below to `mcp_indexer/plugins/base.py`. Do not
request approval again for these changes; the AGENTS.md requirement has been met.
No changes to `DocumentPointer`, chunk retrieval, or indexing policy are needed.

Problem: the existing native API can only return the beginning of two lists, and
the source has no instance-owned location for resumable search state.

Approved constructor addition:

```python
# Add within __init__, after existing assignments. A local import avoids
# runtime cycles through context/plugin registration.
from ..search_state import SearchSessionStore
self._search_sessions = SearchSessionStore()
```

`SearchSessionStore` must not import source classes or `search.py` at runtime;
use TYPE_CHECKING and forward annotations. The constructor performs no I/O and
requires no running event loop. Do not use a class-level dictionary. This one
attribute is the ownership hook; session algorithms belong in the new module.

The user approved this native batch/continuation contract during design review.
The protected base edit remains part of the eventual implementation, not this
documentation-only change.

Define the batch next to the public hit types in `search.py` (and reference it
under TYPE_CHECKING from base), with no runtime import cycle:

```python
@dataclass(frozen=True)
class NativeSearchBatch:
    hits: list[SearchHit]
    document_offset: int
    chunk_offset: int

async def native_search(
    self, query: str, *, document_limit: int, chunk_limit: int,
    document_offset: int = 0, chunk_offset: int = 0,
) -> "NativeSearchBatch | None":
    return None
```

Extend the method docstring with these requirements:

- Offsets are source-defined nonnegative continuation positions, initially zero.
  They need not count returned hits. The caller passes returned positions back
  unchanged; the source owns their meaning. Limits bound returned hits by type.
- Given an unchanged query and corpus, traversal must be deterministic, with
  stable tie-breaking. Return document hits before chunk hits. A source may
  bound internal scanning and return fewer hits than requested.
- A zero limit disables a type. Leave its offset unchanged. If every requested
  type is exhausted, return `None` when there are no remaining hits to deliver.
- A batch, even an empty one, must advance at least one requested type's offset;
  offsets must never decrease. Orchestration validates this to reject stalled
  or malformed continuations instead of looping forever.
- `None` means combined exhaustion. Return final hits in a batch first, then
  `None` on the next call; never discard final hits to signal exhaustion.
- Search orchestration suppresses duplicate hits and already returned documents.
  Sources filter invalid/excluded matches but need no search-session seen set.
- Raise on failure rather than returning `None` or an empty batch.
- Chunk identity is the document ID plus exact retrieval metadata. Equivalent
  references across mechanisms must use the same canonical metadata. A stored
  chunk ID is an additional lookup reference, not a substitute for this contract.

Update every override and handwritten fake to accept offsets and return the named
batch or `None`. Do not add a TypeError fallback for old plugins: silently ignoring
offsets could loop forever. This is an intentional plugin API extension.

## Internal state and lifetime

Add `mcp_indexer/search_state.py` for session records, token lookup, expiration,
and advancement coordination. Keep source/model operations in `search.py`.
Use ordinary dataclasses and an instance-owned store rather than a global cache.

Each session needs:

| State | Purpose |
| --- | --- |
| Saved query, override, effective query, thresholds, candidate sizes, reranker | Fixed search definition |
| Cached query embedding | Avoid embedding the same query on every page |
| Four offsets | Vector document/chunk and native document/chunk progress |
| Two vector exhaustion flags and one combined native flag | Native API only proves combined exhaustion |
| Encounter counter and hit records keyed by identity | Deterministic ties and score/preparation memoization |
| Pending eligible hits grouped by document | Candidates not selected yet |
| Returned document IDs and terminal assembly failures | Suppress later hits for completed documents |
| Retry queues and attempt counts | Retry bounded item failures without rescoring successes |
| Current token, previous token, cached page | Advance and replay window |
| Per-session lock and strong reference to in-flight task | Serialize advancement and survive request cancellation |
| Last-access/deadline and expiry timer | Reclaim all state after inactivity |

Use cryptographically random opaque tokens, for example `secrets.token_urlsafe(32)`.
The source store maps at most two live tokens to a session. No token embeds query,
document IDs, offsets, or trusted caller-provided state. Collection lookup occurs
before token lookup. Existing authorization must run on every request, including
replays.

Use a monotonic clock and a 3600-second inactivity interval. Valid advancement or
replay refreshes the deadline; invalid tokens do not. Pin an in-flight session and
refresh at completion so a long operation is not evicted halfway through. Use a
timer callback scheduled lazily on first session insertion, plus deadline checks
at lookup, to release expired sessions even if no subsequent request arrives.
Cancel/reschedule handles appropriately; do not start a forever-running sweeper
per session. Provide store cleanup for tests and any available source teardown.

Terminal sessions retain only the final replay token and cached page until expiry;
release discovery state, embeddings, pending candidates, and seen sets immediately.
Active seen sets grow with discovery. The two-token window bounds response history,
not all memory. No silent eviction before the promised timeout is proposed; a
deployment requiring hard memory caps needs an explicit admission policy later.
Process-local storage requires a single serving process or sticky routing.

## Cursor transitions and cancellation

The initial call has no retry token. It returns page P0 and token C0 if unfinished.
Losing the initial response can only be recovered by starting another search;
initial-request idempotency is outside this design.

| Request | Response | Retained tokens after success |
| --- | --- | --- |
| C0 | P1 and C1 | C0 replays P1; C1 advances |
| C0 again | Same P1 and C1 | Unchanged |
| C1 | P2 and C2 | C1 replays P2; C2 advances; C0 invalid |
| C2, final page | P3 and null | C2 replays P3 until expiry |

Success means page production and cache commit, not delivery to the client.
Perform page commit, returned-ID updates, token rotation, and previous-token
removal without an intervening await. Snapshot replay data before returning it;
callers must not be able to mutate the cached page through returned lists or ORM
objects. Preserve loaded detached ORM graphs by copying the cached result graph
on return, and test this with real assembled records. Do not refetch on replay.
If ORM copying proves unsafe, introduce an internal immutable page snapshot and
reconstruct detached return objects; do not weaken replay to a fresh DB query.

Under the session lock, recheck token validity, then either return a copy of the
cached page or join/create the one advancement task for the current token.
Wait outside the lock with `asyncio.shield`, holding a strong task reference in
the session. The task alone mutates discovery state. Concurrent current-token
requests share a page; the first request's page limit governs it. Keep the old
replay usable while an advance is running, until the new page commits.

Do not cancel an advancement merely because a caller disconnected. This prevents
the common case of running a reranker successfully, losing its response to caller
cancellation, and reranking it on retry. Unexpected task failure retains staged
discovery and successful scores so the same token can continue; clear the failed
task reference and do not rotate tokens. Exactly-once remote computation cannot
be guaranteed if an external reranker completes but its network response is lost;
the once-only guarantee covers successfully recorded hit scores.

## Hit identity and retained scores

Document key: `("document", document_id)`. Chunk key:
`("chunk", document_id, canonical_metadata)`. Canonicalize supported JSON metadata
with sorted keys and stable separators; reject malformed metadata as a failed hit.
Do not use Python dictionary insertion order or the optional chunk ID in this key.
Source authors must supply consistent metadata for equivalent retrieval spans.
This avoids scoring a native chunk again merely because the vector copy has a
stored chunk ID. Distinct metadata that happens to fetch identical text remains
distinct hits; no text-content deduplication is required.

Deduplicate before text preparation, in fixed encounter order. For conflicting
summaries of the same document, the first occurrence supplies the scoring text;
if absent, use the existing stored/source/generic summary path. Freeze that choice
for the session. Do not update and rerank a hit when its summary or content changes.
Maintain the existing concurrent preparation and assembly behavior.

Refactor current `rerank`: it currently calls `score(..., top_n=hit_limit)` and
discards everything outside the page. Call with `top_n=None` for all distinct
prepared hits, validate the response contains one finite score for every input
index exactly once, and cache all scores. Both current reranker implementations
accept `top_n=None`. Selection and thresholding are separate from scoring.
Keep below-threshold identities marked as successfully scored, but discard their
text and unnecessary payload. A later different chunk can still qualify its parent.
Drop later hits for already returned documents before fetching text or reranking.

## One advancing call

1. Resolve collection and validate cursor/access/arguments. Create or retrieve a
   session and reserve its advancement task. New sessions reserve reranker resources
   before embedding/summary work, preserving the existing setup ordering.
2. Obtain the query embedding once. Concurrently run one native batch and the
   unfinished vector document/chunk batches using saved offsets. Disabled types
   are not queried. Discovery outcomes must distinguish success from failure.
3. Stage each successful batch. Advance vector offsets by raw returned counts
   of each type, before orchestration deduplication, score rejection, or suppression
   of already returned documents. For native discovery, validate and save exactly
   the offsets supplied by `NativeSearchBatch`, never derive them from hit counts.
   Failed batches leave offsets intact. Vector empty batches exhaust their individual
   approximate streams. Only native `None` exhausts native discovery; an empty native
   batch still advances its supplied offsets. Stop querying exhausted mechanisms.
4. Deduplicate and prepare new hits plus eligible preparation retries. Score all
   unscored prepared hits once; keep successful scores immediately. Retry queues
   do not force new discovery to repeat after an unexpected page-production error.
   Track the active advancement's completed phases to prevent double fetching.
5. Merge eligible new hits with pending hits. With reranking, order documents by
   their maximum qualifying hit score, descending; break ties by earliest hit
   encounter, then document ID. Include all qualifying discovered chunk hits for
   a selected document, not just the chunk which provided its best score. Without
   reranking, order documents by earliest encounter and merge their discovered hits.
6. Select up to `limit` distinct unreturned documents and assemble them concurrently.
   Count only successful assembly as returned. Permit a short page on assembly
   failures rather than performing further discovery or unbounded backfill.
   Keep unsuccessful eligible candidates for their bounded assembly retry.
7. Prepare the replay snapshot. Completion requires every enabled discovery stream
   exhausted, no eligible pending documents, and no retryable item work. Otherwise
   issue a fresh token even for an empty page. Commit snapshot, returned IDs,
   removal of those documents' pending hits, and cursor rotation atomically in memory.

An example: the first batch scores summaries A=.90, B=.80, C=.40 and returns A/B
with limit 2. A later batch scores a new chunk of C=.95; C can now be returned.
Neither C's summary nor that chunk is scored twice. A later chunk of A is ignored.

## Vector and native discovery details

Refactor vector discovery to expose separate per-type outcomes rather than catching
exceptions and flattening them into one empty list. An internal `DiscoveryBatch`
record containing document/chunk hits and per-type errors is sufficient. Native
uses `NativeSearchBatch | None`; orchestration wraps errors independently of that
return value. No source sessions or ORM sessions span model calls.

Vector SQL retains collection and cosine-distance predicates before LIMIT/OFFSET.
Use the existing deterministic ordering: distance, document ID, then chunk ID
for chunks. Bind independent offsets. Cache the embedding in the search session
and provide a private helper accepting it so resumes do not embed the query again.
Do not pass seen-hit or returned-document sets into vector/native discovery or add
exclusion predicates for them. Shrinking the candidate list under an offset can
skip unrelated candidates. Discovery returns its candidates; the higher-level
search orchestration alone suppresses duplicates and already returned documents
before text preparation and scoring. This applies equally to both mechanisms.

Keep the existing ANN/HNSW vector search path. Approximate indexes may miss
qualifying documents, and the user explicitly accepts that limitation. An empty
vector batch ends traversal of candidates exposed by that approximate query; it
does not prove every qualifying database row was found. Do not force exact scans,
disable HNSW, or introduce exhaustive fallback discovery. Keep stable SQL tie
ordering but do not promise that ANN results are an exhaustive fixed sequence.
OFFSET may grow costly for deep searches; alternative pagination is deferred.

Wikipedia currently returns document hits only. Pass `document_offset` through
`asyncio.to_thread` to `_native_search`; it now means a raw Xapian match position.
Use deterministic relevance order with stable docid ties supported by the binding.
Native weights remain incomparable with vector distances.

Start `get_mset` directly at the supplied raw position. Retain a bounded raw scan
budget such as `max(100, document_limit * 10)` per invocation and bounded block
allocations. Filter malformed/excluded matches, accumulating at most document_limit
hits. Increment the raw position for each match actually examined, including
rejected matches. If the output limit is reached partway through a fetched block,
return the position immediately after the last examined match, not the end of that
block, so remaining matches are not skipped. Cross-page duplicate suppression is
orchestration's responsibility; duplicate source records may consume output slots.

Return `NativeSearchBatch(hits, next_raw_position, chunk_offset)` after progress,
even when hits is empty. Return `None` only when Xapian is actually exhausted and
there are no hits to deliver. If exhaustion is encountered after collecting hits,
return those hits and their position; the next request may confirm exhaustion.
With document_limit zero (Wikipedia has no chunk hits), or a blank native query,
return `None`. Keep the database lifetime inside the worker; do not fetch document
files during discovery. No rescan from zero or source-side session cache is needed.

## Approved failure behavior

The existing architecture falls back to initial ordering when reranking fails.
For resumable search, the approved design instead returns a retryable error on a batch
reranker failure, retaining discovery/preparation and any successful scores, with
the same current cursor. Mixing unscored fallback hits with retained scored hits
would need another ordering policy and could bypass the threshold. The user
approved this behavior change; reconcile the corresponding architecture wording
as part of implementation.
If setup fails, similarly fail the new search before discovery rather than silently
creating an unreranked session. Sources configured without reranking work normally.

If one discovery mechanism fails while another succeeds, proceed with the usable
batch/pending candidates and include a warning; keep the failed mechanism live.
If all attempted mechanisms fail and there is no usable pending work, return a
retryable error without rotating the token. A permanently failing discovery
mechanism can prevent completion until the session expires; do not label it EOF.

Preparation failures use the bounded retry policy above; other hits can continue.
Assembly failures never add a document to returned IDs. After its last allowed
failure, suppress that document for this session and report the loss explicitly.
No guarantee of exhaustive successful results is possible for unreadable content.
Keep warnings page-scoped; do not accumulate an unbounded history of error strings.

## Tool and CLI integration

Update `core_search` and both tool closures in `server.py`. Both expose optional
`cursor: str | None = None` and optional `query: str | None = None`; the structured
source tool alone continues to expose `rerank_query`. First-page query requirements
are validated in `search`, since the tool schema must allow cursor-only calls.

Return a JSON object, replacing the old top-level array:

```json
{
  "results": [],
  "resume_cursor": "opaque-token",
  "resume_cursor_usage": "Call this tool again and pass in the cursor in the cursor argument to resume.",
  "warnings": []
}
```

Use `result_info(page.results, context.collections)` for the existing result entry
format. Keep global ID formatting and companion fetch tools unchanged. Use one
shared response-envelope helper for MCP and CLI, so empty/terminal usage text
cannot diverge. Document limit as maximum distinct documents, query reuse,
60-minute inactivity expiration, and retry-window semantics in tool descriptions.
Malformed/expired cursor errors must be tool errors, not the terminal envelope.

For CLI, add an explicit `:more` command using the most recently returned cursor;
new query input starts a new session. Keep `query => rerank query` parsing intact.
Display the same envelope and warnings; explain when there is no continuation.
Retain the cursor on errors so `:more` can retry. A separate process cannot resume
another process's cursor, so do not imply that a command-line token survives exit.

## Implementation sequence

1. Follow the approved design below; do not insert another approval gate for
   the specified base changes or design choices. Check the current code and local
   changes before implementation.
2. Add `SearchResult`, store/session records, deterministic identity helper,
   expiration and two-token replay logic with isolated handwritten fakes.
3. Apply approved base changes. Add vector offset/outcome helpers and Wikipedia
   raw-position pagination; update all overrides and fake signatures.
4. Split scoring from document selection. Preserve all successful scores and
   pending eligible hits. Implement the advancement phases and assembly retries.
5. Adapt MCP, CLI, and every Python caller/test that expects search to return a
   list. Search the repository for `search`, `core_search`, `result_info`, and
   `native_search` imports/calls, not only the files mentioned here.
6. Update documentation. Add a new architecture section for cursor semantics if
   useful, and reconcile existing wording with the approved changes to hit limits,
   unreranked four-list limits, fallback ordering, and return type. This approval
   covers those scoped updates, not unrelated architecture revisions.
7. Run targeted tests and relevant existing regressions with `~/venv/bin/python`.
   Do not use the older interpreter named in historical implementation notes.
   No database migration, database cursor table, or index refresh is needed.

## Required verification

Use pytest, monkeypatch, handwritten fakes, and controllable clocks/events. No
unittest mocks. Add `tests/test_search_cursors.py` and extend existing search,
source, server, and CLI tests. Test behavior and counters, not private layout.

- Enumerate stable finite document/chunk lists over multiple pages and verify
  every qualifying document appears once. Cover vector only, native only, mixed,
  no reranker, and more than one page of retained candidates after discovery EOF.
- Duplicate document hits across mechanisms prepare/score once; duplicate chunk
  references with/without stored ID score once. Metadata key order does not matter.
  Different chunks and document summaries remain separate hits.
- A rejected document summary remains rejected without rescoring; a later strong
  chunk returns its parent. Already returned parents' later chunks incur no work.
- Multiple chunks of one document do not consume several document slots. Merge
  all qualifying discovered chunks once, preserving deterministic output order.
- Assert all scores are retained (`top_n=None`), threshold rejection consumes no
  page slots, and new candidates compete with pending scores on resume.
- Vector offsets advance by raw returned counts despite rejection or duplication.
  Native offsets are copied from the batch, including empty batches; only `None`
  ends native discovery. Reject unchanged/decreasing positions. Uneven type
  progress and disabled types do not stall. Discovery never receives seen sets.
- Empty nonterminal pages have a cursor and the continuation usage string; final
  pages have null and the exact end string. No off-by-one loss on final full batches.
- Deterministic fake/native streams traverse all their candidates with stable ties.
  For SQL, inspect LIMIT/OFFSET and threshold filtering using a dedicated fixture.
  Exercise the HNSW path without requiring exhaustive recall or introducing an
  exact-scan fallback; an approximate empty batch terminates its stream.
- Wikipedia resumes at the supplied raw position, including after empty batches.
  Include duplicates, malformed/excluded entries, valid entries beyond one scan
  budget, and output limits reached partway through a block. Assert no unexamined
  matches are skipped and no prefix is rescanned. Orchestration deduplicates the
  resulting hits. Verify final hits precede `None`, exceptions propagate, and zero
  document limit behaves correctly.
- Replay invokes no embedding, discovery, rerank, indexing, or summary calls.
  Mutating the first returned graph cannot change later replay, including MCP JSON.
  Changing corpus content after the first response cannot change cached page data.
- C0/C1/C2 window, final-page replay, generic wrong-collection/expired errors,
  independent source instances, and no state shared through class attributes.
- Sixty-minute boundary, valid replay refresh, invalid access not refreshing,
  idle cleanup without requests, and long-running work protected from expiration.
- Concurrent same-token requests share one operation; old replay during advancement
  is safe; cancellation of one waiter does not cancel work or cause rescoring.
  Inject a failure after scoring but before page commit and verify retry reuses it.
- Query omission/mismatch, blank rerank normalization, saved effective query,
  changed page limit, frozen candidate limits, and replay with changed parameters.
- Discovery errors retain offsets and do not signal EOF; reranker error preserves
  staged candidates; bounded preparation/assembly failures never mark failed
  documents as successfully returned. Verify warnings and retryable error paths.
- Both MCP schemas expose cursor-only continuation, only the structured tool
  exposes rerank_query, all results retain collection-prefixed IDs, and CLI :more
  retains its cursor after failure.

Suggested regression command after implementation:

```sh
~/venv/bin/python -m pytest tests/test_search_cursors.py tests/test_search_rerank.py tests/test_wikipedia.py tests/test_document_source.py tests/test_server_cli.py tests/test_search_client.py tests/test_rerank.py tests/test_llama_cpp_rerank.py
```

Database fixtures use `MCP_INDEXER_TEST_DB_URL` and may create/drop tables. Use only
the dedicated test database. Report unavailable integration prerequisites as such;
do not claim full verification from fake-based tests. Also run `git diff --check`.

## Review summary

The user approved the base changes, bounded-work and failure defaults, and the
design as a whole. No further approval step is required for implementing these
choices or reconciling the corresponding architecture descriptions.
Native batches now carry source-owned continuation positions, allowing Wikipedia
to scan bounded raw ranges without rescanning prefixes. ANN/HNSW recall limits
are accepted; exhaustive vector scans are excluded. The most important traps are
discarding scores outside top_n, deriving native positions from hit counts,
advancing vector offsets after deduplication, treating empty native batches or
errors as EOF, mutating replay data, and rescoring after cancellation.


## Implementation status

Implemented in `mcp_indexer/search.py` and `search_state.py`, with the approved
base ownership/native contract, Wikipedia raw-match pagination, MCP envelope,
and CLI `:more`. The existing Search architecture section now describes the
approved behavior. Replay copies loaded ORM columns and response relationships
into fresh objects because copying SQLAlchemy instrumentation proved unsafe.

Verification with `~/venv/bin/python`: 89 focused tests passed, two optional real
Xapian checks skipped, and three database-dependent cases excluded from that first
run. The two collection-construction regressions passed separately. After the user
supplied the test database connection, all 44 tests in the database regression run
passed, including vector threshold filtering, indexing/search end to end, native
persistence and migration, document/source retrieval, and collection construction.
The runs overlap; their counts should not be added as distinct tests.

The database run set `MCP_INDEXER_TEST_DB_URL` to
`postgresql+psycopg://indexer@/test?host=/srv/datasets/asstr.db/sockets` for the
command only. It covered `test_search_rerank.py`, `test_indexer_e2e.py`,
`test_native_persistence.py`, `test_document_source.py`, `test_context.py`,
`test_file_source.py`, and `test_text_source.py`. No permanent configuration was
changed. These database checks do not establish deployed HNSW query-plan/recall
behavior. The installed interpreter still lacks the optional Xapian binding;
native pagination is verified with handwritten Xapian fakes, including empty
bounded scans, repeated IDs, partial blocks, and offset advancement, rather than
live Xapian integration.

Thread/local-server tests ran outside the sandbox because sandbox restrictions
prevented asyncio worker-thread completions from waking the event loop. No database
or index migration is required or was performed. Compilation and whitespace checks
passed. Existing user-generated `build/` files were left alone.

Follow-up live verification with both `/srv/datasets/asstr.db/config.toml` and
`/srv/datasets/asstr.db/ollama.toml` found a separate Qwen adapter bug: it passed
generic system/user chat messages to a template expecting system/query/document
roles. The rendered Query and Document fields were empty, producing the same
0.0311 score for every candidate. The adapter now uses the distributed template's
roles, supplies the retrieval instruction as system content, and preserves the
answer suffix when truncating long inputs. Regression tests cover role lookup,
distinct rendered pairs, suffix preservation, left padding, and scoring.

The corrected 4B model scored relevant synthetic passages at 1.0 and 0.9805 and
irrelevant ones below 0.00003. Live `asstr` search for `science fiction`, with six
candidates per type, returned two documents without warnings; resuming completed
without duplicates, with that next batch below the configured threshold. The
reranker, cursor, vector SQL, database search, and external-reranker regression
run passed all 59 tests. This live check supplements the earlier fake-based tests.
