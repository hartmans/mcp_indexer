# Subdocument/plugin design checkpoint

This checkpoint distinguishes the implemented baseline, accepted design direction, and the next proposed implementation spike. Updating this plan does not authorize code changes. architecture.md remains deferred to a separate future spike.

## Context and scope

- The user reports substantial recent changes to MCP, search, and chunking. Discard earlier assumptions about those implementations. Inspect current code when implementation requires it; this checkpoint does not depend on old code shapes.
- The current focus is document handlers, subdocument discovery, and content access—not changing search, MCP retrieval, or chunking now.
- EPUB is the motivating implementation. ZIP and notmuch are design probes; implementing ZipDocument or NotmuchMailSource is not currently requested.
- Naming remains provisional where noted. Do not turn assistant suggestions recorded as open questions into accepted requirements.

## Document identities and hierarchy

Documents remain in one collection; cross-collection parenting/movement is not an intended operation. Sources/handlers assign subdocument IDs, probably using a prefix obtained from the parent. Exact ID encoding and routing are undecided.

Earlier discussion established the desired database behavior:

- A document can reference a parent in the same collection via `(collection_id, parent_document_id) -> (collection_id, document_id)`.
- Removing an existing child during a successful parent rescan should delete that child's whole subtree.
- ORM `all, delete-orphan` and a database `ON DELETE CASCADE` were explored for this behavior.
- A model patch was made earlier in the conversation. Its current implementation must be inspected before further work; earlier mapping examples and test claims are not an authoritative specification of the current code.
- No reparenting workflow is anticipated.

Discovery failure must not be confused with authoritative absence. A partially enumerated child set does not establish that unseen children disappeared. Completed child indexing need not be rolled back merely because parent access or later enumeration failed. Exact reconciliation and completion tracking remain undesigned.

## Bound documents and acquisition

**DocumentHandle** is the implemented bound-document interface. “Handler” below describes that role, not an additional wrapper class.

A handler is a bound document object: for example, EpubDocument represents a particular EPUB, not the abstract EPUB format. There is no present need for a separate unbound EpubDocumentType object. The registry can select classes or factories without introducing such an object.

DocumentHandle retains document identity and association with the originating DocumentSource/collection. Its document-facing operations include metadata, chunk enumeration/retrieval, and subdocument enumeration. Exact signatures beyond the agreed shapes below are not settled.

- `get_sub_documents` is an async generator yielding DocumentHandles, parallel to `get_chunks`.
- Future subdocument orchestration will call both chunk and subdocument enumeration; it is not implemented in the current structural migration. Do not enforce exclusive leaf/container categories in orchestration or the base interface.
- Most initial formats may yield only chunks or only subdocuments, but messages are already a concrete case yielding both.
- Future `source.fetch_document(subdocument_id)` must resolve subdocuments, not just roots.
- `fetch_document` is an **async factory returning a DocumentHandle**, not a context manager: `handle = await source.fetch_document(document_id)`. Async acquisition and context-bound lifetime are separate decisions. Resolution may require expensive asynchronous preparation or existence checks; returned handles remain usable independently of the acquisition call.

Direct subdocument resolution may internally reopen and prepare ancestors. Independent addressability does not promise cheap or physically independent access.

## Sources, plugins, and policy

DocumentSource owns collection-level discovery and resolution. FileSource remains a source; TextSource/TextFileSource is removed in the next spike, rather than retained as a text-specific source wrapper.

The accepted hierarchy for the next spike is:

```text
DocumentHandle
    DocumentPlugin
        TextDocument
            WikipediaDocument
        EpubDocument (future)
```

DocumentPlugin is a DocumentHandle that participates in discovery. It consumes a ContentBridge and has a handler-name registry similar to DocumentSource's source registry. Threads and messages in the notmuch probe need not be DocumentPlugins.

The common constructor inputs are `document_id`, `source`, `bridge`, `policy`, `last_modified`, and `metadata`.

- Metadata describes this document, including its title. The constructor makes a shallow copy because the document may modify it. Each get_metadata call returns a fresh shallow copy of the document's metadata; nested values are not isolated by these copies.
- Policy describes interpretation, including text boundary regexps and future behavioral choices such as EPUB decomposition versus delegation.
- Title, boundary regexps, encoding, and summary_length are not separate plugin constructor arguments. TextDocument reads encoding and summary_length from the source's config. Move the text configuration into FileSourceConfig when removing TextFileSourceConfig; encoding already lives there.
- Policy contains a string `handler`, for example `"text"`. Discovery resolves it through the DocumentPlugin registry and passes the whole policy to the constructor without popping that key.
- Extension-to-policy mappings live entirely in code initially. They are separate from the handler-name registry. MIME detection is not required for this spike.
- Markdown, RST, HTML, and plain text can use the same TextDocument with different boundary policies. Do not introduce conversion between these textual formats merely for LLM consumption.
- A handler that knows its child/delegate type can construct it directly with an appropriate policy; registry selection is not a mandatory conversion pipeline.
- Wikipedia retains its known boundaries and article-specific interpretation. EPUB will supply the text policy appropriate to its pandoc output, whether delegating or constructing subdocuments.

### Initial extension policies

The initial code-defined mapping selects TextDocument (`handler="text"`) with these policies:

- `.md` and `.markdown` share one Markdown policy. Semantic boundaries are level-1 and level-2 headings: ATX `#`/`##` and Setext headings. Deeper ATX headings are not semantic boundaries.
- `.html` and `.htm` share one HTML policy. Semantic boundaries are `h1` and `h2`, not `h3` through `h6`. Preserve the markup; no text conversion is introduced.
- `.rst` uses an RST policy recognizing `=` and `-` title adornments as semantic-boundary hints. RST assigns hierarchy by order of adornment styles, not fixed characters; this adopts the documented section/subsection convention, not a claim to infer actual heading levels. See the [Sphinx RST primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections).
- Unknown extensions and extensionless files use the default text fallback. No additional explicit extension mappings are required for this spike.

Settled during implementation: embedding candidates are paragraph breaks for textual formats and paragraph/block endings for HTML. The plain-text fallback treats a line of common separator punctuation (three or more of ``- = * _ +``, with optional trailing whitespace) as its sole semantic boundary. Suffix lookup is case-insensitive.

FileSource registers as `file`; migrate the old text-source configuration to that name. WikipediaSource directly constructs WikipediaDocument with its specialized policy rather than selecting it through the general extension mapping.

### Configuration deliberately deferred

There are no existing boundary-regexp config fields to preserve, and none are being added now. Earlier discussion of collection-wide supplementary text boundaries is a future possibility, not part of the next spike. Supplementation is expected to cover many cases; replacement can be designed when needed.

Config overrides/supplements for policy and discovery mappings are deferred together. A future collection-specific EPUB delegation/decomposition choice belongs in that mechanism; do not add a separate EPUB config mechanism in advance. No named-profile system, policy inheritance, or generalized configuration merge semantics are required now.

## Notmuch design probe

Use this only to test responsibility boundaries, not to launch a notmuch implementation:

```text
NotmuchMailSource
    ThreadDocument
        MessageDocument
            attachment interpreted by an appropriate DocumentPlugin
```

- Threads and messages do not need registry dispatch. Their bridge to their underlying content is inherent in their implementation.
- A message yields body chunks and attachment subdocuments.
- Ordinary non-attachment MIME parts need not become subdocuments; that would add needless summarization and retrieval levels.
- There is no AttachmentDocument wrapper in the chosen direction.
- AttachmentBridge supplies decoded attachment content to the selected plugin, such as EpubDocument.
- The resulting plugin handler carries the attachment's document identity and collection association.

## EPUB and interpretation versus document structure

Whether chapters should be subdocuments or semantic chunks remains unresolved. The concrete bridge design currently uses a non-chaptered EPUB as its example.

For that case:

```text
EpubDocument (document ID X)
    input: FileBridge or AttachmentBridge
    conversion: pandoc produces text/Markdown
    delegate: TextDocument (same document ID X)
        content access refers to conversion output, not EPUB input
        configured with appropriate boundary regexps
```

The delegated TextDocument is not a child returned by get_sub_documents. Conversion stages do not create extra document identities, hierarchy levels, or summaries. EPUB can directly construct its text delegate without redispatching through the registry.

EpubDocument recommends preserving semantic chunk text in the database for both performance and reproducibility: converter versions may produce different output. The user reports an existing text column but no mechanism to request its population yet. The request mechanism is still to be designed and must be checked against current code.

Text retention is an output-persistence concern, separate from input bridge capabilities. A file-backed EPUB can still need retained output text.

Without caching, the conceptual text access could rerun conversion. Under the latest bridge proposal, conversion runs at most once per bridge instance; a new document acquisition may create a new bridge and rerun it.

### Chapter identity remains open

Discussion identified two different needs:

- Independent document search identity/lifecycle for chapters.
- Agent access to a complete named chapter spanning multiple semantic chunks.

The second could be supported by subdocuments or by a public addressable-section abstraction; merely storing headings in opaque reconstruction metadata is insufficient. Do not assume current MCP/search behavior from earlier conversation: those implementations have changed. No chapter/section interface has been chosen.

## Content access and future bridges

The user proposed simplifying away the earlier stream/seekable/path capability hierarchy and explicit temporary-file adaptation.

All bridges offer:

- `read_bytes`: async full-content read.
- `read_byte_range`: async byte-range read.
- `path`: async context manager exposing a filesystem path.

The implemented range method is `read_byte_range(start, end)`, using half-open byte offsets `[start, end)`. `path()` returns an async context manager.

Handlers request whichever operation they need. Each bridge performs its own adaptation. Under this proposal, handler capability declarations and externally inserted TemporaryFileBridge wrappers are unnecessary for now.

### FileBridge

Represents a real filesystem object, used by FileSource or potentially WikipediaSource.

- Full reads use the existing file.
- Range reads can seek directly without materializing the whole file.
- Path access exposes the existing path without copying it.

This preserves efficient access to large filesystem archives if ZIP support is later implemented.

### AttachmentBridge (future)

Likely takes already-decoded bytes at construction.

- Full reads return the held bytes.
- Range reads slice those bytes.
- Path access writes a temporary file, e.g. using NamedTemporaryFile, and scopes its availability appropriately.

Memoizing base64-decoded attachment content is desirable; decoding need not recur for each range or path access.

### StreamBridge (future)

Has/takes a `stream_producer` coroutine. Its exact return type is not yet specified.

- Production is deferred until the first full read, range read, or path request.
- The producer callback is called at most once per bridge instance.
- Produced content is retained and reused for subsequent operations.
- Path requests materialize the content into a temporary file.

Pandoc command output is a motivating producer. Earlier names included StdoutBridge and CommandBridge; a command-specific wrapper may still be useful, but no separate class is required by the current discussion.

The earlier reopening discussion established that recreating command output depends on reacquiring its input. All contemplated inputs can be reopened, although that may be expensive. The latest memoizing design avoids repeated production within one bridge; it does not promise identical output across new bridge instances.

### Accepted scope/performance assumption

The user explicitly accepts keeping content in memory plus swap for the initial design, consistent with other full-text handling in the project. Do not reintroduce a complex streaming/capability system just to solve hypothetical large nested archives.

Large real-file ZIPs can remain file-backed through FileBridge.path. Nested ZIPs are already expensive and are not an implementation target now. Optimize further when concrete needs justify it.

## Bridge details suggested but not yet resolved

The assistant suggested these semantics after the user's bridge proposal. They are considerations for the next design pass, not all independently accepted decisions:

- Concurrent first accesses share one production rather than invoking the callback twice.
- Decide whether at-most-once includes failed production, with retries requiring a new bridge.
- Specify cancellation behavior for shared in-flight production and cleanup.
- For command producers, check successful process exit before publishing cached output; partial stdout must not silently become successful content.
- Define temporary-path lifetime and overlapping path contexts. Creating a temporary file per context is a possible initial implementation; persistent temporary-file memoization is optional.
- Range offsets address the particular bridge's bytes. Text character offsets are not automatically byte offsets, especially for UTF-8.
- FileBridge is not inherently a snapshot if the underlying file changes; decide what stability the surrounding acquisition assumes rather than silently copying every file.

## Implemented baseline and invariants

The first structural migration is complete: DocumentHandle, async factory acquisition, ContentBridge/FileBridge, text interpretation in TextDocument, and WikipediaDocument specialization. Document summaries are handle operations; source-level fetch_chunk remains an async delegating wrapper. TextFileSource still exists in this baseline and is removed by the next spike.

- Handles remain usable independently of acquisition and enumeration. No mandatory close/aclose or acquisition context is introduced.
- FileBridge opens/closes files within reads; its path context exposes the existing file without copying or owning it.
- Text reads use bytes plus decoding, preserving newline conventions and reconstruction byte offsets. Character offsets remain distinct.
- Missing documents remain distinct from unavailable sources: maintenance deletes on DocumentNotFoundError, not on a source outage.
- Current handles yield no subdocuments; actual recursive indexing and reconciliation remain future work.
- Only save_text=False is supported. Retained-text write/read paths remain future work.
- The parent-document schema exists; successful-rescan reconciliation is not thereby implemented.

## Next implementation spike: DocumentPlugin and code-defined discovery

This is the next proposed code change, not authorization to implement it during this documentation update. Resolve the immediate questions below before implementation.

### Implementation scope

1. Add DocumentPlugin as a DocumentHandle subclass with the common constructor and a separate handler-name registry. Preserve the general DocumentHandle interface for handles outside discovery.
2. Store the bridge and policy; shallow-copy incoming metadata and return a fresh shallow copy from get_metadata. Provide the metadata behavior needed for documents to retain and enrich source-supplied fields. Keep policy['handler'] intact.
3. Make TextDocument a DocumentPlugin registered as `text`. Read boundary settings from policy, encoding and summary_length from source config, with text configuration moved into FileSourceConfig. Remove the separate text-specific constructor arguments.
4. Introduce code-defined extension-to-policy discovery. FileSource supplies identity, FileBridge, mtime, and initial metadata and constructs the selected plugin through its registry. Enumeration and direct fetch must select consistently.
5. Remove TextFileSource/TextSource and its now-redundant construction layer. Register FileSource as `file` and update text-source configuration references to `source_config.type="file"`.
6. Adapt WikipediaSource to FileSource and WikipediaDocument to the common plugin contract. Preserve native search, metadata/keywords, source filtering, indexing defaults, boundaries, and lead summaries. WikipediaSource directly constructs its document with the specialized policy, so ordinary FileSource discovery does not acquire Wikipedia's Xapian dependency.
7. Update callers, imports, config examples, and tests to the new contract. Do not preserve superseded APIs merely to satisfy old tests. Keep architecture.md untouched.

### Verification

- Exercise all five mapped extensions plus unknown and extensionless files; verify shared policies for aliases, text fallback, and consistent direct fetch versus enumeration. Boundary-rule behavior (for example, Markdown/HTML level-1/2 boundaries excluding deeper headings and the RST adornment heuristic) is settled by the policy definitions and was spot-checked with a script; it is not encoded in the test suite.
- Construct two documents from a reused policy and incoming metadata. Verify top-level metadata isolation on construction and on repeated get_metadata calls, with the documented shallow-copy semantics, and that construction leaves the policy and handler key unchanged.
- Exercise a TextDocument with a test-only non-file ContentBridge and supplied policy, proving text interpretation does not require filesystem access.
- Verify encoding and summary-setting access through the agreed config contract, plus byte-range round trips for multibyte text and LF/CRLF/CR content.
- Preserve Wikipedia metadata, lead-summary, native-search, and missing-document versus source-outage tests.
- Test the chosen unknown-extension, missing-handler, and source-registration behaviors explicitly.
- Run the full pytest suite, audit the diff against this plan, and report deviations before declaring completion.

### Implementation decisions (settled during implementation)

- Extension lookup normalizes uppercase suffixes (``path.suffix.lower()``). The embedding-boundary and plain-text fallback defaults are as settled above.
- The constructor's ``last_modified`` argument wins over any ``last_modified`` carried in incoming metadata; ``get_metadata`` reports the handle's own value.
- Plugins register under a ``handler_name`` class attribute (falling back to the lowercased class name, mirroring ``source_prefix``); the first registration wins and a duplicate name does not replace the original.
- Policy is read-only input by convention, not enforced: construction leaves the policy and its ``handler`` key unchanged, and one policy may be shared across documents.
- DocumentPlugin and the handler registry live in a new ``plugins/plugin.py`` module; ``plugins/base.py`` is unchanged.

### Out of scope

No EPUB implementation, new bridges, actual subdocument indexing, retained-text support, MIME detection, collection boundary fields, configurable discovery/policy overrides, or architecture.md rewrite.

## Future lifecycle and hierarchy questions

- How does source-level subdocument resolution route IDs and retain access to nested ancestors?
- Which owner closes preparation resources and delegated handlers if future bridges retain external resources between operations?
- Future child handles must retain enough state to survive enumeration. Temporary paths/processes require scoped cleanup; cached bytes and live resources are different concerns.
- How do retained semantic text write/read paths use save_text=True?
- How are discovery completion and per-child indexing status represented? A document row need not necessarily imply that all its chunks exist.
- How does a successful enumeration reconcile removed children without treating partial discovery as deletion?

## Continuation guidance

Implement the DocumentPlugin/discovery spike only after authorization and resolution of its immediate questions. Keep later format, hierarchy, and configurable-policy work separate. Do not reintroduce mandatory acquisition contexts, capability ladders, AttachmentDocument wrappers, or unbound document-type objects without a concrete need.

Tests pin contracts (policy selection, enumeration-versus-fetch consistency, offset and partition invariants, metadata and policy isolation, registration behavior), not specific boundary rules; rule behavior is settled by the policy definitions and verified by script.
