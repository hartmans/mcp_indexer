# Subdocument/plugin design checkpoint

This is a discussion checkpoint, not an implementation specification or a claim that the APIs below already exist. It preserves the design direction and unresolved questions for continuing the conversation.

## Context and scope

- The user reports substantial recent changes to MCP, search, and chunking. Discard earlier assumptions about those implementations. Inspect current code when implementation requires it; this checkpoint does not depend on old code shapes.
- The current focus is document handlers, subdocument discovery, and content access—not changing search, MCP retrieval, or chunking now.
- EPUB is the motivating implementation. ZIP and notmuch are design probes; implementing ZipDocument or NotmuchMailSource is not currently requested.
- The VibeVoice discussion in the conversation was an unrelated wrong-window digression.
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

## DocumentPointer becomes DocumentHandle

The chosen name for the initial migration is **DocumentHandle**, replacing DocumentPointer with little interface change. Earlier discussion called this object DocumentHandler; references to handlers below describe the same bound-document role, not an additional wrapper class.

A handler is a bound document object: for example, EpubDocument represents a particular EPUB, not the abstract EPUB format. There is no present need for a separate unbound EpubDocumentType object. The registry can select classes or factories without introducing such an object.

DocumentHandler retains document identity and association with the originating DocumentSource/collection. Its document-facing operations include metadata, chunk enumeration/retrieval, and subdocument enumeration. Exact signatures beyond the agreed shapes below are not settled.

- `get_sub_documents` is an async generator yielding DocumentHandlers, parallel to `get_chunks`.
- The indexer calls both chunk and subdocument enumeration. Do not enforce exclusive leaf/container categories in orchestration or the base interface.
- Most initial formats may yield only chunks or only subdocuments, but messages are already a concrete case yielding both.
- `source.fetch_document(subdocument_id)` must resolve subdocuments, not just roots.
- `fetch_document` is an **async factory returning a DocumentHandle**, not a context manager: `handle = await source.fetch_document(document_id)`. Async acquisition and context-bound lifetime are separate decisions. Resolution may require expensive asynchronous preparation or existence checks; returned handles remain usable independently of the acquisition call.

Direct subdocument resolution may internally reopen and prepare ancestors. Independent addressability does not promise cheap or physically independent access.

## Sources, plugins, and registry

DocumentSource owns collection-level discovery and resolution. FileSource remains a source. The former TextSource concept becomes TextDocument (earlier called TextDocumentType); EPUB and PDF are other examples of content interpreters.

A provisional `DocumentPlugin` or `RegisteredDocumentHandler` is the handler construction contract at which ContentBridges matter. The exact class name and inheritance arrangement remain open.

- Registered handlers consume a ContentBridge.
- Registry dispatch selects a handler for unknown content types.
- FileSource is expected to select using filename extension and/or MIME detection.
- Some handlers know their child type directly and construct it without registry lookup.
- Registry selection is not a mandatory multistage conversion pipeline. Reusable transformations can be utilities or direct delegation.
- Registry configuration, precedence, and source-specific customization have not been settled.

An earlier possibility was a shared enumeration/resolution abstraction between DocumentSource and a bound container handler. It may still be useful, but the latest design has not committed to an additional nested-source class or shared base interface.

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

## Latest ContentBridge proposal

The user proposed simplifying away the earlier stream/seekable/path capability hierarchy and explicit temporary-file adaptation.

All bridges offer:

- `read_bytes`: async full-content read.
- `read_byte_range`: async byte-range read.
- `path`: async context manager exposing a filesystem path.

Exact signatures and range bounds have not been approved. A half-open `[start, end)` range was suggested by the assistant, not separately decided by the user.

Handlers request whichever operation they need. Each bridge performs its own adaptation. Under this proposal, handler capability declarations and externally inserted TemporaryFileBridge wrappers are unnecessary for now.

### FileBridge

Represents a real filesystem object, used by FileSource or potentially WikipediaSource.

- Full reads use the existing file.
- Range reads can seek directly without materializing the whole file.
- Path access exposes the existing path without copying it.

This preserves efficient access to large filesystem archives if ZIP support is later implemented.

### AttachmentBridge

Likely takes already-decoded bytes at construction.

- Full reads return the held bytes.
- Range reads slice those bytes.
- Path access writes a temporary file, e.g. using NamedTemporaryFile, and scopes its availability appropriately.

Memoizing base64-decoded attachment content is desirable; decoding need not recur for each range or path access.

### StreamBridge

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

## Simpler initial migration: preserve existing functionality

This section records the latest, narrower implementation direction and supersedes earlier context-scoped handle proposals. Implementation has not been authorized by this documentation update.

### Classes and responsibility changes

- Rename DocumentPointer to DocumentHandle with little interface change.
- Remove the current FileSourcePointer (previously referred to in discussion as FileSourceDocumentPointer). Move its text-reading/chunk-retrieval behavior into TextDocument and its byte-access behavior into FileBridge.
- TextSource/TextFileSource's text interpretation becomes TextDocument. Move `split_text` and related text interpretation there.
- Keep Wikipedia-specific behavior in **WikipediaDocument, a TextDocument subclass**. Preserve article metadata, keywords, boundary behavior, and introductory-lead summary.
- Move the source-level document summary operation to DocumentHandle. TextDocument provides the current text summary behavior; WikipediaDocument provides the Wikipedia override. Exact method naming remains to be settled.
- Introduce ContentBridge with async `read_bytes`, async `read_byte_range`, and async-context-managed `path` as described above.
- Implement **FileBridge only** for now. It likely takes a source and path. AttachmentBridge, StreamBridge, EPUB, ZIP, and notmuch remain future work/design examples.
- TextDocument likely takes a source, document_id, and ContentBridge. How last_modified and initial source metadata enter that construction still needs a concrete decision.
- Provide empty `get_sub_documents` behavior for current handles. Do not add actual subdocument indexing, format detection, or registry expansion as part of this structural migration.
- A `save_text` property (or similar name) may be introduced, but **only False is supported in this migration**. Enabling retention requires separate write/read-path work; the existing nullable text column alone is not sufficient.

### Async factory, not context-bound acquisition

Use:

```python
handle = await source.fetch_document(document_id)
```

Do not use `async with source.fetch_document(...)`. Acquisition may do asynchronous work before returning. For example, resolving a future ID such as `bar/baz/foo.epub:chapter_23144` may require preparation to establish whether that chapter exists. This example motivates the async contract; it does not settle the subdocument ID syntax or routing implementation.

Returned handles own the references needed for later access. Their lifetime is not bounded by the fetch call. Discovery can continue batching handles, and indexing tasks can retain them. The earlier recommendation to rewrite scheduling around IDs solely to avoid closed handles is therefore unnecessary for this migration.

All current `fetch_document` callers need `await`, including enumeration, maintenance, indexing-on-demand, and source chunk wrappers. Move summary callers to the handle operation. The production callers inspected during discussion were already async; the synchronous Wikipedia Xapian worker emits IDs rather than acquiring handles, so no sync-to-async workaround was identified there.

### Resource lifetime

Distinguish handle lifetime, cached bytes, and active I/O resources:

- Handles/bridges may remain usable while referenced; do not impose mandatory context-scoped use.
- FileBridge opens and closes files inside read operations. Its path context exposes the existing file without owning/deleting it.
- Future temporary paths and command processes need cleanup scoped to the operations that use them, not merely eventual garbage collection.
- Cached bytes can be reclaimed when their owners become unreachable. `gc.collect()` cannot evict bytes from a still-referenced handle; explicit cache eviction would be a separate policy.
- A close/aclose method and optional context-manager convenience were discussed, but are **not required for the initial FileBridge-only migration**. Reconsider if an implementation retains external resources between operations.
- Future enumeration-produced child handles must retain enough content-access state to survive enumeration if following this durable-handle model. Exact recursive ownership and producer cancellation behavior remain future design work, not a reason to context-bind current file handles.

### Preserve behavior and defer configuration redesign

- `get_chunks` reads bytes and decodes them, avoiding `read_text` universal-newline translation. Byte metadata must continue to address the same underlying bytes; decoded character offsets remain distinct.
- **Defer configuration restructuring.** Encoding, boundary regexps, summary limits, source registration, and the current `text` source prefix have real coupling, but the user explicitly wants that design postponed. Preserve existing configuration behavior with minimal plumbing; do not silently remove current source registrations or introduce a new configuration schema.
- Keep source-specific behavior working while moving interpretation; deferring configuration redesign is not permission to drop Wikipedia metadata or summaries.
- Preserve missing-document versus unavailable-source errors. Maintenance deletes records on DocumentNotFoundError; filesystem failures must still check source availability rather than being misclassified as deletion. FileBridge/handle construction must preserve this distinction for stat and reads.
- Async bridge methods do not automatically make filesystem I/O nonblocking. Thread offloading, if needed, is an implementation choice rather than a new public API requirement.
- Existing source `fetch_chunk` can remain an async wrapper that awaits a handle and delegates. Repeated opens in whole-document retrieval and semantic reconstruction are acceptable for FileBridge; later expensive bridges may motivate reusing a handle across those loops.
- Ignore test compatibility as a design constraint at this stage. The user is willing to rewrite tests for the new interfaces; this is not an instruction to skip verification when implementation is eventually requested.

## Other open lifecycle questions

- How does source-level subdocument resolution route IDs and retain access to nested ancestors?
- Which owner closes preparation resources and delegated handlers if future bridges retain external resources between operations?
- What metadata comes from the source versus the format handler, and how is it combined?
- How does a handler request database retention of semantic text when save_text=True is implemented later?
- How are discovery completion and per-child indexing status represented? The user is open to relaxing the invariant that a document row implies all its chunks exist.
- How does a successful enumeration reconcile removed children without treating partial discovery as deletion?

## Continuation guidance

Continue from the simpler initial migration above: DocumentHandle, async factory fetch_document, TextDocument/WikipediaDocument, and FileBridge only. Do not reintroduce mandatory acquisition contexts, the older capability hierarchy, AttachmentDocument, or unbound type objects without a concrete need. Configuration redesign and actual new format/subdocument functionality are deferred. Reinspect current code before implementation. This checkpoint authorizes documentation only, not code changes.
