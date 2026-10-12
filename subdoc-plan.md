# Document plugins and subdocuments

This plan describes the target architecture and the next implementation spike. It is a coherent design, not a chronological record of proposals. Updating it does not authorize implementation. Changes to architecture.md remain a separate task.

## Scope and implemented baseline

DocumentHandle, async document acquisition, ContentBridge/FileBridge, DocumentPlugin and its registry, code-defined extension policies, TextDocument, and WikipediaDocument are implemented. FileSource registers as `file`; the separate TextFileSource layer has been removed. The database already has a parent-document relationship.

Subdocument resolution and indexing, retained-text write/read paths, document metadata persistence, and the handle cache described below are target work, not claims about the current implementation. The current source-level fetch_chunk wrapper is replaced by the shared retrieval helper in this design.

PDF is the next format to design; EPUB remains another motivating format. ZIP and notmuch are responsibility-boundary probes, not implementation requests. Format-specific decisions stay separate from the common infrastructure.

## Identity and hierarchy

Database rows, source methods, and handles use collection-local document IDs. External/tool IDs prefix these with `collection_id:`. The prefix identifies a collection, not a source type.

Literal `:` within a collection-local document ID separates subdocument levels. For example:

```text
collection:foo/bar/baz.pdf:picture_1
```

Here the handle's document_id is `foo/bar/baz.pdf:picture_1`, and its parent's ID is `foo/bar/baz.pdf`. Nested subdocuments add further colon-separated components. Literal colons are not accepted as path-component content; filenames containing colons remain representable with percent encoding (`%3A`). Split hierarchy delimiters before decoding path components. Chunk IDs continue to append `?c=...` to the full collection-local document ID.

The rightmost `rpartition(':')` separates a child from its parent. Sources and handlers assign complete child IDs using the parent identity. The handle-level child-resolution entry point receives the **full collection-local ID**, not merely `picture_1`; the resulting handle retains that full ID and uses it to identify its own descendants. The method name (`fetch_document` or `fetch_sub_document`) is an implementation naming choice, not a different identity contract.

Documents remain in one collection. The parent relationship is `(collection_id, parent_document_id) -> (collection_id, document_id)`. Deleting a document deletes its subtree; successful parent enumeration reconciles removed children and their pending work. Partial or failed enumeration never establishes absence. Reparenting and cross-collection movement are not intended workflows.

## Bound handles, resolution, and caching

A DocumentHandle is a bound document, not an unbound format descriptor. It retains its source, full collection-local document_id, and last_modified. Its operations cover metadata, semantic chunk enumeration/retrieval, document summaries, immediate-subdocument enumeration, and direct child resolution.

- `get_sub_documents` is an async generator yielding DocumentHandles, parallel to `get_chunks`.
- `source.fetch_document(document_id)` resolves roots and nested subdocuments. Resolution can acquire ancestors and delegate to their handle-level resolution entry points.
- Acquisition is an async factory returning a durable handle, not a context manager. Handles remain usable independently of acquisition and enumeration.
- Direct addressability does not promise physically independent or cheap access to a child.
- Missing documents raise DocumentNotFoundError; unavailable sources remain distinct access failures.

Document shapes are combinations of operations, not exclusive leaf/container types. Support chunks plus subdocuments, chunks only, and summary-only documents such as images. A handle may enumerate no chunks or no children; orchestration must not assume every document has textual chunks. Subdocument-only containers also fit the interface. Summary-only documents still receive document summaries and summary embeddings.

### Process-local handle cache

Each source owns an in-memory cache keyed by full collection-local document_id. Fetching an already cached document reuses its handle. Root enumeration, direct acquisition, and subdocument enumeration populate the cache. A public source operation allows child enumeration to register handles; `cache_document(handle)` is a provisional name.

This cache replaces the need for a document-bound chunk-reader object or caller-managed handle caches in retrieval. Text retained in the database can be returned without acquiring a handle at all.

Cache eviction, expiration, resource-cost tuning, and concurrent-miss coordination are deferred until PDF/EPUB preparation requirements are understood. There is no refresh flag, freshness probe, or invalidation protocol required now. Stale handles during a process lifetime are accepted; discovering changes only after process restart is acceptable. This is not a snapshot guarantee for mutable underlying bytes. The existing maintenance path is not a constraint on this design and is not being redesigned here.

## Sources, plugins, and interpretation policy

DocumentSource owns collection-level discovery and resolution. FileSource acquires file-backed content and selects interpretation through code-defined extension policies.

```text
DocumentHandle
    DocumentPlugin
        TextDocument
            WikipediaDocument
        PdfDocument (to be designed)
        EpubDocument (future)
```

DocumentPlugin is a discoverable DocumentHandle consuming a ContentBridge. Its common constructor inputs are `document_id`, `source`, `bridge`, `policy`, `last_modified`, and `metadata`.

- Metadata describes the document. Construction shallow-copies it; each get_metadata call returns a fresh shallow copy. Nested values are not isolated.
- The constructor's last_modified wins over a value in incoming metadata; get_metadata reports the handle's value.
- Policy describes interpretation, including boundary regexps and future decomposition/delegation choices. It is read-only by convention, may be shared, and retains its `handler` key.
- The handler-name registry is separate from extension-to-policy mappings. Registration uses `handler_name`, falling back to the lowercased class name; first registration wins, matching source-registry behavior.
- TextDocument reads encoding and summary_length from the source configuration. These, title, and boundary regexps are not separate plugin constructor arguments.
- A handler that knows a child or delegate's type may construct it directly. Registry dispatch is not a mandatory conversion pipeline.
- WikipediaSource directly constructs WikipediaDocument with specialized policy, preserving native search and article-specific interpretation without adding an Xapian dependency to ordinary FileSource discovery.

### Existing text policies

Extension lookup is case-insensitive. `.md`/`.markdown` share Markdown policy, `.html`/`.htm` share HTML policy, and `.rst` uses RST policy. Unknown extensions and extensionless files use the text fallback. Textual markup is preserved, not converted solely for LLM consumption.

Markdown semantic boundaries use level-1/2 ATX and Setext headings; HTML uses h1/h2. RST recognizes `=` and `-` title adornments as a convention, not inferred document-relative heading levels. The text fallback uses common separator-punctuation lines. Embedding candidates use paragraph breaks for textual formats and paragraph/block endings for HTML. These rules live in policy definitions; tests pin the contracts rather than exact rule-driven chunk counts.

Configurable policy overrides/supplements, collection-wide boundary configuration, MIME detection, named profiles, policy inheritance, and generalized configuration merging remain deferred. Future EPUB configuration belongs in that shared policy mechanism, not a separate advance-built configuration path.

## Document metadata and tool presentation

Persist get_metadata's JSON-compatible contents on Document. Work from a copy, popping title and keywords into their existing dedicated fields when present; retain the remaining metadata in a dictionary. The Python ORM attribute must avoid SQLAlchemy's reserved `metadata` name; `metadata_dict` is an appropriate name.

Tool fetch_document responses and document search results include metadata fields that do not conflict with tool-defined response fields. Tool fields take precedence; metadata must not replace document identity, summaries, chunks, or subdocument information. Metadata persistence supports structured values, not only strings, for EPUB/PDF authorship and other format metadata.

The MCP fetch_document tool exposes direct `sub_documents`, so children are discoverable rather than merely addressable through guessed IDs. Detailed child-entry formatting and binary/image delivery belong to the tool/format implementation design. Summary-only documents remain useful without invented text chunks.

## Chunk text retention and retrieval

Semantic chunks remain the handle-facing retrieval unit. get_chunks yields metadata and candidate texts whose concatenation exactly matches handle.fetch_chunk for that semantic chunk. Embedding chunks partition that text; embedding offsets and lengths are character offsets, distinct from bridge byte offsets.

`DocumentHandle.save_text` controls persistence of each embedding chunk's exact text in `DocumentChunk.text`. This is output retention, independent of input bridge capabilities: a file-backed converted document can still need saved text for reproducibility and to avoid conversion work.

The text column is deferred by default in ORM loads. Most consumers need chunk identity, metadata, lengths, and summaries, not its text.

### Shared retrieval helper

Replace DocumentSource.fetch_chunk with a shared application-level helper usable by search, the indexer, and MCP retrieval. Its intended interface is:

```text
fetch_chunk(source, chunk_id, *, scope="semantic") -> str
```

The helper operates on a stored chunk ID, not a DocumentChunk object or a native-search metadata-only hit. Its module location is an implementation detail; it does not belong to a format plugin.

- Embedding scope explicitly selects the requested embedding row's text.
- Semantic scope selects the embedding rows belonging to the requested row's semantic chunk, ordered for reconstruction.
- Each scope can obtain the relevant text and fallback information in one SQL statement; semantic scope need not first load an ORM row to discover its group.
- If retained text is available, return it without calling fetch_document.
- Otherwise, acquire the document through the source cache and call handle.fetch_chunk with semantic metadata. Embedding scope applies the stored character slice itself.
- A missing row is distinct from a row whose text is NULL. Empty retained text is not the same as NULL. Do not silently assemble incomplete retained text or mix it with a different source version.

Callers already holding a DocumentChunk pass its chunk_id; they need not load its deferred text or carry handles between retrieval calls. Semantic scope works when the supplied chunk ID is not the semantic root, while MCP can retain its existing root-reference reporting behavior.

### Reranker hits

A populated ChunkHit.chunk_id means the chunk is in the database. Reranker preparation sends such IDs to the shared helper. Hits without chunk_id are handled directly by `_prepare_hit`: acquire the handle, fetch the semantic text from hit metadata, and slice the requested embedding span. This path does not require indexing the candidate merely to rerank it.

No native-versus-database origin flag is required. The distinction is whether the hit has a stored chunk identity. Source.fetch_chunk is removed rather than retained as a parallel retrieval API.

## Indexing, durable child work, and failures

Indexing a document processes its own chunks and enumerates its immediate subdocuments. Child modification times cascade from their ancestors, allowing existing timestamp comparisons to detect changed indexed children after fresh acquisition.

The parent document, its own chunk rows, and `indexing_work` entries for discovered child IDs are committed together. A work entry records a full collection-local target document ID and its parent in the same collection. Its target need not have a Document row yet; its parent does. The work table records pending work, not a saved handle or a frozen source revision.

Complete immediate-child enumeration permits reconciliation of removed children and their subtrees. Failure to finish enumeration does not authorize deletion. A parent commit does not claim its entire subtree is indexed, and indexing does not require one whole-subtree transaction.

Child work acquires the document through source.fetch_document, benefiting from the process-local handle cache. On success, commit the child and work for its children while retiring completed work atomically. Interrupted work remains recoverable. Handled failures are recorded in FailedDocument rather than kept in an automatic retry loop. Parents may create indexing_work entries even for failed IDs: index_collection continues to exclude FailedDocument entries when selecting work.

### Scheduling

Process pending indexing_work first when index_collection restarts, then proceed to source discovery. Do not interleave work-table queries into the discovery loop or complicate semaphore handling to implement fairness between the two streams. No same-run queue-draining guarantee is added by this scheduling decision.

### Direct requests for descendants

`ensure_document_indexed` may be asked for a child before its ancestors have database rows, for example through native search or an explicit caller. It indexes from the necessary parent/ancestor down to the requested child, establishing parent rows before child rows. This does not require completing every sibling subtree first; ordinary discovered child work remains represented through indexing_work.

### Consistency scope

Do not add parent generations, subtree revision tokens, or special refresh coordination now. Work refers to an ID resolved through the normal acquisition path; it does not promise an old source version. Disappeared children are reconciled by successful parent enumeration. Existing source-change-during-indexing limitations and process-lifetime handle staleness are accepted. Maintenance redesign is deferred.

## Content bridges and delegation

All ContentBridges provide:

- `read_bytes()`: async full-content read.
- `read_byte_range(start, end)`: async half-open byte-range read.
- `path()`: async context manager exposing a path valid during that context.

Each bridge adapts its content to these operations. No capability ladder, external TemporaryFileBridge adaptation layer, or mandatory handle-wide close/aclose protocol is required.

FileBridge reads real files and exposes their paths without copying. Reads own their file lifetimes. Text reads use bytes followed by decoding, preserving LF, CRLF, and CR byte offsets rather than applying universal-newline translation.

A future AttachmentBridge can hold decoded attachment bytes, serve slices, and materialize a temporary path for scoped access. A future StreamBridge takes a producer coroutine, defers production until first content access, and retains output for reuse; production runs at most once per bridge instance. Pandoc output motivates this design. Exact producer result type, failure/cancellation semantics, concurrent production, and overlapping temporary-path lifetimes remain bridge implementation questions, not additional settled requirements.

Keeping content in memory plus swap is accepted initially. Large real-file archives can remain file-backed; hypothetical nested archives do not justify a more complex streaming architecture now. Future bridges retaining live resources must define cleanup without invalidating handles when enumeration ends.

### EPUB delegation

For a non-chaptered EPUB, EpubDocument consumes its input bridge, converts via pandoc, and delegates text interpretation to a TextDocument with the **same document ID** and an output bridge over the converted text. The delegate receives policy appropriate to that output. It is not a subdocument and adds no hierarchy or summary level. Retained embedding text provides access to the indexed conversion output across later acquisitions or converter changes.

Whether EPUB chapters are subdocuments or semantic chunks remains format-specific. Independent chapter lifecycle and agent access to a whole named chapter are different needs; opaque reconstruction metadata alone does not provide a public chapter retrieval interface.

### Notmuch responsibility probe

```text
NotmuchMailSource
    ThreadDocument
        MessageDocument
            attachment interpreted by a DocumentPlugin
```

Threads/messages need not participate in plugin discovery. A message can yield body chunks and attachment subdocuments. Ordinary non-attachment MIME parts need not become documents. AttachmentBridge supplies decoded content directly to the selected plugin; there is no AttachmentDocument wrapper.

## Next implementation spike: common subdocument infrastructure

This is proposed implementation scope, not authorization to implement during this plan update. PdfDocument design is the next format discussion; no PDF extraction or image policy is implied by this infrastructure spike.

### Scope

1. Implement reserved-colon identity routing, full-ID handle-level child resolution, and the per-source process-local handle cache with a public insertion operation for enumeration.
2. Implement save_text writes, defer DocumentChunk.text by default, and add the chunk-ID-based shared retrieval helper. Replace all source.fetch_chunk consumers, including MCP, indexer reconstruction, and reranker preparation; keep metadata-only hits local to `_prepare_hit`.
3. Persist document metadata separately from title/keywords and expose nonconflicting fields in tool document responses/results. Add direct sub_documents to MCP fetch_document.
4. Add durable indexing_work storage and parent/child orchestration: atomic parent/work publication, successful-enumeration reconciliation, failure exclusion, pending-work-first scheduling on restart, and ensure_document_indexed traversal from missing ancestors to a requested descendant.
5. Support chunk-plus-child, chunk-only, and summary-only handles without format-specific orchestration. Preserve summary embeddings for documents without chunks.
6. Supply schema migration steps for new storage and preserve existing source filtering, text policies, Wikipedia behavior, and byte/character offset distinctions. Inspect current callers and tests before changing signatures. Required base-interface changes belong to this explicit scope and require implementation approval under repository rules.

### Verification

- Establish a green pytest baseline before implementation; use synthetic handles/bridges and pytest monkeypatch rather than requiring a PDF library for common infrastructure tests.
- Verify nested full-ID resolution, root/child cache population and reuse, and encoded-colon paths without confusing collection prefixes, hierarchy separators, or chunk suffixes.
- Verify retained embedding and semantic retrieval (including a non-root embedding ID), deferred ordinary text loading, a single retrieval statement per scope, and no handle acquisition on the retained-text path. Verify unretained fallback, correct character slicing, and missing-row versus NULL-text behavior.
- Verify reranking of stored chunk IDs and metadata-only hits without adding an origin flag or forcing indexing for the latter.
- Verify JSON-compatible metadata persistence, title/keyword separation, response-field precedence, and direct-child visibility through MCP.
- Verify parent/work atomicity, restart recovery, failure exclusion even after work recreation, descendant indexing with missing ancestors, and subtree removal only after complete enumeration. Verify the pending-work-first scheduling contract without interleaved polling.
- Verify all supported document shapes, including summary-only searchability, and preserve newline/multibyte round trips and missing-document versus source-outage distinctions.
- Run the full suite with `~/ai/venv/bin/python -m pytest -q`. Audit the diff against this plan and report consequential choices or deviations rather than silently expanding scope.

### Non-goals and next format decisions

No architecture.md rewrite, maintenance redesign, cache freshness/eviction policy, whole-subtree transaction, generation-token system, configurable policy machinery, ZIP/notmuch implementation, or generalized streaming framework.

PdfDocument design will determine extraction and semantic boundaries, image selection and identity (embedded object versus page occurrence), image summarization, parent-summary use of child summaries, and any image-content delivery. EPUB chapter structure and concrete future-bridge lifecycle details remain deferred to their respective format work.
