# Document plugins and subdocuments

This plan describes the target architecture and the next implementation spike. It is a coherent design, not a chronological record of proposals. Updating it does not authorize implementation. Changes to architecture.md remain a separate task.

## Scope and implemented baseline

DocumentHandle, async document acquisition, ContentBridge/FileBridge, DocumentPlugin and its registry, code-defined extension policies, TextDocument, and WikipediaDocument are implemented. FileSource registers as `file`; the separate TextFileSource layer has been removed. The database already has a parent-document relationship.

Subdocument resolution and indexing, retained-text write/read paths, document metadata persistence, and the handle cache described below are target work, not claims about the current implementation. The current source-level fetch_chunk wrapper is replaced by the shared retrieval helper in this design.

DoclingDocument handles PDF and selected office formats, with ImageDocument children for extracted figures. These plugins and bottom-up tree summarization are part of the next spike. EPUB remains another motivating format; ZIP and notmuch remain responsibility-boundary probes, not implementation requests.

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

Cache entries carry a monotonic last-used timestamp, updated on insertion and cache hits. When cache_document is called and no expiration handle is registered, the source schedules an expiration callback for 300 seconds later. The callback evicts entries unused for at least 300 seconds and registers another callback only if the cache remains nonempty. There is at most one scheduled expiration callback per source. The interval is overridable for tests. Eviction releases cache references; it does not invalidate handles still held by running work.

This is idle expiration for memory management, not source-freshness validation. There is no refresh flag, freshness probe, or invalidation protocol required now. Stale handles are accepted; discovering changes only after process restart remains acceptable. This is not a snapshot guarantee for mutable underlying bytes. The existing maintenance path is not a constraint on this design and is not being redesigned here. Further resource-cost tuning and concurrent-miss coordination remain implementation considerations.

## Sources, plugins, and interpretation policy

DocumentSource owns collection-level discovery and resolution. FileSource acquires file-backed content and selects interpretation through code-defined extension policies.

```text
DocumentHandle
    DocumentPlugin
        TextDocument
            WikipediaDocument
        DoclingDocument
        ImageDocument
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

Process pending indexing_work first when index_collection restarts, then proceed to source discovery. After its own commit, index_document schedules its enumerated subdocuments through the existing semaphore-limited task machinery. This uses the already-enumerated children, not work-table polling inside the discovery loop, and allows reuse of prepared document content while handles are available. Durable work entries remain the recovery mechanism if the process stops between commit and scheduling or completion. Failure exclusion applies to post-commit scheduling as well as restart work.

Parents schedule children without waiting for child slot acquisition or completion while holding their own semaphore slot. Descendant work participates in the existing task/idle tracking; do not introduce a separate child concurrency pool.

### Direct requests for descendants

`ensure_document_indexed` may be asked for a child before its ancestors have database rows, for example through native search or an explicit caller. It indexes from the necessary parent/ancestor down to the requested child, establishing parent rows before child rows. This does not require completing every sibling subtree first; ordinary discovered child work remains represented through indexing_work.

### Consistency scope

Do not add parent generations, subtree revision tokens, or special refresh coordination now. Work refers to an ID resolved through the normal acquisition path; it does not promise an old source version. Disappeared children are reconciled by successful parent enumeration. Existing source-change-during-indexing limitations and process-lifetime handle staleness are accepted. Maintenance redesign is deferred.

## Bottom-up document summarization

Document summarization waits until the tree is ready: descendant work has completed or failed, and available child summaries have been produced. Summaries are generated bottom-up, not provisionally regenerated on every child completion. Waiting for readiness must not hold a semaphore slot needed by descendants. Failed work follows the general failure rules rather than causing automatic retries or an indefinite readiness wait.

Generic document summarization incorporates immediate-child summaries along with the document’s own available textual inputs. Prompt sections distinguish:

- Full document text, when supplied (label excerpts as excerpts rather than full text).
- Summaries of parts of the document: chunk summaries.
- Summaries of subdocuments within the document: each child document_id paired with its summary.

Use clear section boundaries and omit unavailable input sections. Descendant information propagates through immediate-child summaries rather than flattening the whole tree into every ancestor prompt. A handle-provided summary must not bypass child-summary incorporation for a parent with subdocuments. ImageDocument remains a summary-only leaf using get_document_summary. Partial conversion and incomplete enumeration follow the same generic indexing/failure rules as other documents; there is no separate Docling-specific partial-conversion policy.

## Content bridges and delegation

All ContentBridges provide:

- `read_bytes()`: async full-content read.
- `read_byte_range(start, end)`: async half-open byte-range read.
- `path()`: async context manager exposing a path valid during that context.

Each bridge adapts its content to these operations. No capability ladder, external TemporaryFileBridge adaptation layer, or mandatory handle-wide close/aclose protocol is required.

FileBridge reads real files and exposes their paths without copying. Reads own their file lifetimes. Text reads use bytes followed by decoding, preserving LF, CRLF, and CR byte offsets rather than applying universal-newline translation.

A future AttachmentBridge can hold decoded attachment bytes, serve slices, and materialize a temporary path for scoped access. A future StreamBridge takes a producer coroutine, defers production until first content access, and retains output for reuse; production runs at most once per bridge instance. Pandoc output motivates this design. Exact producer result type, failure/cancellation semantics, concurrent production, and overlapping temporary-path lifetimes remain bridge implementation questions, not additional settled requirements.

Keeping content in memory plus swap is accepted initially. Large real-file archives can remain file-backed; hypothetical nested archives do not justify a more complex streaming architecture now. Future bridges retaining live resources must define cleanup without invalidating handles when enumeration ends.

### DoclingDocument and ImageDocument

`plugins/docling.py` defines our DoclingDocument plugin (distinct from Docling's own result type). Register it for `.pdf`, `.docx`, `.odt`, `.ods`, `.ppt`, `.pptx`, `.doc`, and `.rtf` through the existing extension-policy mechanism.

DoclingDocument consumes a ContentBridge and retains the processed representation in memory. Enable OCR using RapidOCR with its **PyTorch backend**, retain extracted figure images, and do not retain page images. Internal page rendering for OCR/extraction is not prohibited. Do not use Docling's vision-model enrichment/description support. Backend-specific extraction details are implementation choices; no separate PDF chunker is introduced.

Export Docling's Markdown and expose it through a bridge to a TextDocument delegate using the existing Markdown policy. The delegate has the same document_id as its parent plugin and is not a subdocument. DoclingDocument delegates chunk enumeration/retrieval to it and sets save_text=True. If straightforward serializer hooks or reference mapping allow it, make image references identify the image subdocuments; otherwise retain Docling's ordinary Markdown output. Do not build fragile placeholder matching or custom Markdown reconstruction. Any substitution precedes chunking so offsets and retained text address the same representation.

Extracted figures become ImageDocument children. Choose IDs with best-effort stability across repeated processing, but occasional rerendering-induced subdocument churn is acceptable. Exact image identity/selection details are left to implementation and reported with the result.

`plugins/images.py` defines ImageDocument, consuming a ContentBridge for image content. It supplies `get_document_summary` using the existing image-capable LLM path, and yields no chunks or subdocuments. It sets save_text=True, although this has no effect without chunk rows; its document summary is persisted normally. No LLM response cache or separate image-model routing system is introduced. Child summaries feed generic bottom-up parent summarization.

### Optional dependency and regression samples

Docling is an optional dependency installed via `mcp_indexer[docling]`, including the dependencies needed for the selected OCR backend. The extension mappings and plugin registration remain available without Docling; instantiating DoclingDocument then raises an informative ImportError rather than falling back to text or breaking unrelated plugin imports.

Real conversion regression tests cover `.doc`, `.docx`, and `.pdf`. Short sample files may be checked into the repository under `tests/`. Skip Docling-dependent conversion tests when Docling is unavailable; keep generic infrastructure tests and missing-dependency behavior testable without it. Installed-but-broken conversion is not silently treated as an absent optional dependency. Model provisioning/download requirements must be reported during verification rather than hidden by fabricated results or broad skips.

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

## Next implementation spike: subdocuments, Docling, and image summaries

This is proposed implementation scope, not authorization to implement during this plan update. Implement the common infrastructure and the Docling/ImageDocument integration described above as one verified spike; retain the distinction between general orchestration and format interpretation.

### Scope

1. Implement reserved-colon identity routing, full-ID handle-level child resolution, and the per-source process-local handle cache with public insertion, last-used tracking, and a single recurring 300-second idle-expiration callback. Make the interval overridable for tests.
2. Implement save_text writes, defer DocumentChunk.text by default, and add the chunk-ID-based shared retrieval helper. Replace all source.fetch_chunk consumers, including MCP, indexer reconstruction, and reranker preparation; keep metadata-only hits local to `_prepare_hit`.
3. Persist document metadata separately from title/keywords and expose nonconflicting fields in tool document responses/results. Add direct sub_documents to MCP fetch_document.
4. Add durable indexing_work storage and parent/child orchestration: atomic parent/work publication, successful-enumeration reconciliation, failure exclusion, pending-work-first scheduling on restart, post-commit scheduling of enumerated children through the existing semaphore pool, and ensure_document_indexed traversal from missing ancestors to a requested descendant.
5. Support chunk-plus-child, chunk-only, and summary-only handles without format-specific orchestration. Implement tree-readiness gating and bottom-up document summarization with distinct text, chunk-summary, and child-ID/summary sections. Preserve summary embeddings for documents without chunks; readiness waits must not occupy semaphore slots.
6. Add plugins/docling.py and plugins/images.py, bridge adapters needed for converted Markdown and figure content, Markdown delegation, and all eight agreed extension mappings. Use OCR with RapidOCR/PyTorch, extracted figures without retained page images, and the existing image-capable LLM rather than Docling vision enrichment. Set save_text=True on both plugins. Choose best-effort stable image IDs and substitute Markdown image references only if straightforward.
7. Add the docling optional extra, lazy dependency failure on plugin instantiation, and short real .doc/.docx/.pdf regression samples under tests/. Skip conversion tests only for absent Docling.
8. Supply schema migration steps for new storage and preserve existing source filtering, text policies, Wikipedia behavior, and byte/character offset distinctions. Inspect current callers and tests before changing signatures. Required base-interface changes belong to this explicit scope and require implementation approval under repository rules.

### Verification

- Establish a green pytest baseline before implementation; use synthetic handles/bridges and pytest monkeypatch rather than requiring a PDF library for common infrastructure tests.
- Verify nested full-ID resolution, root/child cache population and reuse, and encoded-colon paths without confusing collection prefixes, hierarchy separators, or chunk suffixes. With a shortened interval, verify last-use extension, idle eviction, only one scheduled callback, rescheduling for a nonempty cache, stopping when empty, and continued usability of handles retained outside the cache.
- Verify retained embedding and semantic retrieval (including a non-root embedding ID), deferred ordinary text loading, a single retrieval statement per scope, and no handle acquisition on the retained-text path. Verify unretained fallback, correct character slicing, and missing-row versus NULL-text behavior.
- Verify reranking of stored chunk IDs and metadata-only hits without adding an origin flag or forcing indexing for the latter.
- Verify JSON-compatible metadata persistence, title/keyword separation, response-field precedence, and direct-child visibility through MCP.
- Verify parent/work atomicity, restart recovery, failure exclusion even after work recreation, descendant indexing with missing ancestors, and subtree removal only after complete enumeration. Verify pending-work-first restart handling and post-commit child scheduling without interleaved polling. Exercise nested work with a single semaphore slot to catch parent/child deadlocks, and verify idle tracking includes scheduled descendants.
- Verify all supported document shapes, including summary-only searchability, and preserve newline/multibyte round trips and missing-document versus source-outage distinctions.
- Verify bottom-up summary ordering, child readiness/failure handling, and labeled prompt sections containing immediate-child IDs and summaries. Ensure a parent is not finalized before its available child summaries and that waiting does not occupy a semaphore slot.
- Run real .doc/.docx/.pdf conversion using checked-in short fixtures; verify Markdown delegation and retrieval round trips, metadata, and extracted image children where present. Include PDF coverage that exercises OCR and figure extraction, not merely parsing an existing text layer. Verify repeat processing yields consistent enumeration/direct lookup and document the chosen image-ID strategy without promising cross-render identity stability.
- Verify all agreed extension mappings, import/registration without Docling, informative instantiation failure, and conversion-test skipping when the optional dependency is absent. Verify ImageDocument sends bridge image content through the existing LLM path and creates a document summary without chunks.
- Run the full suite with `~/ai/venv/bin/python -m pytest -q`. Audit the diff against this plan and report consequential choices or deviations rather than silently expanding scope.

### Settled implementation decisions

Recorded as the spike closed. These settle the items the plan left to implementation, reported with the result; they do not reopen settled architecture.

- Figure child IDs are `fig-<reading-order index>-<sha1 of the PNG bytes, first 12 hex chars>`. Best-effort stable across repeat processing of the same document; rerender-induced churn is accepted as an ID change, and no cross-render identity stability is promised.
- Markdown image-reference substitution uses `export_to_markdown(image_mode="embedded")`: the emitted data-URI references are matched to figure children in document order and, only when the count exactly equals the figure count, each is replaced with `![Figure N](<child document id>)`. On a count mismatch the Markdown is retained exactly as produced. No placeholder matching or custom reconstruction.
- Docling pipeline options must reach the converter as `DocumentConverter(format_options={InputFormat.X: XFormatOption(pipeline_options=opts)})`; a separately constructed options object is silently ignored. PDF options: RapidOCR/torch OCR, `generate_picture_images=True`, `generate_page_images=False`, no vision-model enrichment. Office formats use their per-format `FormatOption`; `.doc` converts natively (no `soffice` at runtime — `soffice` is used only to produce the checked-in `.doc` fixture).
- Real conversion fixtures are short files checked in under `tests/fixtures/` (`sample.pdf` with a text page, a figure, and a full-page bitmap scan to exercise OCR; `sample.docx`; `sample.doc`).
- Summarization convergence is event-driven rather than pumped: completion triggers schedule the parent's document summary when a child's work lands (index commit, chunk summarization, child document summary, and failure record all trigger), `summarize_documents_collection` performs one catch-up scan for work completed in an earlier process, and `summarize_results` converges the full tree bottom-up in a single pass, joining in-flight runs. Pumped schedule-ready → wait_for_idle rounds were rejected: pending work rows decrement only through completion or failure, and a stuck task would loop indefinitely. `summarize_results` does not interact with the idle mechanism, and duplicate work it may schedule is accepted.
- Pending `indexing_work` is processed in batches of at most 256 rows at the start of `index_collection`, each batch fully awaited before rescan; every row seen at the start of an iteration retires by the end of its batch (index commit or failure record), so the pass is bounded by the trees it recovers. A task that neither completes nor fails is not detected — an accepted limitation of the shared task machinery, not handled by this spike.
- Pass-level document summarization is gated by a flag set at the start of the indexing pass, so `index_all(summarize_documents=False)` stays silent about document summaries even while its indexing and chunk work completes.
- Test hygiene: tests that end with trigger-scheduled summary tasks settle them with `wait_for_idle` before fixture teardown (a summary task running against a torn-down database surfaces as a teardown error, not a test failure).

### Non-goals and implementation discretion

No architecture.md rewrite, maintenance redesign, freshness-validation protocol, whole-subtree transaction, generation-token system, configurable policy machinery, LLM response cache, separate image-model routing, Docling vision enrichment, ZIP/notmuch implementation, or generalized streaming framework. Idle cache expiration is in scope; more elaborate memory budgeting is not.

Image identity/selection, straightforward Markdown reference adaptation, and small backend integration details are implementation choices with best-effort stability; report them rather than reopening settled architecture. Do not add a Docling-specific partial-conversion policy in place of general failure handling. Binary image delivery through tools, EPUB chapter structure, and unrelated future-bridge lifecycle details remain deferred.
