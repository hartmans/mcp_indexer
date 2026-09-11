# Goal

This is a server than indexes documents in collections and provides a set of mcp tools for each collection to search and fetch documents.

## DocumentSource

A DocumentSource represents a strategy for indexing documents as well as a retrieval strategy. For example one source might handle spreadsheets and another source might handle text files that could be converted to markdown.

### DocumentPointers

There is a generic DocumentPointer type that defines operations on a document and includes metadata.
DocumentSources need to subclass this type.


### Document and Chunk Identifiers

A document_id is a hierarchical identifier: *source*:*collection_id*:collection_specific/path/and/other/info.

* The *source* identifies which *DocumentSource* implementation is used.

* The *collection_id* is a collection from the config.
* The remainder is whatever the source needs to find a document.

Chunk IDs are formed by adding `?c=0` and so on to a document id.
### Chunking

Chunking happens at multiple levels.
A semantic chunk is an appropriate granularity for summarization and for retrieval for tasks where in-depth understanding is desired.

Embedding chunks are appropriately sized for vector embeddings.

*DocumentChunk* is a model stored in the database for each document chunk. It contains a metadata dictionary used to actually retrieve chunks; that can be passed into DocumentPointer.fetch_chunk to get the semantic chunk. *DocumentSource.fetch_chunk* can get semantic or embedding chunks.
* *DocumentPointer.get_chunks* describes the chunk metadata dictionary in detail.


## Collections

A collection  is an instance of a DocumentSource configured for a given set of documents. For example pointing the TextSource at a directory of stories could form a story collection.

Collection configuration includes:

* DocumentSource specific configuration (directory, chunking config, etc)
* A MCP tool prefix
*  Description of the collection to include in the mcp tool schemas
* A prompt to describe what should be included in the summary of a document and document chunks.

## Context

The context includes:

* Access to LangChain chat models for summarization
* Access to the configured embedding model
* PostgreSQL/pgvector sessions
* A shared reranker assigned to collections with reranking enabled. Context
  construction does not load model weights; the first search requiring scoring
  calls setup. The instance's semaphore coordinates scoring across collections.

## Search

Search orchestration lives in `search.py`; MCP and CLI consumers call it directly.
Sources may override `DocumentSource.native_search` to search a corpus that has
not been embedded in advance. Source-independent vector search always searches
the indexed subset of the collection. Native and vector search coexist.

Discovery returns detached `DocumentHit` and `ChunkHit` values, not ORM objects.
Document hits carry a document ID, score, and optional summary. Chunk hits carry
a document ID, retrieval metadata, score, and an optional stored chunk ID. The
metadata follows the existing semantic/embedding retrieval conventions. Native
hits can exist before their database records do. Each mechanism ranks document
and chunk hits separately; scores are not comparable between lists.

Without reranking, take up to the result limit from each document/chunk list,
vector lists first, then native lists. This can yield four times the limit
before grouping. With reranking, combine candidate lists and score their text,
selecting the requested number of hits before grouping. Candidate overfetch
defaults are centralized in search, not in MCP tools. A separate optional
rerank query is available to Python callers.

Document scoring uses the supplied hit summary when available. Otherwise it
uses a stored summary, or indexes the document and obtains a summary through
the existing source-summary interface and generic summarization fallback.
Chunk scoring retrieves embedding-chunk text through the source. Reranking
preparation may index candidates that ultimately are not selected; this work
is independent of result assembly.

Assembly processes selected hits only. Missing documents are fully chunked and
embedded; persisted documents are not reindexed by search. If a document has
indexed chunks, its complete set of chunks is stored. Embeddings remain
non-null. Semantic chunks and their summary spans retain their existing meaning.

Missing summaries are filled, with the source's document-summary interface
tried first. Hit summaries do not need a separate persistence channel: sources
that provide summaries can return them through that existing interface.
Existing document summaries are preserved. The current generic fallback can
summarize all spans to produce a document summary.

Response ORM objects are loaded in fresh sessions after summary writes finish.
Assembly groups by first occurrence of each document, merges relevant chunks,
and removes duplicate chunk references. Failures are logged and other usable
results are returned; reranker failure falls back to initial candidate order.
Search does not add concurrency coordination for indexing or summarization.

## Configuration

Configuration is managed via a TOML file and supports a hierarchical structure:

1.  **Server Config**: Global settings for the server (e.g., database URI).
2.  **Infrastructure Defaults**: Global defaults for collection-level settings (e.g., summary prompts, embedding models).
3.  **Collection Config**: Specific settings for each collection. This includes:
    *   **Inherited Infra**: Values inherited from Infrastructure Defaults, which can be overridden at the collection level.
    *   **Required Fields**: Settings that must be defined per collection (e.g., MCP tool prefix).
    *   **Source Config**: A plugin-specific configuration blob. The `DocumentSource` plugin is responsible for validating this blob into its own schema.

*ConfigManager* is a class that collects all the configuration together.
