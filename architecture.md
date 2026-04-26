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
**### Chunking

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

* Access to LangChain chait models for summarization
* Access to the appropriate LanceDb embedding function for embeddings
* The LanceDb connection

## Configuration

Configuration is managed via a TOML file and supports a hierarchical structure:

1.  **Server Config**: Global settings for the server (e.g., database URI).
2.  **Infrastructure Defaults**: Global defaults for collection-level settings (e.g., summary prompts, embedding models).
3.  **Collection Config**: Specific settings for each collection. This includes:
    *   **Inherited Infra**: Values inherited from Infrastructure Defaults, which can be overridden at the collection level.
    *   **Required Fields**: Settings that must be defined per collection (e.g., MCP tool prefix).
    *   **Source Config**: A plugin-specific configuration blob. The `DocumentSource` plugin is responsible for validating this blob into its own schema.

*ConfigManager* is a class that collects all the configuration together.

