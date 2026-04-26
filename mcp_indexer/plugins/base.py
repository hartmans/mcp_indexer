import dataclasses
from datetime import datetime
from typing import List, AsyncGenerator, Generic, TypeVar, Any, AsyncIterator
from typing import List, AsyncGenerator, Generic, TypeVar, Any, AsyncIterator, Literal
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field
from ..context import Context
from ..config import CollectionConfig
from ..llm import VECTOR_DIMENSIONS

class DocumentChunk(LanceModel):
    '''A DocumentChunk as stored in the database after indexing has been handled.
    This is *not* a type used within DocumentSources.
    '''

    document_id:str = Field(description="Unique identifier for the document to which this chunk belongs.")
    chunk_id: str = Field(description="Identifier of this chunk; document_id should be a prefix")
    text: str|None = Field(
        default=None,
        description="Full text of this chunk. If the document source can retrieve easily it can supply the embedding explicitly and avoid duplicating the text.")
    summary: str = Field(
        default="",
        description="An llm summary of this chunk.")
    embedding: Vector(VECTOR_DIMENSIONS) # pyright: ignore[reportInvalidTypeForm]
    metadata: dict = Field(default=dict)


class Document(LanceModel):
    document_id: str
    title: str = Field(description="Title or file name of this document")
    embedding: Vector(VECTOR_DIMENSIONS) = Field( # pyright: ignore[reportInvalidTypeForm]
        description="Embedding either of the entire document or of the chunk level summaries.")
    keywords: list[str] = []
    summary: str = ""
    last_modified: datetime = Field(json_schema_extra={"tz": "UTC"})

ChunkInfo = tuple[dict[str,Any], list[str]]

@dataclasses.dataclass
class DocumentPointer:
    '''
    A potentially abstract class representing a document to be indexed. Returned from DocumentSource.
    '''
    
    source: "DocumentSource"
    document_id: str
    last_modified: datetime

    async def get_metadata(self)-> dict[str,str]:
        '''
        Returns metadata, which should include at least title.
        If the metadata includes summary or keywords these will be used rather than having the indexer construct them.
        '''
        ...
    
    async def get_chunks(self)-> AsyncGenerator[ChunkInfo]:
        '''
        Returns the chunks of the document. 

        Chunking happens at two levels. the DocumentSource is responsible for turning the document into a textual representation, identifying semantic chunking boundaries, and identifying potential inner chunk boundares.

        Often, semantic chunks will be too large, so the underlying indexing system may need to break them apart further for embedding chunks.

        Each item yielded by this generator is a semantic chunk. It consists of:

        * a dictionary of metadata that can be used to find the chunk again.
        * A list of strings. If these strings are concatenated, it forms the full semantic chunk.

        Keys in the dictionary are typically single letters for efficiency. The indexing system reserves the following keys:

        o: Offset within the semantic chunk where this embedding chunk begins
        S: Span (length) of the embedding chunk.
        
        So For example, the semantic chunk length might be 1200 characters, but embedding chunks embedding hun of 400 characters are desired. The first item in the list is 100 characters, and the second is 300 characters. The indexing system would take the return of this function and create a new chunk, copying the metadata dictionary, adding {'o':0, 's':400} and include the text from the first two list items.

        The implementation of this function is responsible for splitting into semantic chunks and within the semantic chunks splitting at reasonable boundaries (such as paragraphs) or functions.
        '''
        ...

    async def fetch_chunk(self, chunk_metadata: dict[str,Any])-> list[str]:
        '''
        Return the semantic chunk identified by chunk_info.  This function should ignore the *o* and *s* entries in the chunk_info if present; callers will handle those.
        
        It is acceptable for the list to be different than that returned by get_chunks so long as positions in the concatenated string resulting from joining the list are the same. So for example if the metadata dictionary identifies the beginning and ending position within a file, returning a single element list with that span from the file is an efficient implementation strategy.
        '''
        ...

async def create_embedding_chunks(
    chunks: AsyncIterator[ChunkInfo],
    min_size: int,
    max_size: int
) -> AsyncGenerator[ChunkInfo, None]:
    """
    Processes semantic chunks into embedding-sized chunks.

    Chunks are constructed by accumulating items from the semantic text_list to 
    respect boundaries. Chunks never span semantic boundaries.

    The resulting chunks are always represented as a list of length 1.

    Chunks may be smaller than min_size in the following cases:
    1. The remaining text at the end of a semantic chunk is smaller than 
       min_size, but must be emitted as chunks never span semantic boundaries.
    2. The final fragment of a hard-sliced item (where the item was larger 
       than max_size) is smaller than min_size.
    """
    async for metadata, text_list in chunks:
        buffer = []
        buffer_len = 0
        semantic_offset = 0
        
        for i, item in enumerate(text_list):
            item_len = len(item)
            
            # Case: Single item is larger than max_size -> hard slice
            if item_len > max_size:
                # First, emit whatever is currently in the buffer
                if buffer:
                    chunk_text = "".join(buffer)
                    yield {**metadata, 'o': semantic_offset, 's': len(chunk_text)}, [chunk_text]
                    semantic_offset += len(chunk_text)
                    buffer = []
                    buffer_len = 0
                
                # Now hard-slice the giant item
                start = 0
                while start < item_len:
                    end = min(start + max_size, item_len)
                    chunk_text = item[start:end]
                    yield {**metadata, 'o': semantic_offset, 's': len(chunk_text)}, [chunk_text]
                    semantic_offset += len(chunk_text)
                    start = end
                continue

            # Case: Adding this item would exceed max_size -> emit buffer first
            if buffer_len + item_len > max_size:
                if buffer:
                    chunk_text = "".join(buffer)
                    yield {**metadata, 'o': semantic_offset, 's': len(chunk_text)}, [chunk_text]
                    semantic_offset += len(chunk_text)
                    buffer = []
                    buffer_len = 0
                # Now the item starts a new buffer
            
            buffer.append(item)
            buffer_len += item_len
            
            # Case: We've reached the minimum size.
            # We only emit now if the remaining text in this semantic chunk 
            # is either empty or large enough to form its own chunk (>= min_size).
            if buffer_len >= min_size:
                # Calculate remaining length in text_list
                remaining_text = "".join(text_list[i + 1:])
                remaining_len = len(remaining_text)
                
                # If the remainder is too small to be its own chunk, 
                # we try to absorb it into the current chunk, provided it doesn't exceed max_size.
                if 0 < remaining_len < min_size and (buffer_len + remaining_len <= max_size):
                    # Absorb everything remaining and emit
                    buffer.extend(text_list[i + 1:])
                    chunk_text = "".join(buffer)
                    yield {**metadata, 'o': semantic_offset, 's': len(chunk_text)}, [chunk_text]
                    semantic_offset += len(chunk_text)
                    buffer = []
                    buffer_len = 0
                    # We must break the outer loop because we've consumed the entire list
                    break
                
                # Otherwise, emit the current buffer as a small-as-possible chunk
                chunk_text = "".join(buffer)
                yield {**metadata, 'o': semantic_offset, 's': len(chunk_text)}, [chunk_text]
                semantic_offset += len(chunk_text)
                buffer = []
                buffer_len = 0
        
        # Final emit for any remaining text in the semantic chunk
        if buffer:
            chunk_text = "".join(buffer)
            yield {**metadata, 'o': semantic_offset, 's': len(chunk_text)}, [chunk_text]
            semantic_offset += len(chunk_text)

p = TypeVar("p", bound=DocumentPointer)
class DocumentSource(Generic[p]):
    '''An abstract plugin representing a source of documents. It can:

    * Initialize  given a collection config

    * Get all documents modified since a given time.

    * Reindex a document given a document ID

    *  Fetch a document given a document_id

    * Fetch a chunk given a chunk ID

    * Get all valid document_ids in a collection to facilitate deleting outdated documents

    * Get all valid document_ids in a collection to facilitate deleting outdated documents

    '''
    
    def __init__(self, collection_id:str,
                 *, context:Context,
                 collection_config:CollectionConfig):
        self.context = context
        self.config = collection_config
        self.id = collection_id
        
        # Use source_prefix if defined as class var, otherwise use class name
        prefix = getattr(self, 'source_prefix', self.__class__.__name__.lower())
        self.id_prefix = f"{prefix}:{self.id}:"

    def strip_id_prefix(self, document_id: str) -> str:
        """Removes the source:collection: prefix from a document_id. Raises if prefix not found."""
        if not document_id.startswith(self.id_prefix):
            raise ValueError(f"document_id '{document_id}' does not start with expected prefix '{self.id_prefix}'")
        return document_id[len(self.id_prefix):]

    async def get_documents(self, last_modified:datetime|None = None)-> AsyncGenerator[p]:
        '''
        Async Generator yielding the set of documents modified since the given time.
        '''
        ...

    async def fetch_document(document_id:str)-> p:
        ...


    async def fetch_chunk(self, document_id: str, chunk_metadata: dict[str, Any], scope: Literal['semantic', 'embedding']) -> str:
        '''
        Fetch a chunk from the document given its metadata and the desired scope.
        '''
        pointer = self.fetch_document(document_id)
        semantic_chunk_list = await pointer.fetch_chunk(chunk_metadata)
        full_text = "".join(semantic_chunk_list)
        
        if scope == 'semantic':
            return full_text
            
        offset = chunk_metadata.get('o', 0)
        span = chunk_metadata.get('s', len(full_text))
        return full_text[offset : offset + span]

    def document_id(self, chunk_id: str) -> str:
        '''Return the document id of a chunk_id.'''
        return chunk_id.rsplit('?', 1)[0]

        
