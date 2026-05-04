import dataclasses
import json
from datetime import datetime
from types import MappingProxyType
from typing import List, AsyncGenerator, Generic, TypeVar, Any, AsyncIterator, Literal, Dict, Type, Mapping
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field, model_validator
from ..context import Context
from ..config import CollectionConfig
from ..llm import VECTOR_DIMENSIONS
from ..context import SOURCE_REGISTRY

class DocumentChunk(LanceModel):
    '''A DocumentChunk as stored in the database after indexing has been handled.
    This is *not* a type used within DocumentSources.
    '''
    document_id:str = Field(description="Unique identifier for the document to which this chunk belongs.")
    order: int = Field(description="Order of the chunk within the document, starting at 0")
    chunk_id: str = Field(description="Identifier of this chunk; document_id should be a prefix")
    text: str|None = Field(
        default=None,
        description="Full text of this chunk. If the document source can retrieve easily it can supply the embedding explicitly and avoid duplicating the text.")
    embedding: Vector(VECTOR_DIMENSIONS) # pyright: ignore[reportInvalidTypeForm]
    summary_span: int|None = Field(
        default=None,
        description="Which summary span covers this chunk; None if not yet summarized.",
    )
    metadata_str: str = Field(
        default="{}",
        description="JSON-encoded chunk metadata used to retrieve the source chunk.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_metadata_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        if "metadata" in normalized:
            normalized["metadata_str"] = cls._serialize_metadata(normalized.pop("metadata"))
        elif "metadata_str" in normalized:
            normalized["metadata_str"] = cls._serialize_metadata(normalized["metadata_str"])

        return normalized

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(self._deserialize_metadata(self.metadata_str))

    @metadata.setter
    def metadata(self, value: Mapping[str, Any]) -> None:
        self.metadata_str = self._serialize_metadata(value)

    @staticmethod
    def _serialize_metadata(value: Mapping[str, Any] | str | None) -> str:
        if value is None:
            return "{}"

        if isinstance(value, str):
            parsed = DocumentChunk._deserialize_metadata(value)
            return json.dumps(parsed, sort_keys=True)

        return json.dumps(dict(value), sort_keys=True)

    @staticmethod
    def _deserialize_metadata(value: str | None) -> dict[str, Any]:
        if not value:
            return {}

        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError("DocumentChunk metadata must decode to a JSON object.")

        return parsed


class Document(LanceModel):
    document_id: str
    title: str = Field(description="Title or file name of this document")
    embedding: Vector(VECTOR_DIMENSIONS) = Field( # pyright: ignore[reportInvalidTypeForm]
        description="Embedding either of the entire document or of the chunk level summaries.")
    keywords: list[str] = []
    summary: str = ""
    last_modified: datetime = Field(json_schema_extra={"tz": "UTC"})

class ChunkSummary(LanceModel):
    document_id: str
    summary_span: int
    summary: str
    
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
        ...
    
    async def get_chunks(self, min_size: int, max_size: int)-> AsyncGenerator[ChunkInfo, None]:
        ...

    async def fetch_chunk(self, chunk_metadata: dict[str,Any])-> list[str]:
        ...

def create_embedding_chunks(
    metadata: dict, 
    text_list: list[str], 
    min_size: int, 
    max_size: int
) -> List[tuple[dict, str]]:
    """
    Processes a single semantic chunk into embedding-sized chunks.
    Returns a list of (metadata, text) tuples.
    """
    text = "".join(text_list)
    text_len = len(text)
    
    results = []
    offset = 0
    
    while offset < text_len:
        end = min(offset + max_size, text_len)
        if offset > 0 and (end - offset) < min_size:
            if results:
                last_meta, last_text = results[-1]
                results[-1] = (last_meta, last_text + text[offset:end])
                offset = end
                continue
        
        chunk_text = text[offset:end]
        chunk_meta = {**metadata, 'o': offset, 's': len(chunk_text)}
        results.append((chunk_meta, chunk_text))
        offset = end
        
    return results

p = TypeVar("p", bound=DocumentPointer)
class DocumentSource(Generic[p]):
    '''An abstract plugin representing a source of documents.'''
    
    def __init__(self, collection_id:str,
                 *, context:Context,
                 collection_config:CollectionConfig):
        self.context = context
        self.config = collection_config
        self.id = collection_id
        
        prefix = getattr(self, 'source_prefix', self.__class__.__name__.lower())
        self.id_prefix = f"{prefix}:{self.id}:"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prefix = getattr(cls, 'source_prefix', cls.__name__.lower())
        if prefix not in SOURCE_REGISTRY:
            SOURCE_REGISTRY[prefix] = cls

    def strip_id_prefix(self, document_id: str) -> str:
        if not document_id.startswith(self.id_prefix):
            raise ValueError(f"document_id '{document_id}' does not start with expected prefix '{self.id_prefix}'")
        return document_id[len(self.id_prefix):]

    async def get_documents(self, last_modified:datetime|None = None)-> AsyncGenerator[p, None]:
        ...

    async def fetch_document(document_id:str)-> p:
        ...


    async def fetch_chunk(self, document_id: str, chunk_metadata: dict[str, Any], scope: Literal['semantic', 'embedding']) -> str:
        pointer = self.fetch_document(document_id)
        semantic_chunk_list = await pointer.fetch_chunk(chunk_metadata)
        full_text = "".join(semantic_chunk_list)
        
        if scope == 'semantic':
            return full_text
            
        offset = chunk_metadata.get('o', 0)
        span = chunk_metadata.get('s', len(full_text))
        return full_text[offset : offset + span]

    def document_id(self, chunk_id: str) -> str:
        return chunk_id.rsplit('?', 1)[0]
