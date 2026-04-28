import asyncio
import time
import logging
from typing import List, Any, Dict, Tuple, Optional
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

logger = logging.getLogger(__name__)

VECTOR_DIMENSIONS = 768
_MISSING = object()

class BatchCall:
    """
    Base class providing async batching and queueing logic.
    Subclasses must implement _execute_batch and _process_item.
    """
    def __init__(self, model: Any, batch_size: int = 10, batch_timeout: float = 3.0):
        self.model = model
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.queue: asyncio.Queue = asyncio.Queue()
        self.buffer: List[Tuple[int, str, int]] = []
        self._worker_task = asyncio.create_task(self._queue_worker())

    async def __call__(self, prompts: List[str]) -> List[Any]:
        if not prompts:
            return []

        future = asyncio.get_event_loop().create_future()
        await self.queue.put((future, prompts))
        
        try:
            return await future
        except asyncio.CancelledError:
            raise

    async def _queue_worker(self):
        pending_requests: Dict[int, Dict[str, Any]] = {}

        while True:
            should_dispatch_partial = False
            try:
                current_timeout = None if not self.buffer else self.batch_timeout
                future, prompts = await asyncio.wait_for(self.queue.get(), timeout=current_timeout)
                
                f_id = id(future)
                pending_requests[f_id] = {
                    "future": future,
                    "results": [_MISSING] * len(prompts),
                    "expected": len(prompts)
                }
                for idx, p in enumerate(prompts):
                    self.buffer.append((f_id, p, idx))
            except asyncio.TimeoutError:
                should_dispatch_partial = True
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self.__class__.__name__} worker error: {e}")
                await asyncio.sleep(1)
                continue

            while len(self.buffer) > 0 and (len(self.buffer) >= self.batch_size or should_dispatch_partial):
                batch_size = min(len(self.buffer), self.batch_size)
                batch_to_send = self.buffer[:batch_size]
                self.buffer = self.buffer[batch_size:]
                await self._dispatch_batch(batch_to_send, pending_requests)
                
                if batch_size < self.batch_size:
                    should_dispatch_partial = False

    async def _dispatch_batch(self, items: List[Tuple[int, str, int]], pending_requests: Dict[int, Dict[str, Any]]):
        if not items:
            return

        prompts = [item[1] for item in items]
        try:
            results = await self._execute_batch(prompts)
            if len(results) != len(items):
                raise ValueError(
                    f"{self.__class__.__name__} returned {len(results)} results for {len(items)} prompts"
                )
            
            for (f_id, _, idx), res in zip(items, results):
                val = self._process_item(res)
                
                if f_id in pending_requests:
                    req = pending_requests[f_id]
                    req["results"][idx] = val
                    
                    if all(r is not _MISSING for r in req["results"]):
                        fut = req["future"]
                        if not fut.done():
                            fut.set_result(req["results"])
                        del pending_requests[f_id]
                            
        except Exception as e:
            for f_id, _, _ in items:
                if f_id in pending_requests:
                    fut = pending_requests[f_id]["future"]
                    if not fut.done():
                        fut.set_exception(e)
                    del pending_requests[f_id]

    async def _execute_batch(self, prompts: List[str]) -> List[Any]:
        raise NotImplementedError

    def _process_item(self, item: Any) -> Any:
        raise NotImplementedError

class LlmCall(BatchCall):
    """
    Wrapper for a LangChain ChatModel with request batching.
    """
    def __init__(self, batch_size: int = 10, batch_timeout: float = 3.0, **kwargs):
        llm = init_chat_model(**kwargs).with_retry(stop_after_attempt=3)
        super().__init__(llm, batch_size=batch_size, batch_timeout=batch_timeout)

    async def _execute_batch(self, prompts: List[str|list[dict]]) -> List[Any]:
        return await self.model.abatch(prompts)

    def _process_item(self, item: Any) -> str:
        return item.content if hasattr(item, 'content') else str(item)

class EmbeddingCall(BatchCall):
    """
    Wrapper for a LangChain Embeddings model with batching for documents
    and real-time processing for queries.
    """
    def __init__(self, dimensions: int = VECTOR_DIMENSIONS, batch_size: int = 10, batch_timeout: float = 3.0, **kwargs):
        self.dimensions = dimensions
        embeddings = init_embeddings(**kwargs)
        super().__init__(embeddings, batch_size=batch_size, batch_timeout=batch_timeout)

    async def query(self, text: str, dimensions: Optional[int] = None) -> List[float]:
        dims = dimensions if dimensions is not None else self.dimensions
        res = await self.model.aembed_query(text)
        return res[:dims]

    async def _execute_batch(self, texts: List[str]) -> List[Any]:
        return await self.model.aembed_documents(texts)

    def _process_item(self, item: Any) -> List[float]:
        return item[:self.dimensions]
