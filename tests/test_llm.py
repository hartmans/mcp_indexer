import asyncio
import pytest
import mcp_indexer.llm as llm_module

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChatModel:
    def __init__(self):
        self.calls = []

    async def abatch(self, inputs):
        self.calls.append(inputs)
        return [MockMessage(content=f"Response to {p}") for p in inputs]
    
    def with_retry(self, **kwargs):
        return self

class MockEmbeddings:
    def __init__(self):
        self.calls = []

    async def aembed_query(self, text):
        self.calls.append(("query", text))
        return [0.1] * 1536
    
    async def aembed_documents(self, texts):
        self.calls.append(("docs", texts))
        return [[0.1] * 1536 for _ in texts]

@pytest.fixture
def mock_llm_factory():
    original_factory = llm_module.init_chat_model
    mock_model = MockChatModel()
    
    def factory(**kwargs):
        return mock_model
        
    llm_module.init_chat_model = factory
    yield mock_model
    llm_module.init_chat_model = original_factory

@pytest.fixture
def mock_emb_factory():
    original_factory = llm_module.init_embeddings
    mock_emb = MockEmbeddings()
    
    def factory(**kwargs):
        return mock_emb
        
    llm_module.init_embeddings = factory
    yield mock_emb
    llm_module.init_embeddings = original_factory

@pytest.mark.asyncio
async def test_llm_single_request(mock_llm_factory):
    caller = LlmCall(model="test")
    res = await caller(["Hello"])
    assert res == ["Response to Hello"]
    assert len(mock_llm_factory.calls) == 1
    assert mock_llm_factory.calls[0] == ["Hello"]

@pytest.mark.asyncio
async def test_llm_batch_split(mock_llm_factory):
    caller = LlmCall(model="test")
    # Request 15 prompts -> should split into 10 and 5
    prompts = [f"Prompt {i}" for i in range(15)]
    res = await caller(prompts)
    
    assert len(res) == 15
    assert res[0] == "Response to Prompt 0"
    assert res[14] == "Response to Prompt 14"
    assert len(mock_llm_factory.calls) == 2
    assert len(mock_llm_factory.calls[0]) == 10
    assert len(mock_llm_factory.calls[1]) == 5

@pytest.mark.asyncio
async def test_llm_multi_request_batching(mock_llm_factory):
    caller = LlmCall(model="test")
    
    # Fire off two requests: one for 6, one for 4
    # These should be combined into one batch of 10
    task1 = asyncio.create_task(caller([f"P{i}" for i in range(6)]))
    task2 = asyncio.create_task(caller([f"P{i}" for i in range(4)]))
    
    await asyncio.gather(task1, task2)
    
    # Check that they were batched together
    # We expect one call of size 10
    assert len(mock_llm_factory.calls) == 1
    assert len(mock_llm_factory.calls[0]) == 10

@pytest.mark.asyncio
async def test_llm_timeout_dispatch(mock_llm_factory):
    caller = LlmCall(model="test")
    
    # Send 3 prompts. Buffer now has 3.
    # Should not be dispatched until timeout.
    task = asyncio.create_task(caller(["P1", "P2", "P3"]))
    
    # Small sleep to let the worker process the queue
    await asyncio.sleep(0.1)
    assert len(mock_llm_factory.calls) == 0
    
    # Now wait for the sliding timeout (3s)
    await asyncio.sleep(3.1)
    
    res = await task
    assert res == ["Response to P1", "Response to P2", "Response to P3"]
    assert len(mock_llm_factory.calls) == 1
    assert len(mock_llm_factory.calls[0]) == 3

@pytest.mark.asyncio
async def test_embedding_call(mock_emb_factory):
    caller = EmbeddingCall(model="test")
    res = await caller(["Text 1", "Text 2"])
    assert len(res) == 2
    assert res[0] == [0.1] * 1536
    assert len(mock_emb_factory.calls) == 1
    assert mock_emb_factory.calls[0] == ("docs", ["Text 1", "Text 2"])

@pytest.mark.asyncio
async def test_emb_query_realtime(mock_emb_factory):
    caller = EmbeddingCall(model="test")
    res = await caller.query("Hello world")
    assert res == [0.1] * 1536
    assert len(mock_emb_factory.calls) == 1
    assert mock_emb_factory.calls[0] == ("query", "Hello world")

@pytest.mark.asyncio
async def test_emb_batch_split(mock_emb_factory):
    caller = EmbeddingCall(model="test")
    texts = [f"Doc {i}" for i in range(15)]
    res = await caller(texts)
    assert len(res) == 15
    assert res[0] == [0.1] * 1536
    assert len(mock_emb_factory.calls) == 2
    assert len(mock_emb_factory.calls[0][1]) == 10
    assert len(mock_emb_factory.calls[1][1]) == 5

@pytest.mark.asyncio
async def test_emb_multi_request_batching(mock_emb_factory):
    caller = EmbeddingCall(model="test")
    task1 = asyncio.create_task(caller([f"D{i}" for i in range(6)]))
    task2 = asyncio.create_task(caller([f"D{i}" for i in range(4)]))
    await asyncio.gather(task1, task2)
    assert len(mock_emb_factory.calls) == 1
    assert len(mock_emb_factory.calls[0][1]) == 10

@pytest.mark.asyncio
async def test_emb_timeout_dispatch(mock_emb_factory):
    caller = EmbeddingCall(model="test")
    task = asyncio.create_task(caller(["D1", "D2", "D3"]))
    await asyncio.sleep(0.1)
    assert len(mock_emb_factory.calls) == 0
    await asyncio.sleep(3.1)
    res = await task
    assert len(res) == 3
    assert len(mock_emb_factory.calls[0][1]) == 3

# Import the classes from the module we patched
from mcp_indexer.llm import LlmCall, EmbeddingCall
