import asyncio
import pytest
import mcp_indexer.llm as llm_module
from mcp_indexer.llm import EmbeddingCall, LlmCall, VECTOR_DIMENSIONS

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


class MockShortChatModel(MockChatModel):
    async def abatch(self, inputs):
        self.calls.append(inputs)
        return [MockMessage(content=f"Response to {p}") for p in inputs[:-1]]


class MockNoneChatModel(MockChatModel):
    async def abatch(self, inputs):
        self.calls.append(inputs)
        return [MockMessage(content=None) for _ in inputs]

class MockTimeoutOnceChatModel(MockChatModel):
    async def abatch(self, inputs):
        self.calls.append(inputs)
        if len(self.calls) == 1:
            await asyncio.sleep(0.05)
        return [MockMessage(content=f"Response to {p}") for p in inputs]

class MockEmbeddings:
    def __init__(self):
        self.calls = []

    async def aembed_query(self, text):
        self.calls.append(("query", text))
        return [0.1] * 1536
    
    async def aembed_documents(self, texts):
        self.calls.append(("docs", texts))
        return [[0.1] * 1536 for _ in texts]

class MockTimeoutOnceEmbeddings(MockEmbeddings):
    async def aembed_query(self, text):
        self.calls.append(("query", text))
        if len(self.calls) == 1:
            await asyncio.sleep(0.05)
        return [0.1] * 1536

    async def aembed_documents(self, texts):
        self.calls.append(("docs", texts))
        if len(self.calls) == 1:
            await asyncio.sleep(0.05)
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
def mock_short_llm_factory():
    original_factory = llm_module.init_chat_model
    mock_model = MockShortChatModel()
    
    def factory(**kwargs):
        return mock_model
        
    llm_module.init_chat_model = factory
    yield mock_model
    llm_module.init_chat_model = original_factory


@pytest.fixture
def mock_none_llm_factory():
    original_factory = llm_module.init_chat_model
    mock_model = MockNoneChatModel()
    
    def factory(**kwargs):
        return mock_model
        
    llm_module.init_chat_model = factory
    yield mock_model
    llm_module.init_chat_model = original_factory

@pytest.fixture
def mock_timeout_once_llm_factory():
    original_factory = llm_module.init_chat_model
    mock_model = MockTimeoutOnceChatModel()
    
    def factory(**kwargs):
        return mock_model
        
    llm_module.init_chat_model = factory
    yield mock_model
    llm_module.init_chat_model = original_factory

@pytest.fixture
def mock_emb_factory():
    original_factory = llm_module.init_embeddings
    mock_emb = MockEmbeddings()
    calls = []
    
    def factory(**kwargs):
        calls.append(kwargs)
        return mock_emb
        
    llm_module.init_embeddings = factory
    mock_emb.factory_calls = calls
    yield mock_emb
    llm_module.init_embeddings = original_factory

@pytest.fixture
def mock_timeout_once_emb_factory():
    original_factory = llm_module.init_embeddings
    mock_emb = MockTimeoutOnceEmbeddings()
    
    def factory(**kwargs):
        return mock_emb
        
    llm_module.init_embeddings = factory
    yield mock_emb
    llm_module.init_embeddings = original_factory

@pytest.mark.asyncio
async def test_llm_single_request(mock_llm_factory):
    caller = LlmCall(model="test", batch_timeout=0.05)
    res = await caller(["Hello"])
    assert res == ["Response to Hello"]
    assert len(mock_llm_factory.calls) == 1
    assert mock_llm_factory.calls[0] == ["Hello"]

@pytest.mark.asyncio
async def test_llm_batch_split(mock_llm_factory):
    caller = LlmCall(model="test", batch_timeout=0.05)
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
    caller = LlmCall(model="test", batch_timeout=0.05)
    
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
    caller = LlmCall(model="test", batch_timeout=0.05)

    # Send 3 prompts. Buffer now has 3.
    # Should not be dispatched until timeout.
    task = asyncio.create_task(caller(["P1", "P2", "P3"]))

    # Small sleep to let the worker process the queue
    await asyncio.sleep(0.01)
    assert len(mock_llm_factory.calls) == 0

    # Now wait for the sliding timeout (3s)
    await asyncio.sleep(0.1)
    
    res = await task
    assert res == ["Response to P1", "Response to P2", "Response to P3"]
    assert len(mock_llm_factory.calls) == 1
    assert len(mock_llm_factory.calls[0]) == 3


@pytest.mark.asyncio
async def test_llm_raises_on_short_batch_result(mock_short_llm_factory):
    caller = LlmCall(model="test", batch_timeout=0.05)

    with pytest.raises(ValueError, match="returned 1 results for 2 prompts"):
        await caller(["P1", "P2"])


@pytest.mark.asyncio
async def test_llm_allows_none_content(mock_none_llm_factory):
    caller = LlmCall(model="test", batch_timeout=0.05)
    res = await asyncio.wait_for(caller(["Hello"]), timeout=0.2)
    assert res == [None]

@pytest.mark.asyncio
async def test_llm_retries_after_request_timeout(mock_timeout_once_llm_factory):
    caller = LlmCall(
        model="test",
        batch_timeout=0.01,
        request_timeout=0.01,
        timeout_retries=2,
    )
    res = await caller(["Hello"])
    assert res == ["Response to Hello"]
    assert len(mock_timeout_once_llm_factory.calls) == 2

@pytest.mark.asyncio
async def test_embedding_call(mock_emb_factory):
    caller = EmbeddingCall(model="test", batch_timeout=0.05)
    res = await caller(["Text 1", "Text 2"])
    assert len(res) == 2
    assert res[0] == [0.1] * VECTOR_DIMENSIONS
    assert len(mock_emb_factory.calls) == 1
    assert mock_emb_factory.calls[0] == ("docs", ["Text 1", "Text 2"])

@pytest.mark.asyncio
async def test_emb_query_realtime(mock_emb_factory):
    caller = EmbeddingCall(model="test", batch_timeout=0.05)
    res = await caller.query("Hello world")
    assert res == [0.1] * VECTOR_DIMENSIONS
    assert len(mock_emb_factory.calls) == 1
    assert mock_emb_factory.calls[0] == ("query", "Hello world")

@pytest.mark.asyncio
async def test_openai_embedding_disables_langchain_token_id_path(mock_emb_factory):
    EmbeddingCall(model="test", provider="openai")
    assert mock_emb_factory.factory_calls[-1]["check_embedding_ctx_length"] is False

@pytest.mark.asyncio
async def test_openai_embedding_preserves_explicit_length_check_config(mock_emb_factory):
    EmbeddingCall(model="test", provider="openai", check_embedding_ctx_length=True)
    assert mock_emb_factory.factory_calls[-1]["check_embedding_ctx_length"] is True

@pytest.mark.asyncio
async def test_emb_batch_split(mock_emb_factory):
    caller = EmbeddingCall(model="test", batch_timeout=0.05)
    texts = [f"Doc {i}" for i in range(15)]
    res = await caller(texts)
    assert len(res) == 15
    assert res[0] == [0.1] * VECTOR_DIMENSIONS
    assert len(mock_emb_factory.calls) == 2
    assert len(mock_emb_factory.calls[0][1]) == 10
    assert len(mock_emb_factory.calls[1][1]) == 5

@pytest.mark.asyncio
async def test_emb_multi_request_batching(mock_emb_factory):
    caller = EmbeddingCall(model="test", batch_timeout=0.05)
    task1 = asyncio.create_task(caller([f"D{i}" for i in range(6)]))
    task2 = asyncio.create_task(caller([f"D{i}" for i in range(4)]))
    await asyncio.gather(task1, task2)
    assert len(mock_emb_factory.calls) == 1
    assert len(mock_emb_factory.calls[0][1]) == 10

@pytest.mark.asyncio
async def test_emb_timeout_dispatch(mock_emb_factory):
    caller = EmbeddingCall(model="test", batch_timeout=0.05)
    task = asyncio.create_task(caller(["D1", "D2", "D3"]))
    await asyncio.sleep(0.1)
    assert len(mock_emb_factory.calls) == 1
    res = await task
    assert len(res) == 3
    assert len(mock_emb_factory.calls[0][1]) == 3

@pytest.mark.asyncio
async def test_emb_retries_after_document_request_timeout(mock_timeout_once_emb_factory):
    caller = EmbeddingCall(
        model="test",
        batch_timeout=0.01,
        request_timeout=0.01,
        timeout_retries=2,
    )
    res = await caller(["D1"])
    assert res == [[0.1] * VECTOR_DIMENSIONS]
    assert mock_timeout_once_emb_factory.calls == [
        ("docs", ["D1"]),
        ("docs", ["D1"]),
    ]

@pytest.mark.asyncio
async def test_emb_retries_after_query_request_timeout(mock_timeout_once_emb_factory):
    caller = EmbeddingCall(
        model="test",
        batch_timeout=0.01,
        request_timeout=0.01,
        timeout_retries=2,
    )
    res = await caller.query("Hello world")
    assert res == [0.1] * VECTOR_DIMENSIONS
    assert mock_timeout_once_emb_factory.calls == [
        ("query", "Hello world"),
        ("query", "Hello world"),
    ]
