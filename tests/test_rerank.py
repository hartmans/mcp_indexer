import asyncio
import sys
from types import SimpleNamespace

import pytest

from mcp_indexer.rerank import AbstractReranker, QwenReranker


def test_abstract_reranker_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractReranker()


@pytest.mark.parametrize(
    ("cuda_available", "expected_model", "quantized"),
    [
        (False, QwenReranker.CPU_MODEL, False),
        (True, QwenReranker.CUDA_MODEL, True),
    ],
)
async def test_setup_selects_model(monkeypatch, cuda_available, expected_model, quantized):
    calls = []

    class FakeTokenizer:
        eos_token = "eos"
        pad_token = None

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("tokenizer", model_name, kwargs))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("model", model_name, kwargs))
            return cls()

        def eval(self):
            return self

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available))
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeModel,
        AutoTokenizer=FakeTokenizer,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    reranker = QwenReranker()
    monkeypatch.setattr(asyncio, "to_thread", _run_inline)
    await reranker.setup()
    await reranker.setup()

    assert reranker.model_name == expected_model
    model_call = next(call for call in calls if call[0] == "model")
    if quantized:
        assert model_call[2]["device_map"] == "auto"
        assert model_call[2]["quantization_config"].kwargs == {"load_in_8bit": True}
    else:
        assert model_call[2] == {}
    assert sum(call[0] == "model" for call in calls) == 1


async def test_score_sorts_and_limits(monkeypatch):
    reranker = QwenReranker(batch_size=2)
    reranker.model = object()
    reranker.tokenizer = object()
    batches = []

    def fake_score(query, documents):
        batches.append((query, list(documents)))
        return [0.2, 0.9, 0.5]

    monkeypatch.setattr(reranker, "_score_sync", fake_score)
    monkeypatch.setattr(asyncio, "to_thread", _run_inline)

    result = await reranker.score("query", ["zero", "one", "two"], top_n=2)

    assert result == [(1, 0.9), (2, 0.5)]
    assert batches == [("query", ["zero", "one", "two"])]


def test_score_sync_batches_documents(monkeypatch):
    reranker = QwenReranker(batch_size=2)
    reranker.model = object()
    reranker.tokenizer = object()
    batches = []

    def fake_score_batch(query, documents):
        batches.append((query, list(documents)))
        return [float(len(document)) for document in documents]

    monkeypatch.setattr(reranker, "_score_batch_sync", fake_score_batch)

    result = reranker._score_sync("query", ["a", "bb", "ccc", "dddd", "eeeee"])

    assert result == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert batches == [
        ("query", ["a", "bb"]),
        ("query", ["ccc", "dddd"]),
        ("query", ["eeeee"]),
    ]


def test_score_only_requests_last_token_logits(monkeypatch):
    import torch

    calls = []

    class Tokenizer:
        def convert_tokens_to_ids(self, token):
            return {"no": 0, "yes": 1}[token]

        def __call__(self, prompts, **kwargs):
            return {"input_ids": torch.ones((len(prompts), 3), dtype=torch.long)}

    class Model:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0]]] * 2))

    reranker = QwenReranker()
    reranker.tokenizer = Tokenizer()
    reranker.model = Model()
    monkeypatch.setattr(reranker, "_format_prompt", lambda query, document: document)

    assert len(reranker._score_batch_sync("query", ["one", "two"])) == 2
    assert calls[0]["logits_to_keep"] == 1


async def test_score_requires_setup():
    with pytest.raises(RuntimeError, match="setup"):
        await QwenReranker().score("query", ["document"])


async def test_score_holds_semaphore(monkeypatch):
    reranker = QwenReranker(max_calls_in_flight=1)
    reranker.model = object()
    reranker.tokenizer = object()
    active = 0
    maximum_active = 0

    async def fake_to_thread(function, *args):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        result = function(*args)
        active -= 1
        return result

    monkeypatch.setattr(reranker, "_score_sync", lambda query, documents: [0.5])
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    await asyncio.gather(
        reranker.score("one", ["document"]),
        reranker.score("two", ["document"]),
    )

    assert maximum_active == 1


async def _run_inline(function, *args):
    return function(*args)
