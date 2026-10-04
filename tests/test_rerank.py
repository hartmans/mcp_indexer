import asyncio
import sys
from types import SimpleNamespace

import pytest

from mcp_indexer.rerank import AbstractReranker, QwenReranker


class RoleTemplateTokenizer:
    """Mirror the template's role lookup, including empty fields for wrong roles."""
    def apply_chat_template(self, messages, **kwargs):
        roles = {message["role"]: message["content"] for message in messages}
        return (
            f"<Instruct>: {roles.get('system', '')}\n"
            f"<Query>: {roles.get('query', '')}\n"
            f"<Document>: {roles.get('document', '')}" + QwenReranker.PROMPT_SUFFIX
        )


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
            return {"input_ids": [[3, 4, 5] for _ in prompts]}

        def encode(self, text, **kwargs):
            return [6]

        def pad(self, inputs, **kwargs):
            return {"input_ids": torch.tensor(inputs["input_ids"])}

    class Model:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0]]] * 2))

    reranker = QwenReranker()
    reranker.tokenizer = Tokenizer()
    reranker.model = Model()
    monkeypatch.setattr(reranker, "_format_prompt", lambda query, document: document + reranker.PROMPT_SUFFIX)

    assert len(reranker._score_batch_sync("query", ["one", "two"])) == 2
    assert calls[0]["logits_to_keep"] == 1


def test_prompt_uses_template_query_document_and_instruction_roles():
    reranker = QwenReranker(instruction="Match the requested topic")
    reranker.tokenizer = RoleTemplateTokenizer()
    prompt = reranker._format_prompt("Explain gravity", "Gravity attracts bodies.")
    assert "<Instruct>: Match the requested topic" in prompt
    assert "<Query>: Explain gravity" in prompt
    assert "<Document>: Gravity attracts bodies." in prompt
    assert reranker._format_prompt("Explain gravity", "Beijing is a city.") != prompt


def test_batch_truncation_preserves_answer_suffix_and_padding_mask():
    import torch
    model_inputs = []
    bodies = []
    class Tokenizer(RoleTemplateTokenizer):
        def encode(self, text, **kwargs):
            assert kwargs["add_special_tokens"] is False
            assert text == QwenReranker.PROMPT_SUFFIX
            return [12, 13]
        def convert_tokens_to_ids(self, token):
            return {"no": 0, "yes": 1}[token]
        def __call__(self, prompts, **kwargs):
            assert kwargs["max_length"] == 6
            assert not kwargs["padding"] and kwargs["truncation"]
            assert not kwargs["add_special_tokens"]
            bodies.extend(prompts)
            # An overlong document and a short one after body-only truncation.
            return {"input_ids": [[10, 11, 2, 3, 4, 5], [10, 11, 7]]}
        def pad(self, inputs, **kwargs):
            rows = inputs["input_ids"]
            assert rows == [[10, 11, 2, 3, 4, 5, 12, 13], [10, 11, 7, 12, 13]]
            return {"input_ids": torch.tensor([rows[0], [0, 0, 0] + rows[1]]),
                    "attention_mask": torch.tensor([[1] * 8, [0, 0, 0] + [1] * 5])}
    class Model:
        device = torch.device("cpu")
        def __call__(self, **kwargs):
            model_inputs.append(kwargs)
            return SimpleNamespace(logits=torch.tensor([[[0.0, 2.0]], [[2.0, 0.0]]]))
    reranker = QwenReranker(max_length=8)
    reranker.tokenizer = Tokenizer()
    reranker.model = Model()
    scores = reranker._score_batch_sync("gravity", ["Long " * 100, "Short"])
    assert scores[0] > 0.5 > scores[1]
    assert "<Query>: gravity" in bodies[0] and "<Document>: Long" in bodies[0]
    assert model_inputs[0]["attention_mask"][1].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert model_inputs[0]["logits_to_keep"] == 1


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
