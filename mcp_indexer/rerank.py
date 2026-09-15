"""Local reranking models."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeAlias

RerankResult: TypeAlias = tuple[int, float]


class AbstractReranker(ABC):
    """Interface for allocating and invoking a reranker.

    Enabled collections share one instance so its scoring semaphore limits
    concurrent calls across collections. Construction must not load the model;
    search calls setup on first use and awaits it before scoring.
    """

    @abstractmethod
    async def setup(self) -> None:
        """Download the model, allocate its resources, and make it ready.

        Must be idempotent and coordinate concurrent callers: simultaneous
        first searches must initialize the shared model only once. Subsequent
        calls after successful initialization must reuse the allocated model.
        """

    @abstractmethod
    async def score(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Return ``(document_index, score)`` pairs in descending score order."""


class QwenReranker(AbstractReranker):
    """Run a Qwen3 reranker locally with Transformers.

    CUDA hosts use the 4B model with 8-bit weights. CPU-only hosts use the
    0.6B model without bitsandbytes quantization.
    """

    CPU_MODEL = "Qwen/Qwen3-Reranker-0.6B"
    CUDA_MODEL = "Qwen/Qwen3-Reranker-4B"
    DEFAULT_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(
        self,
        *,
        batch_size: int = 8,
        max_calls_in_flight: int = 1,
        max_length: int = 8192,
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_calls_in_flight < 1:
            raise ValueError("max_calls_in_flight must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction
        self.semaphore = asyncio.Semaphore(max_calls_in_flight)
        self._setup_lock = asyncio.Lock()
        self.model_name: str | None = None
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    async def setup(self) -> None:
        """Select, download, and load the model for the available hardware."""
        async with self._setup_lock, self.semaphore:
            if self.model is not None:
                return
            await asyncio.to_thread(self._setup_sync)

    def _setup_sync(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_cuda = torch.cuda.is_available()
        self.model_name = self.CUDA_MODEL if use_cuda else self.CPU_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if use_cuda:
            from transformers import BitsAndBytesConfig

            model_kwargs.update(
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        ).eval()

    async def score(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("QwenReranker.setup() must complete before score()")
        if top_n is not None and top_n < 0:
            raise ValueError("top_n cannot be negative")
        if not documents or top_n == 0:
            return []

        async with self.semaphore:
            scores = await asyncio.to_thread(self._score_sync, query, documents)

        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        return ranked

    def _score_sync(self, query: str, documents: Sequence[str]) -> list[float]:
        assert self.model is not None
        assert self.tokenizer is not None

        scores: list[float] = []

        for offset in range(0, len(documents), self.batch_size):
            batch = documents[offset : offset + self.batch_size]
            scores.extend(self._score_batch_sync(query, batch))

        return scores

    def _score_batch_sync(self, query: str, documents: Sequence[str]) -> list[float]:
        import torch

        assert self.model is not None
        assert self.tokenizer is not None

        true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        prompts = [self._format_prompt(query, document) for document in documents]
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
            pair_logits = torch.stack(
                (logits[:, false_token_id], logits[:, true_token_id]), dim=1
            )
            batch_scores = torch.softmax(pair_logits, dim=1)[:, 1]

        return [float(score) for score in batch_scores.cpu().tolist()]

    def _format_prompt(self, query: str, document: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Judge whether the Document meets the requirements based on the "
                    "Query and the Instruct provided. Answer only yes or no."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<Instruct>: {self.instruction}\n\n"
                    f"<Query>: {query}\n\n<Document>: {document}"
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
