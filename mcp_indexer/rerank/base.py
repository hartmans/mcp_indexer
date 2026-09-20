"""Common reranker protocol."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

RerankResult: TypeAlias = tuple[int, float]


class AbstractReranker(ABC):
    """Interface for allocating and invoking a shared reranker.

    Construction must not allocate model resources. ``setup`` is called on
    first use and must coordinate concurrent first callers.
    """

    @abstractmethod
    async def setup(self) -> None:
        """Allocate resources and make the reranker ready for requests."""

    @abstractmethod
    async def score(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Return ``(document_index, score)`` pairs in descending score order."""
