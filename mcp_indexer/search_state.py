"""Collection-owned, process-local continuation state. No source/model imports."""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class InvalidSearchCursor(ValueError):
    def __init__(self):
        super().__init__("Invalid or expired search cursor")


class SearchAdvanceError(RuntimeError):
    """The same continuation can be retried without losing recorded work."""


@dataclass
class HitRecord:
    hit: Any
    encounter: int
    text: str | None = None
    score: float | None = None
    preparation_attempts: int = 0
    finished: bool = False


@dataclass
class PageWork:
    limit: int
    discovery_done: bool = False
    warnings: list[str] = field(default_factory=list)
    assembled: dict[str, Any] = field(default_factory=dict)
    assembly_attempted: set[str] = field(default_factory=set)

    def warn(self, message: str) -> None:
        # Repeated failed attempts retain work, not an unbounded error history.
        if message not in self.warnings:
            self.warnings.append(message)


@dataclass(eq=False)
class SearchSession:
    query: str
    rerank_query: str | None
    natural_query: str
    document_limit: int
    chunk_limit: int
    min_cosine_distance: float
    min_rerank_score: float
    reranker: Any
    current_cursor: str | None = None
    previous_cursor: str | None = None
    cached_page: Any = None
    embedding: Any = None
    vector_document_offset: int = 0
    vector_chunk_offset: int = 0
    native_document_offset: int = 0
    native_chunk_offset: int = 0
    vector_documents_done: bool = False
    vector_chunks_done: bool = False
    native_done: bool = False
    records: dict[tuple, HitRecord] = field(default_factory=dict)
    encounter: int = 0
    returned: set[str] = field(default_factory=set)
    failed_documents: set[str] = field(default_factory=set)
    assembly_attempts: dict[str, int] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task | None = None
    work: PageWork | None = None
    deadline: float = 0
    timer: asyncio.TimerHandle | None = None

    def release_candidates(self) -> None:
        self.records.clear()
        self.returned.clear()
        self.failed_documents.clear()
        self.assembly_attempts.clear()
        self.embedding = None
        self.reranker = None


class SearchSessionStore:
    """Two live tokens per session, with inactivity cleanup even without requests."""

    def __init__(self, *, ttl: float = 3600, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self.clock = clock
        self._tokens: dict[str, SearchSession] = {}

    def add(self, session: SearchSession) -> None:
        session.current_cursor = self._new_token(session)
        self.touch(session)

    def _new_token(self, session: SearchSession) -> str:
        while (token := secrets.token_urlsafe(32)) in self._tokens:
            pass
        self._tokens[token] = session
        return token

    def lookup(self, token: str) -> SearchSession:
        session = self._tokens.get(token)
        if session is None:
            raise InvalidSearchCursor()
        if self.clock() >= session.deadline and not self._running(session):
            self.remove(session)
            raise InvalidSearchCursor()
        return session

    @staticmethod
    def _running(session: SearchSession) -> bool:
        return session.task is not None and not session.task.done()

    def touch(self, session: SearchSession) -> None:
        session.deadline = self.clock() + self.ttl
        if session.timer is not None:
            session.timer.cancel()
        session.timer = asyncio.get_running_loop().call_later(self.ttl, self._expire, session)

    def task_finished(self, session: SearchSession, task: asyncio.Task) -> None:
        # Retrieve failures even when all callers disconnected, and give long
        # failed operations a fresh retry window as well as successful ones.
        if not task.cancelled():
            task.exception()
        if any(self._tokens.get(token) is session
               for token in (session.current_cursor, session.previous_cursor) if token is not None):
            self.touch(session)

    def _expire(self, session: SearchSession) -> None:
        remaining = session.deadline - self.clock()
        if self._running(session):
            remaining = self.ttl
        if remaining > 0:
            session.timer = asyncio.get_running_loop().call_later(remaining, self._expire, session)
        else:
            self.remove(session)

    def commit(self, session: SearchSession, page: Any, *, complete: bool, initial: bool) -> None:
        """Called without awaits, after the page has been snapshotted."""
        if session.previous_cursor is not None:
            self._tokens.pop(session.previous_cursor, None)
        consumed = session.current_cursor
        session.previous_cursor = None if initial else consumed
        if initial:
            self._tokens.pop(consumed, None)
        session.current_cursor = None if complete else self._new_token(session)
        # The result is replaced by the orchestrator with this new token.
        session.cached_page = page
        session.work = None
        if complete:
            session.release_candidates()
        self.touch(session)

    def remove(self, session: SearchSession) -> None:
        for token in (session.current_cursor, session.previous_cursor):
            self._tokens.pop(token, None)
        if session.timer is not None:
            session.timer.cancel()
            session.timer = None
        session.cached_page = None
        session.work = None
        session.release_candidates()

    def close(self) -> None:
        """Release idle state and cancel active work during application teardown."""
        for session in set(self._tokens.values()):
            if self._running(session):
                session.task.cancel()
            self.remove(session)
