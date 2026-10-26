"""Dispatcher robustness for the indexer's pending-work queue.

The dispatcher is the single routine that fills free worker slots; a
work error or cancellation must never bring it down. These tests cover
the queue mechanics with a mocked context, no database.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from mcp_indexer.indexer import Indexer, PRIORITY_INDEX

SRC = SimpleNamespace(id="col")


def make_indexer(capacity=2):
    server_cfg = SimpleNamespace(min_size=10, max_size=100)
    context = SimpleNamespace(
        config=SimpleNamespace(get_server_config=lambda: server_cfg),
        collections={},
    )
    indexer = Indexer(context)
    indexer._work_capacity = capacity
    return indexer


async def pump(indexer, n=10):
    """Give the dispatcher and dispatched work turns to run."""
    for _ in range(n):
        await asyncio.sleep(0)


def make_record_failure():
    failures = []

    async def record(source, document_id, reason):
        failures.append((document_id, reason.splitlines()[0]))

    return record, failures


async def test_work_error_does_not_stop_dispatcher(caplog):
    """A failing run raises on its future, records the failure before
    wait_for_idle returns, logs the error, and leaves the dispatcher
    able to run further work."""
    caplog.set_level(logging.ERROR, "mcp_indexer.indexer")

    async def failing(source, document_id):
        raise RuntimeError("boom")

    record, failures = make_record_failure()
    indexer = make_indexer()
    indexer._record_failure = record
    bad = indexer.submit(PRIORITY_INDEX, failing, source=SRC, document_id="bad")
    ran = []

    async def ok(source, document_id):
        ran.append(document_id)

    good = indexer.submit(PRIORITY_INDEX, ok, source=SRC, document_id="good")
    with pytest.raises(RuntimeError, match="boom"):
        await bad
    await good
    await indexer.wait_for_idle()
    # The failure record is committed before wait_for_idle returns.
    assert failures == [("bad", "RuntimeError: boom")]
    assert any("bad" in record.message for record in caplog.records)
    # The dispatcher survives: later work still runs.
    async def later(source, document_id):
        ran.append(document_id)

    later_fut = indexer.submit(PRIORITY_INDEX, later, source=SRC, document_id="later")
    await indexer.wait_for_idle()
    await later_fut
    assert ran == ["good", "later"]


async def test_synchronous_factory_error_does_not_stop_dispatcher():
    """A factory that raises synchronously lands as an ordinary task
    failure; the dispatcher keeps running."""
    async def record(source, document_id, reason):
        recorded.append(document_id)

    recorded = []

    def sync_fail(source, document_id):
        raise ValueError("sync boom")

    indexer = make_indexer()
    indexer._record_failure = record
    bad = indexer.submit(PRIORITY_INDEX, sync_fail, source=SRC, document_id="bad")
    with pytest.raises(ValueError, match="sync boom"):
        await bad
    ran = []

    async def ok(source, document_id):
        ran.append(document_id)

    good = indexer.submit(PRIORITY_INDEX, ok, source=SRC, document_id="good")
    await indexer.wait_for_idle()
    await good
    assert recorded == ["bad"]
    assert ran == ["good"]


async def test_work_cancellation_does_not_stop_dispatcher():
    """Cancelling a running work task resolves its future as cancelled
    and leaves the dispatcher able to dispatch further work."""
    async def slow(source, document_id):
        await asyncio.sleep(1.0)

    indexer = make_indexer()
    fut = indexer.submit(PRIORITY_INDEX, slow, source=SRC, document_id="slow")
    await pump(indexer)
    assert len(indexer._in_flight) == 1
    task = next(iter(indexer._in_flight))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fut

    ran = []

    async def ok(source, document_id):
        ran.append(document_id)

    good = indexer.submit(PRIORITY_INDEX, ok, source=SRC, document_id="good")
    await indexer.wait_for_idle()
    await good
    assert ran == ["good"]


async def test_cancelled_waiter_does_not_stop_work():
    """A run awaited from several submissions (dedup join) outlives one
    waiter's cancellation: the run completes and the remaining waiter
    receives its result."""
    ran = []

    async def work(source, document_id):
        await asyncio.sleep(0.02)
        ran.append(document_id)
        return "done"

    indexer = make_indexer()
    # Two submissions, one run: each waiter holds its own shielded future.
    run_a = indexer.submit(PRIORITY_INDEX, work, source=SRC, document_id="a",
                           key=("k", 1))
    run_b = indexer.submit(PRIORITY_INDEX, work, source=SRC, document_id="a",
                           key=("k", 1))

    async def wait_for_run(run):
        return await run

    waiter_1 = asyncio.create_task(wait_for_run(run_a))
    waiter_2 = asyncio.create_task(wait_for_run(run_b))
    await pump(indexer)
    waiter_1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_1
    assert await waiter_2 == "done"
    assert ran == ["a"]
    await indexer.wait_for_idle()


async def test_dispatcher_exits_when_no_work_remains_and_respawns():
    """The dispatcher exits only when no work remains (running or
    queued), and the next submission respawns it."""
    async def work(source, document_id):
        return 1

    indexer = make_indexer()
    first = indexer.submit(PRIORITY_INDEX, work, source=SRC, document_id="a")
    await indexer.wait_for_idle()
    assert await first == 1
    assert indexer._dispatcher is None or indexer._dispatcher.done()
    second = indexer.submit(PRIORITY_INDEX, work, source=SRC, document_id="b")
    await indexer.wait_for_idle()
    assert await second == 1
