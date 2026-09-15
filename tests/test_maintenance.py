from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy.sql.dml import Delete

from mcp_indexer.config import ServerConfig
from mcp_indexer.indexer import Indexer
from mcp_indexer.plugins.base import DocumentNotFoundError


def context_for(get_session=None):
    return SimpleNamespace(
        config=SimpleNamespace(get_server_config=ServerConfig),
        collections={}, get_session=get_session,
    )


@pytest.mark.parametrize("mode", ["full", "indexed"])
async def test_maintenance_refreshes_changed_and_deletes_immediately(monkeypatch, mode):
    now = datetime(2026, 1, 1)
    stored = {name: now for name in ("absent", "denied", "older", "same", "updated")}
    events = []

    class Session:
        async def execute(self, statement):
            params = statement.compile().params
            if isinstance(statement, Delete):
                events.append(("delete", statement.table.name, params["document_id_1"]))
                return None
            after = params.get("document_id_1", "")
            return SimpleNamespace(all=lambda: [(k, v) for k, v in stored.items() if k > after])

        async def commit(self):
            events.append("commit")

    @asynccontextmanager
    async def session():
        yield Session()

    def fetch(document_id):
        if document_id == "denied":
            # The missing document must already have been deleted and committed.
            assert events == [("fetch", "absent"), ("delete", "document", "absent"),
                              ("delete", "failed_document", "absent"), "commit"]
        events.append(("fetch", document_id))
        if document_id == "absent":
            raise DocumentNotFoundError(document_id)
        if document_id == "denied":
            raise PermissionError("unavailable")
        delta = {"older": -1, "same": 0, "updated": 1}[document_id]
        return SimpleNamespace(document_id=document_id, last_modified=now + timedelta(days=delta))

    source = SimpleNamespace(id="wiki", indexing_mode=mode, fetch_document=fetch)
    indexer = Indexer(context_for(session))

    async def index_document(pointer, stats, *, replace=False):
        assert replace
        events.append(("refresh", pointer.document_id, stats.operation))

    async def schedule(coroutine, stats):
        await coroutine

    monkeypatch.setattr(indexer, "index_document", index_document)
    monkeypatch.setattr(indexer, "_schedule_document_task", schedule)
    await indexer.maintain_collection(source)
    assert [event for event in events if isinstance(event, tuple) and event[0] == "refresh"] == [
        ("refresh", "older", "refresh"), ("refresh", "updated", "refresh"),
    ]


async def test_indexed_discovery_never_contacts_source_or_database():
    source = SimpleNamespace(indexing_mode="indexed")
    await Indexer(context_for()).index_collection(source)


async def test_ensure_existing_document_does_not_fetch():
    class Session:
        async def get(self, model, identity):
            assert identity == ("wiki", "local.mediawiki")
            return object()

    @asynccontextmanager
    async def session():
        yield Session()

    # No fetch_document attribute: ensuring an existing row must not need it.
    await Indexer(context_for(session)).ensure_document_indexed(
        SimpleNamespace(id="wiki"), "local.mediawiki",
    )


@pytest.mark.parametrize("maintenance", [False, True])
async def test_index_all_maintenance_is_explicit(monkeypatch, maintenance):
    context = context_for()
    context.collections = {"wiki": SimpleNamespace(id="wiki")}
    async def build():
        pass
    context.build_collections = build
    indexer = Indexer(context)
    events = []
    async def maintain(source):
        events.append(source.id)
    monkeypatch.setattr(indexer, "maintain_collection", maintain)
    await indexer.index_all(index=False, summarize_chunks=False, summarize_documents=False,
                            maintenance=maintenance)
    assert events == (["wiki"] if maintenance else [])


@pytest.mark.parametrize("flags, expected", [
    ([], (True, True, True, False)),
    (["--maintenance"], (False, False, False, True)),
    (["--maintenance", "--index"], (True, False, False, True)),
    (["--summarize-documents"], (False, False, True, False)),
])
async def test_cli_maintenance_is_an_independent_operation(monkeypatch, flags, expected):
    import sys
    from mcp_indexer import indexer as module
    calls = []
    initialized = []
    class FakeIndexer:
        def __init__(self, *args, **kwargs):
            pass
        async def index_all(self, **kwargs):
            calls.append(tuple(kwargs[k] for k in ("index", "summarize_chunks", "summarize_documents", "maintenance")))
        async def wait_for_idle(self):
            pass
    monkeypatch.setattr(sys, "argv", ["indexer", "-c", "example.toml", *flags])
    monkeypatch.setattr(module, "Indexer", FakeIndexer)
    async def init_db():
        initialized.append(True)
    context = SimpleNamespace(init_db=init_db)
    monkeypatch.setattr(module.Context, "build_context", lambda paths: context)
    await module.main()
    assert initialized == [True]
    assert calls == [expected]


@pytest.mark.parametrize("deleted", [False, True])
async def test_stale_summary_cannot_restore_replaced_or_deleted_document(deleted):
    now = datetime(2026, 1, 1)
    current = None if deleted else SimpleNamespace(last_modified=now + timedelta(days=1), summary="")
    class Session:
        async def execute(self, statement):
            assert statement._for_update_arg is not None
            return SimpleNamespace(scalar_one_or_none=lambda: current)
        async def commit(self):
            raise AssertionError("Stale summary must not commit")
    @asynccontextmanager
    async def session():
        yield Session()
    doc = SimpleNamespace(document_id="article", last_modified=now)
    assert not await Indexer(context_for(session))._update_document(SimpleNamespace(id="wiki"), doc)
