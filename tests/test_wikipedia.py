from types import SimpleNamespace

import pytest

from mcp_indexer.config import CollectionConfig
from mcp_indexer.plugins.base import DocumentNotFoundError, create_embedding_chunks
from mcp_indexer.plugins.text_source import TextFileSource
from mcp_indexer.plugins.wikipedia import WikipediaPointer, WikipediaSource


def source_for(tmp_path, **config_values):
    config = CollectionConfig(
        collection_id="wiki", tool_prefix="wiki", **config_values,
        source_blob={"directory": str(tmp_path), "xapian_directory": str(tmp_path / "xapian")},
    )
    return WikipediaSource("wiki", context=SimpleNamespace(llm=None), collection_config=config)


def test_source_default_can_be_overridden(tmp_path):
    wiki = source_for(tmp_path)
    assert wiki.indexing_mode == "indexed"
    assert wiki.separate_rerank
    assert source_for(tmp_path, indexing_mode="full").indexing_mode == "full"
    text = TextFileSource("wiki", context=wiki.context, collection_config=wiki.config)
    assert text.indexing_mode == "full"
    assert not text.separate_rerank


async def test_wikipedia_raw_text_round_trip(tmp_path, monkeypatch):
    text = "{{Infobox|name=Élan}}\nLead text.\n\n== History ==\nCafé history.\n"
    filename = "100% Élan.mediawiki"
    (tmp_path / filename).write_text(text)
    (tmp_path / "ignore.txt").write_text("Not an article")
    source = source_for(tmp_path)
    # Raw-text chunking is independent of the optional Xapian binding.
    monkeypatch.setattr(source, "_xapian_keywords", lambda title: [])
    pointers = [p async for p in source.get_documents()]
    assert len(pointers) == 1
    pointer = pointers[0]
    assert isinstance(pointer, WikipediaPointer)
    assert pointer.document_id == source.encode_document_path(filename)
    metadata = await pointer.get_metadata()
    assert metadata["title"] == "100% Élan"
    assert metadata["title_strength"] == 10
    assert metadata["keywords"] == []
    semantic = [chunk async for chunk in pointer.get_chunks(1, 8)]
    assert len(semantic) == 2
    assert "".join("".join(parts) for _, parts in semantic) == text
    for meta, parts in semantic:
        assert "".join(await pointer.fetch_chunk(meta)) == "".join(parts)
        for embedding_meta, embedding_text in create_embedding_chunks(meta, parts, 1, 8):
            assert await source.fetch_chunk(pointer.document_id, embedding_meta, "embedding") == embedding_text


async def test_missing_fetch_and_missing_root_are_distinct(tmp_path):
    source = source_for(tmp_path)
    with pytest.raises(DocumentNotFoundError):
        source.fetch_document("absent.mediawiki")
    path = tmp_path / "present.mediawiki"
    path.write_text("present")
    pointer = source.fetch_document(path.name)
    path.unlink()
    with pytest.raises(DocumentNotFoundError):
        await pointer.fetch_chunk({"b": 0, "e": 7})
    source.source_config.directory = tmp_path / "unmounted"
    with pytest.raises(OSError):
        source.fetch_document("absent.mediawiki")


def test_path_escape_is_rejected(tmp_path):
    source = source_for(tmp_path)
    with pytest.raises(ValueError):
        source.fetch_document("..%2Fescape.mediawiki")


async def test_native_search_maps_ids_without_fetching_files(tmp_path, monkeypatch):
    xapian = pytest.importorskip("xapian")
    source = source_for(tmp_path)
    database = xapian.WritableDatabase(str(tmp_path / "xapian"), xapian.DB_CREATE_OR_OPEN)
    try:
        for name in ["100% Élan", "Other"]:
            document = xapian.Document()
            document.set_data(name)
            document.add_boolean_term("Q" + name)
            generator = xapian.TermGenerator()
            generator.set_document(document)
            generator.set_stemmer(xapian.Stem("english"))
            generator.index_text("physics running")
            generator.index_text("Physics", 1, "C")
            generator.index_text(name, 1, "T")
            database.add_document(document)
        database.commit()
    finally:
        database.close()
    (tmp_path / "100% Élan.mediawiki").write_text("Article")
    metadata = await source.fetch_document("100%25%20%C3%89lan.mediawiki").get_metadata()
    assert metadata["keywords"] == ["physics"]
    def unexpected_fetch(*args):
        raise AssertionError("Native discovery must not probe article presence")
    monkeypatch.setattr(source, "fetch_document", unexpected_fetch)
    hits = await source.native_search("cat:Physics", document_limit=2, chunk_limit=0)
    assert {hit.document_id for hit in hits.hits} == {
        source.encode_document_path("100% Élan.mediawiki"), "Other.mediawiki",
    }
    assert len((await source.native_search("physics", document_limit=1, chunk_limit=0)).hits) == 1
    assert await source.native_search("physics", document_limit=0, chunk_limit=5) is None
    assert await source.native_search(" ", document_limit=2, chunk_limit=0) is None


def _native_fixture(tmp_path, names, monkeypatch):
    import sys
    requests = []

    class Database:
        def __init__(self, path):
            pass
        def close(self):
            pass

    class Parser:
        STEM_SOME = FLAG_DEFAULT = FLAG_BOOLEAN_ANY_CASE = 1
        def set_database(self, value): pass
        def set_stemmer(self, value): pass
        def set_stemming_strategy(self, value): pass
        def set_default_op(self, value): pass
        def add_prefix(self, name, term): pass
        def parse_query(self, query, flags): return query

    class Enquire:
        ASCENDING = 1
        def __init__(self, database): pass
        def set_docid_order(self, order): assert order == self.ASCENDING
        def set_query(self, query): pass
        def get_mset(self, offset, count):
            requests.append((offset, count))
            return [SimpleNamespace(document=SimpleNamespace(get_data=lambda name=name: name.encode()),
                                    weight=1.0) for name in names[offset:offset + count]]

    monkeypatch.setitem(sys.modules, "xapian", SimpleNamespace(
        Database=Database, QueryParser=Parser, Enquire=Enquire,
        Stem=lambda language: language, Query=SimpleNamespace(OP_OR=1),
    ))
    async def inline(function, *args):
        return function(*args)
    monkeypatch.setattr("asyncio.to_thread", inline)
    return source_for(tmp_path), requests


async def test_native_empty_scan_advances_past_filtering_budget(tmp_path, monkeypatch):
    source, requests = _native_fixture(tmp_path, [f"bad/{i}" for i in range(100)] + ["Valid"], monkeypatch)
    batch = await source.native_search("physics", document_limit=1, chunk_limit=0)
    assert batch.hits == []
    assert batch.document_offset == 100
    batch = await source.native_search("physics", document_limit=1, chunk_limit=0,
                                       document_offset=batch.document_offset)
    assert [hit.document_id for hit in batch.hits] == ["Valid.mediawiki"]
    assert batch.document_offset == 101
    assert requests[1][0] == 100  # Resume directly; no prefix rescan.
    assert await source.native_search("physics", document_limit=1, chunk_limit=0,
                                      document_offset=101) is None


async def test_native_offset_stops_at_last_examined_match_and_preserves_duplicates(tmp_path, monkeypatch):
    source, requests = _native_fixture(tmp_path, ["Same", "Same", "Third"], monkeypatch)
    batches = []
    offset = 0
    for _ in range(3):
        batch = await source.native_search("physics", document_limit=1, chunk_limit=0,
                                           document_offset=offset)
        batches.append(batch)
        offset = batch.document_offset
    assert [batch.document_offset for batch in batches] == [1, 2, 3]
    assert [offset for offset, _ in requests] == [0, 1, 2]
    assert [batch.hits[0].document_id for batch in batches] == [
        "Same.mediawiki", "Same.mediawiki", "Third.mediawiki",
    ]
    assert await source.native_search("physics", document_limit=1, chunk_limit=0,
                                      document_offset=3) is None


async def test_native_discovery_failure_propagates(tmp_path):
    xapian = pytest.importorskip("xapian")
    source = source_for(tmp_path)
    with pytest.raises(xapian.Error):
        await source.native_search("physics", document_limit=1, chunk_limit=0)


def _lead_wikitext() -> str:
    return (
        "{{Short description|A fictional article}}\n"
        "{{Infobox\n"
        "|name = Test article\n"
        "|caption = an infobox\n"
        "}}\n"
        "[[File:example.jpg|thumb|An image]]\n"
        "'''Test article''' is a thing. It is {{inline template|a=1}} described in prose.\n"
        "\n"
        "== History ==\n"
        "History body that must not appear in the summary.\n"
        "\n"
        "== See also ==\n"
        "See also body.\n"
    )


async def test_wikipedia_summary_is_structural_lead(tmp_path):
    pytest.importorskip("mwparserfromhell")
    (tmp_path / "Test article.mediawiki").write_text(_lead_wikitext(), encoding="utf-8")
    source = source_for(tmp_path)
    summary = await source.get_document_summary("Test%20article.mediawiki")

    # The lead renders to plain text: prose survives, wikilinks become their
    # display text, and nothing after the first heading does.
    assert summary == "Test article is a thing. It is described in prose."
    for markup in ("{{", "[[", "'''", "Short description", "Infobox",
                   "File:", "History body", "See also body"):
        assert markup not in summary


async def test_wikipedia_summary_empty_without_prose(tmp_path):
    pytest.importorskip("mwparserfromhell")
    # Only block markup and a magic word: no actual introductory prose.
    (tmp_path / "Empty.mediawiki").write_text(
        "{{Short description|no prose here}}\n"
        "{{Infobox}}\n"
        "__NOTOC__\n",
        encoding="utf-8",
    )
    source = source_for(tmp_path)
    assert await source.get_document_summary("Empty.mediawiki") == ""


async def test_wikipedia_summary_missing_document(tmp_path):
    pytest.importorskip("mwparserfromhell")
    source = source_for(tmp_path)
    with pytest.raises(DocumentNotFoundError):
        await source.get_document_summary("absent.mediawiki")
