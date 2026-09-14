"""Binding-level checks; runnable with --noconftest without application dependencies."""
import pytest

xapian = pytest.importorskip("xapian")


def index_text(text, prefix="", generator=None, document=None):
    document = document if document is not None else xapian.Document()
    if generator is None:
        generator = xapian.TermGenerator()
        generator.set_document(document)
        generator.set_stemmer(xapian.Stem("english"))
        generator.set_termpos(1)
    generator.index_text(text, 1, prefix)
    return generator, document


def positions(document):
    return {item.term: tuple(item.positer) for item in document.termlist()}


def test_body_and_category_positions_are_distinct_sequences():
    generator, document = index_text("First category", "C")
    generator.set_termpos(1)
    generator.index_text("Second group", 1, "C")
    generator.set_termpos(1)
    generator.index_text("First title", 1, "T")
    generator.set_termpos(1)
    generator.index_text("First running runner second")
    terms = positions(document)
    assert terms[b"Cfirst"] == terms[b"Csecond"] == terms[b"Tfirst"] == terms[b"first"]
    assert terms[b"Ccategory"][0] > terms[b"Cfirst"][0]
    assert terms[b"second"][0] > terms[b"first"][0]
    assert terms[b"running"]
    assert terms[b"Zrun"] == ()  # The installed 1.4 default stems have no positions.


def test_semantic_boundaries_can_preserve_body_positions():
    spans = ["Élan running café.\n\n", "== History ==\nRunner repeats words.\n"]
    _, whole = index_text("".join(spans))
    generator, split = index_text(spans[0])
    generator.index_text(spans[1])
    assert positions(whole) == positions(split)


def test_arbitrary_embedding_boundaries_do_not_preserve_positions():
    _, whole = index_text("runner")
    generator, split = index_text("run")
    generator.index_text("ner")
    assert positions(whole) != positions(split)


def test_query_fields_and_matching_terms(tmp_path):
    writer = xapian.WritableDatabase(str(tmp_path / "xapian"), xapian.DB_CREATE_OR_OPEN)
    generator, document = index_text("running through fields")
    generator.set_termpos(1)
    generator.index_text("Physics", 1, "C")
    generator.set_termpos(1)
    generator.index_text("Example", 1, "T")
    document.set_data("Example")
    writer.add_document(document)
    writer.commit()
    writer.close()
    database = xapian.Database(str(tmp_path / "xapian"))
    try:
        parser = xapian.QueryParser()
        parser.set_database(database)
        parser.set_stemmer(xapian.Stem("english"))
        parser.set_stemming_strategy(parser.STEM_SOME)
        parser.add_prefix("cat", "C")
        parser.add_prefix("title", "T")
        enquiry = xapian.Enquire(database)
        enquiry.set_query(parser.parse_query("cat:Physics AND running"))
        matches = enquiry.get_mset(0, 5)
        assert len(matches) == 1
        assert matches[0].document.get_data() == b"Example"
        assert b"Zrun" in set(enquiry.matching_terms(matches[0]))
    finally:
        database.close()
