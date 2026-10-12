from datetime import datetime
from pathlib import Path

import pytest

from mcp_indexer.context import Context, SOURCE_REGISTRY
from mcp_indexer.plugins.file_source import (
    FileSource, FileSourceConfig,
    MARKDOWN_POLICY, HTML_POLICY, RST_POLICY, TEXT_POLICY, extension_policy,
)
from mcp_indexer.plugins.plugin import HANDLER_REGISTRY
from mcp_indexer.plugins.text_document import TextDocument, split_text
from mcp_indexer.plugins.wikipedia import WikipediaDocument


class SimpleCfg:
    def __init__(self, directory, **kwargs):
        self.source_blob = {"directory": directory, **kwargs}

    def resolve_source_config(self, model):
        return model.model_validate(self.source_blob)


@pytest.fixture
def mock_context(test_context):
    return Context(engine=test_context.engine, session_factory=test_context.session_factory, embedding=None, llm=None, config=None)


@pytest.fixture
def temp_file_env(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nParagraph", encoding="utf-8")
    (tmp_path / "b.markdown").write_text("## Sub\n\nParagraph", encoding="utf-8")
    (tmp_path / "c.html").write_text("<h1>T</h1>\n<p>Paragraph</p>", encoding="utf-8")
    (tmp_path / "d.htm").write_text("<h2>T</h2>\n<p>Paragraph</p>", encoding="utf-8")
    (tmp_path / "e.rst").write_text("Title\n=======\n\nParagraph", encoding="utf-8")
    (tmp_path / "f.txt").write_text("Plain paragraph.", encoding="utf-8")
    (tmp_path / "g.foobar").write_text("Unknown format.", encoding="utf-8")
    (tmp_path / "noext").write_text("No extension.", encoding="utf-8")
    (tmp_path / "UPPER.MD").write_text("# Upper\n\nParagraph", encoding="utf-8")
    return tmp_path


def test_registered_names():
    # FileSource registers as "file"; the removed text source is gone.
    assert SOURCE_REGISTRY["file"] is FileSource
    assert "text" not in SOURCE_REGISTRY
    # Handler-name registry is separate and names the plugin classes.
    assert HANDLER_REGISTRY["text"] is TextDocument
    assert HANDLER_REGISTRY["wikipedia"] is WikipediaDocument


def test_extension_policies_are_code_defined():
    assert extension_policy(Path("a.md")) is MARKDOWN_POLICY
    # Alias extensions share the same policy object.
    assert extension_policy(Path("a.markdown")) is MARKDOWN_POLICY
    assert extension_policy(Path("a.html")) is HTML_POLICY
    assert extension_policy(Path("a.htm")) is HTML_POLICY
    assert extension_policy(Path("a.rst")) is RST_POLICY
    # Suffix lookup normalizes case.
    assert extension_policy(Path("A.MD")) is MARKDOWN_POLICY
    assert extension_policy(Path("A.HTML")) is HTML_POLICY
    # Unknown and extensionless files fall back to the default text policy.
    assert extension_policy(Path("a.unknownext")) is TEXT_POLICY
    assert extension_policy(Path("noext")) is TEXT_POLICY
    # Every policy names a handler; the handler key is intact.
    for policy in (MARKDOWN_POLICY, HTML_POLICY, RST_POLICY, TEXT_POLICY):
        assert policy["handler"] == "text"
    # The plain-text fallback treats a line of separator punctuation as its
    # only semantic boundary.
    assert len(TEXT_POLICY["semantic_boundary_regexps"]) == 1


@pytest.mark.asyncio
async def test_discovery_selects_plugins_by_extension(temp_file_env, mock_context):
    source = FileSource("col", context=mock_context, collection_config=SimpleCfg(temp_file_env))

    docs = {doc.document_id: doc async for doc in source.get_documents()}

    assert set(docs) == {
        "a.md", "b.markdown", "c.html", "d.htm", "e.rst",
        "f.txt", "g.foobar", "noext", "UPPER.MD",
    }
    for doc in docs.values():
        assert isinstance(doc, TextDocument)
    # The extension picked the policy for each file.
    assert docs["a.md"].policy is MARKDOWN_POLICY
    assert docs["b.markdown"].policy is docs["a.md"].policy  # alias shares it
    assert docs["UPPER.MD"].policy is MARKDOWN_POLICY        # case normalized
    assert docs["c.html"].policy is HTML_POLICY
    assert docs["d.htm"].policy is HTML_POLICY
    assert docs["e.rst"].policy is RST_POLICY
    assert docs["f.txt"].policy is TEXT_POLICY
    assert docs["g.foobar"].policy is TEXT_POLICY
    assert docs["noext"].policy is TEXT_POLICY


@pytest.mark.asyncio
async def test_direct_fetch_matches_enumeration(temp_file_env, mock_context):
    source = FileSource("col", context=mock_context, collection_config=SimpleCfg(temp_file_env))

    direct = await source.fetch_document("a.md")
    docs = [doc async for doc in source.get_documents()]
    enumerated = next(doc for doc in docs if doc.document_id == "a.md")

    assert type(direct) is type(enumerated) is TextDocument
    assert direct.policy is enumerated.policy is MARKDOWN_POLICY
    assert direct.document_id == enumerated.document_id
    assert await direct.get_metadata() == await enumerated.get_metadata()


@pytest.mark.asyncio
async def test_missing_handler_is_an_error(temp_file_env, mock_context):
    class NoHandlerSource(FileSource):
        def policy_for_path(self, path):
            return {"handler": "nope"}

    source = NoHandlerSource("col", context=mock_context, collection_config=SimpleCfg(temp_file_env))
    with pytest.raises(ValueError, match="nope"):
        await source.fetch_document("f.txt")

    class NoHandlerKeySource(FileSource):
        def policy_for_path(self, path):
            return {"semantic_boundary_regexps": ()}

    bare = NoHandlerKeySource("col", context=mock_context, collection_config=SimpleCfg(temp_file_env))
    with pytest.raises(ValueError):
        await bare.fetch_document("f.txt")


@pytest.mark.asyncio
async def test_metadata_isolation_with_reused_policy_and_metadata(temp_file_env, mock_context):
    source = FileSource("col", context=mock_context, collection_config=SimpleCfg(temp_file_env))
    policy = MARKDOWN_POLICY
    incoming = {"title": "shared", "custom": "original"}
    mtime = datetime.now()

    # No content access happens here, so a placeholder bridge suffices.
    stub = object()
    doc_a = TextDocument("a.md", source, stub, policy, mtime, metadata=incoming)  # type: ignore[arg-type]
    doc_b = TextDocument("b.md", source, stub, policy, mtime, metadata=incoming)  # type: ignore[arg-type]

    # The constructor shallow-copies incoming metadata: mutating the
    # original dict after construction does not affect the documents.
    incoming["custom"] = "mutated"
    incoming["late"] = "added"
    assert (await doc_a.get_metadata())["custom"] == "original"
    assert "late" not in await doc_a.get_metadata()
    assert (await doc_b.get_metadata())["custom"] == "original"

    # Repeated get_metadata calls return fresh copies: mutating one result
    # does not leak into the document or into other results.
    first = await doc_a.get_metadata()
    first["custom"] = "tampered"
    first["injected"] = "x"
    assert (await doc_a.get_metadata())["custom"] == "original"
    assert "injected" not in await doc_a.get_metadata()
    assert (await doc_b.get_metadata())["custom"] == "original"

    # A document may enrich its retained metadata; get_metadata reflects it.
    doc_a.metadata["custom"] = "enriched"
    assert (await doc_a.get_metadata())["custom"] == "enriched"
    assert (await doc_b.get_metadata())["custom"] == "original"

    # Construction leaves the policy and handler key unchanged, even with
    # a policy shared between documents.
    assert policy["handler"] == "text"
    assert set(policy) == {"handler", "semantic_boundary_regexps", "embedding_boundary_regexps"}
    assert doc_a.policy is doc_b.policy is policy


@pytest.mark.asyncio
async def test_policy_from_source_is_read_only_input(temp_file_env, mock_context):
    original = dict(MARKDOWN_POLICY, semantic_boundary_regexps=tuple(MARKDOWN_POLICY["semantic_boundary_regexps"]))
    source = FileSource("col", context=mock_context, collection_config=SimpleCfg(temp_file_env))
    doc = await source.fetch_document("a.md")
    # The construction did not add, drop, or rewrite any policy key.
    assert doc.policy == original
    assert doc.policy["handler"] == "text"
    assert MARKDOWN_POLICY == original


@pytest.mark.asyncio
async def test_encoding_and_summary_length_come_from_source_config(temp_file_env, mock_context):
    source = FileSource(
        "col", context=mock_context,
        collection_config=SimpleCfg(temp_file_env, summary_length=123, encoding="utf-8"),
    )
    doc = await source.fetch_document("f.txt")
    assert doc.encoding == "utf-8"
    assert doc.summary_length == 123
    # The document reads the live source config, not a construction snapshot.
    source.source_config = FileSourceConfig(directory=temp_file_env, summary_length=456)
    assert doc.summary_length == 456


@pytest.mark.asyncio
async def test_file_source_discovery(temp_file_env, mock_context):
    source = FileSource("test_col", context=mock_context, collection_config=SimpleCfg(temp_file_env))

    docs = [doc async for doc in source.get_documents()]

    assert len(docs) == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n", b"\r"])
async def test_byte_offsets_hold_across_newline_conventions(tmp_path, mock_context, line_ending):
    """b/e are byte offsets into the file, so they must hold regardless of
    the file's newline convention. This is the guard for reading via
    read_bytes().decode() rather than read_text(): universal-newline
    translation (CRLF/CR -> LF) would make the decoded text -- and hence
    the offsets computed from it -- disagree with the raw file bytes.
    """
    lines = [b"RST title", b"===", b"Body one", b"---", b"Body two"]
    content = line_ending.join(lines)
    (tmp_path / "nl.rst").write_bytes(content)

    source = FileSource("col", context=mock_context, collection_config=SimpleCfg(tmp_path))
    pointer = await source.fetch_document("nl.rst")

    chunks = []
    async for meta, cands in pointer.get_chunks(min_size=1, max_size=10_000):
        chunks.append((meta, cands))
    assert chunks, "expected at least one semantic chunk"

    # Whole-document round-trip: the decoded candidates must reproduce the
    # file byte-for-byte. A read_text() with newline translation would
    # shorten CRLF/CR content and fail this.
    assert "".join("".join(cands) for _m, cands in chunks) == content.decode()

    # Per chunk: the stored b/e must select exactly this chunk's text from
    # the raw file bytes.
    for meta, cands in chunks:
        assert content[meta["b"]:meta["e"]].decode() == "".join(cands)

    # Contiguity: the chunks partition the file with no gaps or overlaps.
    ordered = sorted(chunks, key=lambda c: c[0]["b"])
    assert ordered[0][0]["b"] == 0
    for prev, cur in zip(ordered, ordered[1:]):
        assert prev[0]["e"] == cur[0]["b"]
    assert ordered[-1][0]["e"] == len(content)

    # fetch_chunk reads the stored byte span and must return the same text.
    for meta, cands in chunks:
        assert await pointer.fetch_chunk(dict(meta)) == "".join(cands)


@pytest.mark.asyncio
async def test_multibyte_byte_offsets_round_trip(tmp_path, mock_context):
    """b/e are byte offsets: for multibyte UTF-8 text they must address
    bytes, not characters."""
    text = "# Café\n\nÉlan and 中文 content.\n\nMore text."
    (tmp_path / "multi.md").write_text(text, encoding="utf-8")
    raw = (tmp_path / "multi.md").read_bytes()
    assert len(raw) != len(text)  # guard: the content is actually multibyte

    source = FileSource("col", context=mock_context, collection_config=SimpleCfg(tmp_path))
    pointer = await source.fetch_document("multi.md")

    chunks = []
    async for meta, cands in pointer.get_chunks(min_size=1, max_size=10_000):
        chunks.append((meta, cands))
    assert chunks
    assert "".join("".join(cands) for _m, cands in chunks) == text
    for meta, cands in chunks:
        assert raw[meta["b"]:meta["e"]].decode("utf-8") == "".join(cands)
        assert await pointer.fetch_chunk(dict(meta)) == "".join(cands)


@pytest.mark.asyncio
async def test_file_source_pointer_retrieval(temp_file_env, mock_context):
    path = temp_file_env / "f.txt"
    path.write_text("Semantic 1\n\nSemantic 2", encoding="utf-8")

    source = FileSource("col", context=mock_context, collection_config=SimpleCfg(temp_file_env))
    source.source_config = FileSourceConfig(directory=temp_file_env)

    pointer = await source.fetch_document("f.txt")
    assert isinstance(pointer, TextDocument)

    chunks = []
    async for chunk_info in pointer.get_chunks(min_size=1, max_size=100):
        chunks.append(chunk_info)

    assert len(chunks) == 1
    meta, _ = chunks[0]
    text = await pointer.fetch_chunk(meta)
    assert "Semantic 1" in text
    meta_dict = await pointer.get_metadata()
    assert meta_dict["title"] == "f.txt"


@pytest.mark.asyncio
async def test_file_source_document_id_escapes_unsafe_path_characters(tmp_path, mock_context):
    nested_dir = tmp_path / "team's notes"
    nested_dir.mkdir()
    weird_path = nested_dir / "alpha?.txt"
    weird_path.write_text("Encoded path doc", encoding="utf-8")

    source = FileSource("col", context=mock_context, collection_config=SimpleCfg(tmp_path))

    docs = [doc async for doc in source.get_documents()]

    assert len(docs) == 1
    document_id = docs[0].document_id
    assert "%27" in document_id
    assert "%3F" in document_id
    assert "team%27s%20notes/alpha%3F.txt" in document_id

    pointer = await source.fetch_document(document_id)
    async with pointer.bridge.path() as resolved:
        assert resolved == weird_path