"""Search an extracted MediaWiki corpus through its existing Xapian index."""
import asyncio
from pathlib import Path

from .text_source import TextFileSource, TextFileSourceConfig, TextFilePointer
from ..search import DocumentHit, SearchHit


class WikipediaSourceConfig(TextFileSourceConfig):
    xapian_directory: Path


class WikipediaPointer(TextFilePointer):
    async def get_metadata(self) -> dict[str, str]:
        return {**await super().get_metadata(), "title": self.path.stem}


class WikipediaSource(TextFileSource):
    source_prefix = "wikipedia"
    default_indexing_mode = "indexed"
    separate_rerank = True
    semantic_boundary_regexps = (rb"(?m)^={2,6}[^\n]+?={2,6}[ \t]*$",)
    embedding_boundary_regexps = (rb"\n\s*\n",)

    def __init__(self, collection_id: str, *, context, collection_config, reranker=None):
        super().__init__(collection_id, context=context,
                         collection_config=collection_config, reranker=reranker)
        self.source_config = collection_config.resolve_source_config(WikipediaSourceConfig)

    def _is_included(self, path: Path) -> bool:
        return path.suffix == ".mediawiki" and super()._is_included(path)

    def fetch_document(self, document_id: str) -> WikipediaPointer:
        return WikipediaPointer(self, document_id, self._document_path(document_id))

    async def native_search(
        self, query: str, *, document_limit: int, chunk_limit: int,
    ) -> list[SearchHit]:
        if not query.strip() or document_limit <= 0:
            return []
        return await asyncio.to_thread(self._native_search, query, document_limit)

    def _native_search(self, query: str, document_limit: int) -> list[SearchHit]:
        import xapian

        database = xapian.Database(str(self.source_config.xapian_directory))
        try:
            parser = xapian.QueryParser()
            parser.set_database(database)
            parser.set_stemmer(xapian.Stem("english"))
            parser.set_stemming_strategy(parser.STEM_SOME)
            parser.set_default_op(xapian.Query.OP_OR)
            parser.add_prefix("title", "T")
            parser.add_prefix("cat", "C")
            enquiry = xapian.Enquire(database)
            enquiry.set_query(parser.parse_query(
                query, parser.FLAG_DEFAULT | parser.FLAG_BOOLEAN_ANY_CASE,
            ))
            hits = []
            seen = set()
            # Bound filtering work even if an index contains excluded/bad entries.
            offset = 0
            budget = max(100, document_limit * 10)
            while len(hits) < document_limit and offset < budget:
                matches = enquiry.get_mset(offset, min(document_limit, budget - offset))
                if not matches:
                    break
                offset += len(matches)
                for match in matches:
                    try:
                        stem = match.document.get_data().decode("utf-8")
                        if not stem or "/" in stem or "\x00" in stem:
                            continue
                        path = Path(stem + ".mediawiki")
                        if not self._is_included(path):
                            continue
                        document_id = self.encode_document_path(path.as_posix())
                        if document_id not in seen:
                            seen.add(document_id)
                            hits.append(DocumentHit(document_id, match.weight))
                    except UnicodeDecodeError:
                        continue
                    if len(hits) == document_limit:
                        break
            return hits
        finally:
            database.close()
