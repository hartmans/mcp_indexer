# Wikipedia implementation handoff

Implemented source-default indexing policy, configuration overrides, opt-in
maintenance, missing-document exceptions, atomic document replacement, summary
write guards, the TextSource-based Wikipedia plugin, and conditional MCP
`rerank_query` input. Native queries use Xapian directly; natural-language queries
feed vector search and reranking. Native discovery currently returns document
hits, not chunk hits.

## Operations

Use `examples/wikipedia.toml` as an additional configuration fragment alongside
your server/model settings. Wikipedia defaults to `indexed`; a fragment setting
`collections.wikipedia.indexing_mode = "full"` enables exhaustive discovery.
The default for existing text sources remains `full`.

`python -m mcp_indexer.indexer -c server.toml -c examples/wikipedia.toml --maintenance`
refreshes changed persisted documents and removes missing ones, without discovering
the unindexed corpus. Add `--index`, `--summarize-chunks`, and/or
`--summarize-documents` to combine operations. No flags preserves the previous
index-and-summarize behavior without maintenance. The Python `index_all` method
also defaults `maintenance=False`; pass the other operation booleans as false for
a maintenance-only invocation.

After extracting another dump, run maintenance on the same collection. Changed
file contents must receive changed modification times. Extraction must remove
absent files and update/rebuild Xapian too; the old scripts under `/srv/wikipedia`
do not do this, and the new plugin does not run them. Searches during the update
window can see stale stored results until maintenance finishes.

## Production steps for review

1. Review `scripts/migrate_native_sources.sql` and back up the production database.
   With application writers stopped, apply it using `psql -v ON_ERROR_STOP=1 -f`
   against the intended database. It updates FK cascades/deferrability and deletes
   orphan summaries in one transaction. The cleanup is recoverable through the
   backup or transaction rollback before commit, not after commit without a backup.
   Do not use the new maintenance code against the old schema.
2. Run the application and PostgreSQL tests in an environment with the project
   dependencies and a dedicated test database. `tests/conftest.py` currently points
   to the old `/srv/datasets/asstr.db/sockets` test endpoint, absent on this host.
   `tests/test_native_persistence.py` covers replacement failure, refresh/deletion,
   summary cleanup, collection rename, and migration from legacy constraints.
3. Review the source/config and run a small end-to-end search with your actual
   embedding/summary models before starting full corpus work.

No database or Python environment has been changed. No packages were successfully
installed. Following the user's instructions, no further installations, database
setup, or approval requests will be made in this implementation session.

## Verification completed here

- `~/ai/venv/bin/python -m pytest --noconftest tests/test_config.py tests/test_xapian_positions.py -q`:
  25 tests passed. `--noconftest` avoids the application's database/dependency
  imports for these independent checks.
- Python compilation and `git diff --check` passed.
- Xapian 1.4.31 opened the real corpus read-only and reported 7,781,446 documents.
  The query parser default operator is OR, as used by the new source.
- For Wikipedia, Ada Lovelace, and Python (programming language), body term
  positions reproduced from current files matched the existing native index.
  Splitting at MediaWiki headings preserved that sequence (67, 26, and 30 spans
  respectively). Synthetic checks confirm that arbitrary embedding boundaries
  inside a word do not preserve tokenization and default stem terms lack positions.

Broader application tests cannot import in the supplied venv: `pgvector`, `mcp`,
LangChain and the PostgreSQL driver are missing. The new application regression
tests are written but unrun here; the SQL migration is also unrun. Compilation
and binding-level checks do not establish end-to-end application correctness.

Mapping body positions to semantic spans is feasible. Exact embedding-span
alignment still needs code handling mid-token cuts and stem-to-surface matching.
The initial source therefore returns article hits. Optional conversion away from
MediaWiki remains separate: template names/arguments may be more valuable to a
model than their expansion, so expansion is not assumed to improve content.
