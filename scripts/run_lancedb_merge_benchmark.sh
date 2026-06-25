#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/venv/bin/python}"
DB_URI="${DB_URI:-/tmp/lancedb-merge-benchmark-700k}"
TABLE_NAME="${TABLE_NAME:-benchmark_chunks}"
SEED_DOCUMENTS="${SEED_DOCUMENTS:-190000}"
SEED_CHUNK_ROWS="${SEED_CHUNK_ROWS:-700000}"
BATCH_DOCUMENTS="${BATCH_DOCUMENTS:-300}"
BATCH_CHUNK_ROWS="${BATCH_CHUNK_ROWS:-3500}"
ROUNDS="${ROUNDS:-100}"
KEEP_DB="${KEEP_DB:-0}"

ARGS=(
  "scripts/benchmark_lancedb_merge_insert.py"
  "--db-uri" "$DB_URI"
  "--table-name" "$TABLE_NAME"
  "--seed-documents" "$SEED_DOCUMENTS"
  "--seed-chunk-rows" "$SEED_CHUNK_ROWS"
  "--batch-documents" "$BATCH_DOCUMENTS"
  "--batch-chunk-rows" "$BATCH_CHUNK_ROWS"
  "--rounds" "$ROUNDS"
)

if [[ "$KEEP_DB" == "1" ]]; then
  ARGS+=("--keep-db")
fi

echo "Running LanceDB merge benchmark"
echo "  db_uri=$DB_URI"
echo "  table_name=$TABLE_NAME"
echo "  seed_documents=$SEED_DOCUMENTS"
echo "  seed_chunk_rows=$SEED_CHUNK_ROWS"
echo "  batch_documents=$BATCH_DOCUMENTS"
echo "  batch_chunk_rows=$BATCH_CHUNK_ROWS"
echo "  rounds=$ROUNDS"
echo "  keep_db=$KEEP_DB"

exec "$PYTHON_BIN" "${ARGS[@]}"
