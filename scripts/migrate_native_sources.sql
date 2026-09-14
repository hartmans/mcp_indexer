-- Manual PostgreSQL migration. Review and back up the production database first.
-- Run with application writers stopped and psql -v ON_ERROR_STOP=1 -f <this file>.
-- All changes, including orphan-summary cleanup, are transactional.
BEGIN;
LOCK TABLE document, document_chunk, chunk_summary IN ACCESS EXCLUSIVE MODE;

-- These summaries have no owning document and cannot be retrieved as documents.
DELETE FROM chunk_summary s
WHERE NOT EXISTS (
    SELECT 1 FROM document d
    WHERE d.collection_id = s.collection_id AND d.document_id = s.document_id
);

-- Existing constraints may have PostgreSQL-generated names. Resolve by endpoints.
DO $migration$
DECLARE constraint_record record;
BEGIN
    FOR constraint_record IN
        SELECT conrelid::regclass AS relation, conname
        FROM pg_constraint
        WHERE contype = 'f' AND (
            (conrelid = 'document_chunk'::regclass AND confrelid IN
                ('document'::regclass, 'chunk_summary'::regclass))
            OR (conrelid = 'chunk_summary'::regclass AND confrelid = 'document'::regclass)
        )
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I',
                       constraint_record.relation, constraint_record.conname);
    END LOOP;
END
$migration$;

ALTER TABLE document_chunk ADD CONSTRAINT fk_document_chunk_document
    FOREIGN KEY (collection_id, document_id)
    REFERENCES document (collection_id, document_id)
    ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE chunk_summary ADD CONSTRAINT fk_chunk_summary_document
    FOREIGN KEY (collection_id, document_id)
    REFERENCES document (collection_id, document_id)
    ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE document_chunk ADD CONSTRAINT fk_document_chunk_summary
    FOREIGN KEY (collection_id, document_id, summary_span)
    REFERENCES chunk_summary (collection_id, document_id, summary_span)
    ON DELETE SET NULL (summary_span) DEFERRABLE INITIALLY IMMEDIATE;

COMMIT;

-- A later collection rename can use SET CONSTRAINTS ALL DEFERRED in a transaction
-- and update collection_id in document, document_chunk, chunk_summary and
-- failed_document. Local document_id/chunk_id values do not change.
