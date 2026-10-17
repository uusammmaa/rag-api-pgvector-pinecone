-- Schema for the pgvector store (table rag_chunks).
-- The API applies this automatically on first use (PGVECTOR_AUTO_MIGRATE=true).
-- Run it by hand, for example in the Supabase SQL editor, if the app user cannot
-- create extensions. vector(1536) must match EMBEDDING_DIM.
-- Kept in sync with app/vectorstores/pgvector.py by tests/test_ops.py.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    id          text PRIMARY KEY,
    doc_id      text NOT NULL,
    chunk_index integer NOT NULL,
    content     text NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(1536) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_chunks_doc_id_idx ON rag_chunks (doc_id);
