-- Run this once in the Neon Console's SQL Editor before anything else this week.
-- It enables the pgvector extension so Postgres can store and search embeddings.

CREATE EXTENSION IF NOT EXISTS vector;

-- Verify it worked:
SELECT * FROM pg_extension WHERE extname = 'vector';
-- You should see one row back. If you see nothing, the extension didn't enable —
-- check that your Neon plan supports extensions (the free tier does).
