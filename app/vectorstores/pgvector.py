"""Postgres + pgvector backend. Works with local Postgres (docker-compose) and Supabase.

Queries are always scoped to one document, so the planner uses the ``doc_id``
B-tree index and computes exact cosine distances over that document's chunks.
There is deliberately no HNSW index: with a selective ``WHERE doc_id = ...`` an
approximate index post-filters its candidate list and can return fewer than
``top_k`` rows. Add one (with ``hnsw.iterative_scan``) only for cross-document
search.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from psycopg.types.json import Jsonb

from app.errors import UpstreamError
from app.rag.chunking import Chunk
from app.vectorstores.base import RetrievedChunk, VectorStore, chunk_id, chunk_metadata

logger = logging.getLogger(__name__)

TABLE = "rag_chunks"


def schema_sql(dim: int) -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""CREATE TABLE IF NOT EXISTS {TABLE} (
            id          text PRIMARY KEY,
            doc_id      text NOT NULL,
            chunk_index integer NOT NULL,
            content     text NOT NULL,
            metadata    jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            embedding   vector({int(dim)}) NOT NULL,
            created_at  timestamptz NOT NULL DEFAULT now()
        )""",
        f"CREATE INDEX IF NOT EXISTS {TABLE}_doc_id_idx ON {TABLE} (doc_id)",
    ]


UPSERT_SQL = f"""
INSERT INTO {TABLE} (id, doc_id, chunk_index, content, metadata, embedding)
VALUES (%s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (id) DO UPDATE SET
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding
"""

QUERY_SQL = f"""
SELECT id, chunk_index, content, metadata, 1 - (embedding <=> %(q)s::vector) AS score
FROM {TABLE}
WHERE doc_id = %(doc_id)s
ORDER BY embedding <=> %(q)s::vector
LIMIT %(k)s
"""


def to_pgvector(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.7g}" for v in vector) + "]"


class PgVectorStore(VectorStore):
    name = "pgvector"

    def __init__(self, pool: Any, dim: int, auto_migrate: bool = True) -> None:
        """``pool`` is anything with a ``connection()`` context manager, normally a
        ``psycopg_pool.ConnectionPool``."""
        self.pool = pool
        self.dim = dim
        if auto_migrate:
            self.migrate()

    @classmethod
    def from_dsn(cls, dsn: str, dim: int, auto_migrate: bool = True) -> PgVectorStore:
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(dsn, min_size=1, max_size=10, kwargs={"autocommit": True}, open=True)
        return cls(pool, dim, auto_migrate)

    def migrate(self) -> None:
        try:
            with self.pool.connection() as conn:
                for statement in schema_sql(self.dim):
                    conn.execute(statement)
        except Exception as exc:
            logger.error("pgvector migration failed: %s", exc)
            raise UpstreamError("Vector store schema setup failed.") from exc

    def upsert(
        self,
        doc_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._check_lengths(chunks, vectors)
        rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            meta = chunk_metadata(doc_id, chunk, extra_metadata)
            meta.pop("text")
            rows.append(
                (
                    chunk_id(doc_id, chunk.index),
                    doc_id,
                    chunk.index,
                    chunk.text,
                    Jsonb(meta),
                    to_pgvector(vector),
                )
            )
        try:
            with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
                cur.executemany(UPSERT_SQL, rows)
        except Exception as exc:
            logger.error("pgvector upsert failed: %s", exc)
            raise UpstreamError("Vector store write failed.") from exc

    def query(self, vector: Sequence[float], doc_id: str, top_k: int) -> list[RetrievedChunk]:
        params = {"q": to_pgvector(vector), "doc_id": doc_id, "k": top_k}
        try:
            with self.pool.connection() as conn:
                rows = conn.execute(QUERY_SQL, params).fetchall()
        except Exception as exc:
            logger.error("pgvector query failed: %s", exc)
            raise UpstreamError("Vector store query failed.") from exc
        return [
            RetrievedChunk(
                id=row[0],
                chunk_index=row[1],
                text=row[2],
                page=(row[3] or {}).get("page"),
                section=(row[3] or {}).get("section"),
                score=float(row[4]),
            )
            for row in rows
        ]
