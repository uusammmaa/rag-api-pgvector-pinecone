"""Round trip against a real Postgres with pgvector. Skipped unless TEST_DATABASE_URL
is set (CI provides one through a service container; locally use docker compose)."""

from __future__ import annotations

import os
import uuid

import pytest

from app.rag.chunking import Chunk
from app.rag.embeddings import HashEmbedder

DSN = os.environ.get("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set"),
]


@pytest.fixture(scope="module")
def store():
    from app.vectorstores.pgvector import PgVectorStore

    s = PgVectorStore.from_dsn(DSN, dim=64)
    yield s
    s.pool.close()


def test_upsert_and_query_round_trip(store) -> None:
    embedder = HashEmbedder(dim=64)
    texts = [
        "Congress shall have power to declare war",
        "The President is commander in chief of the army and navy",
        "Rules for using commas in a series of three terms",
    ]
    chunks = [Chunk(t, i, 0, len(t), section="S", page=i + 1) for i, t in enumerate(texts)]
    doc_a, doc_b = f"it-{uuid.uuid4()}", f"it-{uuid.uuid4()}"
    store.upsert(doc_a, chunks, embedder.embed_documents(texts), {"filename": "a.pdf"})
    store.upsert(doc_b, chunks[:1], embedder.embed_documents(texts[:1]))
    # Re-upserting is idempotent.
    store.upsert(doc_a, chunks, embedder.embed_documents(texts), {"filename": "a.pdf"})

    results = store.query(embedder.embed_query("who can declare war"), doc_id=doc_a, top_k=2)
    assert len(results) == 2
    assert results[0].id == f"{doc_a}-0"
    assert results[0].page == 1
    assert results[0].section == "S"
    assert results[0].score > results[1].score

    assert len(store.query(embedder.embed_query("war"), doc_id=doc_b, top_k=5)) == 1
