"""Vector store abstraction: the in-memory backend for real, Pinecone and pgvector
against mocked clients so no network or database is needed."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.errors import UpstreamError
from app.rag.chunking import Chunk
from app.vectorstores.base import chunk_metadata
from app.vectorstores.memory import InMemoryVectorStore, cosine
from app.vectorstores.pgvector import PgVectorStore, schema_sql, to_pgvector
from app.vectorstores.pinecone import PineconeVectorStore


def make_chunks(n: int, section: str | None = None) -> list[Chunk]:
    return [Chunk(f"chunk {i}", i, i * 10, i * 10 + 7, section, page=1) for i in range(n)]


def unit(i: int, dim: int = 4) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(dim)]


# --- in-memory -------------------------------------------------------------


def test_memory_store_ranks_by_cosine_similarity() -> None:
    store = InMemoryVectorStore()
    store.upsert("doc", make_chunks(3), [unit(0), unit(1), [0.7, 0.7, 0, 0]])
    results = store.query(unit(1), doc_id="doc", top_k=3)
    assert [r.chunk_index for r in results] == [1, 2, 0]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].id == "doc-1"
    assert results[0].page == 1


def test_memory_store_scopes_queries_to_doc_id() -> None:
    store = InMemoryVectorStore()
    store.upsert("a", make_chunks(2), [unit(0), unit(1)])
    store.upsert("b", make_chunks(1), [unit(0)])
    assert {r.id for r in store.query(unit(0), doc_id="b", top_k=5)} == {"b-0"}
    assert store.query(unit(0), doc_id="missing", top_k=5) == []


def test_memory_store_respects_top_k_and_is_idempotent() -> None:
    store = InMemoryVectorStore()
    store.upsert("doc", make_chunks(4), [unit(i) for i in range(4)])
    store.upsert("doc", make_chunks(4), [unit(i) for i in range(4)])
    assert len(store.query(unit(0), doc_id="doc", top_k=2)) == 2
    assert len(store.query(unit(0), doc_id="doc", top_k=10)) == 4


def test_upsert_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        InMemoryVectorStore().upsert("doc", make_chunks(2), [unit(0)])


def test_cosine_handles_zero_vectors() -> None:
    assert cosine([0, 0], [1, 0]) == 0.0


def test_chunk_metadata_drops_none_values() -> None:
    meta = chunk_metadata("doc", Chunk("t", 0, 0, 1), {"filename": "a.pdf"})
    assert meta == {"doc_id": "doc", "chunk_index": 0, "text": "t", "filename": "a.pdf"}


# --- Pinecone (mocked index) -------------------------------------------------


def test_pinecone_upsert_batches_and_sends_flat_metadata() -> None:
    index = MagicMock()
    store = PineconeVectorStore(index)
    store.upsert("doc", make_chunks(250, section="Intro"), [unit(0)] * 250, {"filename": "a.pdf"})

    batches = [call.kwargs["vectors"] for call in index.upsert.call_args_list]
    assert [len(b) for b in batches] == [100, 100, 50]
    first = batches[0][0]
    assert first["id"] == "doc-0"
    assert first["metadata"] == {
        "doc_id": "doc",
        "chunk_index": 0,
        "text": "chunk 0",
        "page": 1,
        "section": "Intro",
        "filename": "a.pdf",
    }


def test_pinecone_query_filters_by_doc_and_maps_matches() -> None:
    index = MagicMock()
    index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(
                id="doc-3",
                score=0.91,
                metadata={"text": "hello", "chunk_index": 3.0, "page": 2.0, "section": "S"},
            )
        ]
    )
    results = PineconeVectorStore(index).query([0.1, 0.2], doc_id="doc", top_k=4)

    index.query.assert_called_once_with(
        vector=[0.1, 0.2], top_k=4, include_metadata=True, filter={"doc_id": {"$eq": "doc"}}
    )
    assert results[0].id == "doc-3"
    assert results[0].chunk_index == 3
    assert results[0].page == 2
    assert results[0].section == "S"
    assert results[0].score == pytest.approx(0.91)


def test_pinecone_errors_become_upstream_errors() -> None:
    index = MagicMock()
    index.query.side_effect = RuntimeError("boom")
    with pytest.raises(UpstreamError):
        PineconeVectorStore(index).query([0.1], doc_id="doc", top_k=1)


# --- pgvector (fake connection pool) ----------------------------------------


class FakeCursor:
    def __init__(self, log: list) -> None:
        self.log = log

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def executemany(self, sql: str, rows: list) -> None:
        self.log.append(("executemany", sql, rows))


class FakeConnection:
    def __init__(self, log: list, rows: list | None = None, fail: bool = False) -> None:
        self.log = log
        self.rows = rows or []
        self.fail = fail

    def execute(self, sql: str, params: object = None) -> SimpleNamespace:
        if self.fail:
            raise RuntimeError("connection refused")
        self.log.append(("execute", sql, params))
        return SimpleNamespace(fetchall=lambda: self.rows)

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.log)

    @contextmanager
    def transaction(self):
        yield


class FakePool:
    def __init__(self, **kwargs: object) -> None:
        self.log: list = []
        self.conn = FakeConnection(self.log, **kwargs)

    @contextmanager
    def connection(self):
        yield self.conn


def test_pgvector_migrates_schema_on_init() -> None:
    pool = FakePool()
    PgVectorStore(pool, dim=8)
    executed = [entry[1] for entry in pool.log]
    assert executed == schema_sql(8)
    assert "vector(8)" in executed[1]


def test_pgvector_skips_migration_when_disabled() -> None:
    pool = FakePool()
    PgVectorStore(pool, dim=8, auto_migrate=False)
    assert pool.log == []


def test_pgvector_upsert_writes_one_row_per_chunk() -> None:
    pool = FakePool()
    store = PgVectorStore(pool, dim=4, auto_migrate=False)
    store.upsert("doc", make_chunks(2), [unit(0), unit(1)], {"filename": "a.pdf"})

    ((kind, sql, rows),) = pool.log
    assert kind == "executemany"
    assert "ON CONFLICT (id) DO UPDATE" in sql
    row_id, doc_id, index, content, meta, vector = rows[1]
    assert (row_id, doc_id, index, content) == ("doc-1", "doc", 1, "chunk 1")
    assert meta.obj == {"doc_id": "doc", "chunk_index": 1, "page": 1, "filename": "a.pdf"}
    assert vector == "[0,1,0,0]"


def test_pgvector_query_is_scoped_and_maps_rows() -> None:
    pool = FakePool(rows=[("doc-0", 0, "hello", {"page": 3, "section": "A"}, 0.88)])
    store = PgVectorStore(pool, dim=4, auto_migrate=False)
    results = store.query(unit(2), doc_id="doc", top_k=5)

    ((_, sql, params),) = pool.log
    assert "WHERE doc_id = %(doc_id)s" in sql
    assert params == {"q": "[0,0,1,0]", "doc_id": "doc", "k": 5}
    assert results[0].text == "hello"
    assert results[0].page == 3
    assert results[0].section == "A"
    assert results[0].score == pytest.approx(0.88)


def test_pgvector_errors_become_upstream_errors() -> None:
    store = PgVectorStore(FakePool(fail=True), dim=4, auto_migrate=False)
    with pytest.raises(UpstreamError):
        store.query(unit(0), doc_id="doc", top_k=1)


def test_to_pgvector_literal() -> None:
    assert to_pgvector([0.5, -1, 1e-9]) == "[0.5,-1,1e-09]"
