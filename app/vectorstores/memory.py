"""In-process vector store for tests, evals and local runs without a database."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any

from app.rag.chunking import Chunk
from app.vectorstores.base import RetrievedChunk, VectorStore, chunk_id, chunk_metadata


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore(VectorStore):
    name = "memory"

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, tuple[list[float], dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def upsert(
        self,
        doc_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._check_lengths(chunks, vectors)
        with self._lock:
            records = self._docs.setdefault(doc_id, {})
            for chunk, vector in zip(chunks, vectors, strict=True):
                records[chunk_id(doc_id, chunk.index)] = (
                    list(vector),
                    chunk_metadata(doc_id, chunk, extra_metadata),
                )

    def query(self, vector: Sequence[float], doc_id: str, top_k: int) -> list[RetrievedChunk]:
        with self._lock:
            records = list(self._docs.get(doc_id, {}).items())
        scored = sorted(
            ((cosine(vector, vec), cid, meta) for cid, (vec, meta) in records),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            RetrievedChunk(
                id=cid,
                text=meta["text"],
                score=score,
                chunk_index=meta["chunk_index"],
                page=meta.get("page"),
                section=meta.get("section"),
            )
            for score, cid, meta in scored[:top_k]
        ]
