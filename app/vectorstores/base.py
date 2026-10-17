"""Vector store interface. The RAG pipeline only talks to this abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.rag.chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    score: float
    chunk_index: int
    page: int | None = None
    section: str | None = None


def chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}-{index}"


def chunk_metadata(
    doc_id: str, chunk: Chunk, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Flat metadata shared by every backend. ``None`` values are dropped because
    Pinecone rejects nulls."""
    meta: dict[str, Any] = {
        "doc_id": doc_id,
        "chunk_index": chunk.index,
        "text": chunk.text,
        "page": chunk.page,
        "section": chunk.section,
        **(extra or {}),
    }
    return {k: v for k, v in meta.items() if v is not None}


class VectorStore(ABC):
    name: str

    @abstractmethod
    def upsert(
        self,
        doc_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store chunks and their vectors under ``doc_id``. Idempotent per chunk id."""

    @abstractmethod
    def query(self, vector: Sequence[float], doc_id: str, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks of ``doc_id`` most similar to ``vector``,
        best first, with cosine similarity scores."""

    @staticmethod
    def _check_lengths(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
