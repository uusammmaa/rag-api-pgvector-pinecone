"""Pinecone backend (serverless or pod index, cosine metric)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from app.errors import UpstreamError
from app.rag.chunking import Chunk
from app.vectorstores.base import RetrievedChunk, VectorStore, chunk_id, chunk_metadata

logger = logging.getLogger(__name__)


class PineconeVectorStore(VectorStore):
    name = "pinecone"
    batch_size = 100

    def __init__(self, index: Any) -> None:
        self.index = index

    @classmethod
    def from_settings(cls, api_key: str, index_name: str) -> PineconeVectorStore:
        from pinecone import Pinecone

        return cls(Pinecone(api_key=api_key).Index(index_name))

    def upsert(
        self,
        doc_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._check_lengths(chunks, vectors)
        records = [
            {
                "id": chunk_id(doc_id, chunk.index),
                "values": list(vector),
                "metadata": chunk_metadata(doc_id, chunk, extra_metadata),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            for i in range(0, len(records), self.batch_size):
                self.index.upsert(vectors=records[i : i + self.batch_size])
        except Exception as exc:  # the SDK raises several unrelated exception types
            logger.error("pinecone upsert failed: %s", exc)
            raise UpstreamError("Vector store write failed.") from exc

    def query(self, vector: Sequence[float], doc_id: str, top_k: int) -> list[RetrievedChunk]:
        try:
            result = self.index.query(
                vector=list(vector),
                top_k=top_k,
                include_metadata=True,
                filter={"doc_id": {"$eq": doc_id}},
            )
        except Exception as exc:
            logger.error("pinecone query failed: %s", exc)
            raise UpstreamError("Vector store query failed.") from exc
        return [
            RetrievedChunk(
                id=match.id,
                text=match.metadata["text"],
                score=float(match.score),
                chunk_index=int(match.metadata.get("chunk_index", 0)),
                page=_opt_int(match.metadata.get("page")),
                section=match.metadata.get("section"),
            )
            for match in result.matches
        ]


def _opt_int(value: Any) -> int | None:
    # Pinecone returns numeric metadata as floats.
    return int(value) if value is not None else None
