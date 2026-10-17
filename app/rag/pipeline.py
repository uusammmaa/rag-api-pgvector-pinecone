"""Ingestion and retrieval: the provider-agnostic core shared by the API and evals."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

from app.rag.chunking import Chunk, Chunker
from app.rag.embeddings import Embedder
from app.rag.pdf import ExtractedDocument
from app.vectorstores.base import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    chunks: list[Chunk]
    pages: int


def chunk_document(document: ExtractedDocument, chunker: Chunker) -> list[Chunk]:
    """Chunk extracted text and tag each chunk with the page it starts on."""
    return [
        replace(chunk, page=document.page_for_offset(chunk.start))
        for chunk in chunker.split(document.text)
    ]


def ingest(
    document: ExtractedDocument,
    doc_id: str,
    *,
    chunker: Chunker,
    embedder: Embedder,
    store: VectorStore,
    extra_metadata: dict[str, str] | None = None,
) -> IngestResult:
    started = time.perf_counter()
    chunks = chunk_document(document, chunker)
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    store.upsert(doc_id, chunks, vectors, extra_metadata)
    logger.info(
        "ingested document",
        extra={
            "doc_id": doc_id,
            "chunks": len(chunks),
            "pages": document.pages,
            "store": store.name,
            "chunker": chunker.name,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return IngestResult(doc_id=doc_id, chunks=chunks, pages=document.pages)


def retrieve(
    question: str, doc_id: str, *, embedder: Embedder, store: VectorStore, top_k: int
) -> list[RetrievedChunk]:
    return store.query(embedder.embed_query(question), doc_id=doc_id, top_k=top_k)
