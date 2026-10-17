"""HTTP routes."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app import __version__
from app.config import StoreName
from app.errors import (
    AppError,
    BadRequestError,
    DocumentNotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.logging_config import request_id_var
from app.rag.generation import complete
from app.rag.pdf import PDF_MAGIC, extract_pdf
from app.rag.pipeline import ingest, retrieve
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryResponse,
    QuestionRequest,
    Source,
    UploadResponse,
)
from app.services import Services
from app.vectorstores.base import RetrievedChunk

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}
READ_CHUNK = 1024 * 1024
MULTIPART_OVERHEAD = 64 * 1024

ERRORS = {code: {"model": ErrorResponse} for code in (400, 404, 413, 415, 422, 502, 503)}


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(version=__version__)


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    data = bytearray()
    while chunk := await file.read(READ_CHUNK):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise PayloadTooLargeError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    return bytes(data)


@router.post("/upload", response_model=UploadResponse, responses=ERRORS, tags=["documents"])
async def upload_document(
    request: Request,
    services: ServicesDep,
    file: Annotated[UploadFile, File(description="Text-based PDF")],
    vector_store: Annotated[StoreName | None, Form()] = None,
) -> UploadResponse:
    """Upload a PDF: extract text, chunk, embed and store it. Returns a ``doc_id``."""
    settings = services.settings
    max_bytes = settings.max_upload_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + MULTIPART_OVERHEAD:
        raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_mb:g} MB limit.")

    filename = PurePath(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise UnsupportedMediaTypeError("Only .pdf files are supported.")
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise UnsupportedMediaTypeError(f"Unsupported content type: {content_type}.")

    data = await _read_limited(file, max_bytes)
    if not data:
        raise BadRequestError("The uploaded file is empty.")
    if not data.startswith(PDF_MAGIC):
        raise UnsupportedMediaTypeError("The file is not a PDF (missing %PDF header).")

    store = await run_in_threadpool(services.store, vector_store)
    document = await run_in_threadpool(extract_pdf, data)
    doc_id = str(uuid.uuid4())
    result = await run_in_threadpool(
        lambda: ingest(
            document,
            doc_id,
            chunker=services.chunker,
            embedder=services.embedder,
            store=store,
            extra_metadata={"filename": filename},
        )
    )
    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunks=len(result.chunks),
        pages=result.pages,
        vector_store=store.name,  # type: ignore[arg-type]
        chunk_strategy=services.settings.chunk_strategy,
    )


def _retrieve(services: Services, body: QuestionRequest) -> list[RetrievedChunk]:
    if len(body.question) > services.settings.max_question_chars:
        raise BadRequestError(
            f"Question exceeds {services.settings.max_question_chars} characters."
        )
    store = services.store(body.vector_store)
    chunks = retrieve(
        body.question,
        body.doc_id,
        embedder=services.embedder,
        store=store,
        top_k=body.top_k or services.settings.top_k,
    )
    if not chunks:
        raise DocumentNotFoundError(
            f"No chunks found for doc_id {body.doc_id!r} in the {store.name} store."
        )
    return chunks


@router.post(
    "/chat",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/plain": {}}}, **ERRORS},
    tags=["chat"],
)
def chat(body: QuestionRequest, services: ServicesDep) -> StreamingResponse:
    """Stream a grounded answer as plain text.

    Retrieval runs before the response starts, so bad input, unknown documents and
    provider outages return proper status codes. A failure after streaming has begun
    ends the body with an ``[error]`` line carrying the request ID.
    """
    chunks = _retrieve(services, body)
    generator = services.generator
    request_id = request_id_var.get()

    def body_stream() -> Iterator[str]:
        try:
            yield from generator.stream(body.question, chunks)
        except AppError as exc:
            logger.error("stream aborted: %s", exc.message)
            yield f"\n\n[error] {exc.message} (request_id={request_id})"

    return StreamingResponse(
        body_stream(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Retrieved-Chunks": str(len(chunks))},
    )


@router.post("/query", response_model=QueryResponse, responses=ERRORS, tags=["chat"])
def query(body: QuestionRequest, services: ServicesDep) -> QueryResponse:
    """Non-streaming answer with the source chunks used to produce it."""
    chunks = _retrieve(services, body)
    answer = complete(services.generator, body.question, chunks)
    return QueryResponse(
        answer=answer,
        model=services.generator.model,
        sources=[
            Source(chunk_id=c.id, text=c.text, score=c.score, page=c.page, section=c.section)
            for c in chunks
        ],
    )
