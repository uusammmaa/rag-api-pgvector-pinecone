"""Request and response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.config import ChunkStrategy, StoreName, VectorStoreName


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int = Field(description="Number of chunks stored.")
    pages: int
    vector_store: VectorStoreName
    chunk_strategy: ChunkStrategy


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, examples=["Who has the power to declare war?"])
    doc_id: str = Field(min_length=1, max_length=128, description="ID returned by /upload.")
    vector_store: StoreName | None = Field(
        default=None, description="Overrides VECTOR_STORE for this request."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question", "doc_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class Source(BaseModel):
    chunk_id: str
    text: str
    score: float
    page: int | None = None
    section: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    details: object | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
