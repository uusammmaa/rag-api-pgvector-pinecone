"""Application settings, loaded from environment variables and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BeforeValidator, Field, SecretStr
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

VectorStoreName = Literal["pinecone", "pgvector", "memory"]
ChunkStrategy = Literal["fixed", "recursive", "heading"]


def normalize_store_name(value: Any) -> Any:
    """Accept ``supabase`` as an alias: Supabase is managed Postgres with pgvector."""
    if isinstance(value, str):
        value = value.strip().lower()
        return "pgvector" if value == "supabase" else value
    return value


def _split_csv(value: Any) -> Any:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


StoreName = Annotated[VectorStoreName, BeforeValidator(normalize_store_name)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Service
    app_name: str = "RAG Document API"
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: Annotated[list[str], NoDecode, BeforeValidator(_split_csv)] = [
        "http://localhost:3000"
    ]

    # Provider credentials. All optional: a provider is only required when selected.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "OPEN_AI_API_KEY"),
    )
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "rag-documents"
    database_url: SecretStr | None = Field(
        default=None,
        description="Postgres DSN with the pgvector extension (local Postgres or Supabase).",
    )
    pgvector_auto_migrate: bool = True

    # Retrieval
    vector_store: StoreName = "pinecone"
    embedding_provider: Literal["openai", "hash"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = Field(default=1536, ge=8, le=4096)
    chunk_strategy: ChunkStrategy = "heading"
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)
    top_k: int = Field(default=5, ge=1, le=50)

    # Generation
    llm_provider: Literal["anthropic", "mock"] = "anthropic"
    chat_model: str = "claude-haiku-5-5"
    chat_effort: Literal["low", "medium", "high"] = "low"
    chat_max_tokens: int = Field(default=4096, ge=256)
    judge_model: str = "claude-opus-5-5"

    # Input limits
    max_upload_mb: float = Field(default=20, gt=0)
    max_question_chars: int = Field(default=2000, ge=10)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
