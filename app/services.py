"""Builds providers from settings and caches them for the life of the process."""

from __future__ import annotations

import threading
from functools import cached_property

from pydantic import SecretStr

from app.config import Settings, VectorStoreName
from app.errors import ProviderNotConfiguredError
from app.rag.chunking import Chunker, build_chunker
from app.rag.embeddings import Embedder, HashEmbedder, OpenAIEmbedder
from app.rag.generation import AnthropicGenerator, ExtractiveGenerator, Generator
from app.vectorstores.base import VectorStore
from app.vectorstores.memory import InMemoryVectorStore


def _require(value: SecretStr | None, name: str, purpose: str) -> str:
    if value is None or not value.get_secret_value().strip():
        raise ProviderNotConfiguredError(f"{purpose} is not configured: set {name}.")
    return value.get_secret_value()


class Services:
    """Lazily constructed providers. Missing credentials surface as a 503 on the
    request that needs them instead of crashing the process at import time."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._stores: dict[str, VectorStore] = {}
        self._lock = threading.Lock()

    @cached_property
    def chunker(self) -> Chunker:
        s = self.settings
        return build_chunker(s.chunk_strategy, s.chunk_size, s.chunk_overlap)

    @cached_property
    def embedder(self) -> Embedder:
        s = self.settings
        if s.embedding_provider == "hash":
            return HashEmbedder(dim=s.embedding_dim)
        key = _require(s.openai_api_key, "OPENAI_API_KEY", "OpenAI embeddings")
        return OpenAIEmbedder(key, model=s.embedding_model, dim=s.embedding_dim)

    @cached_property
    def generator(self) -> Generator:
        s = self.settings
        if s.llm_provider == "mock":
            return ExtractiveGenerator()
        key = _require(s.anthropic_api_key, "ANTHROPIC_API_KEY", "Claude")
        return AnthropicGenerator(
            key, model=s.chat_model, max_tokens=s.chat_max_tokens, effort=s.chat_effort
        )

    def store(self, name: VectorStoreName | None = None) -> VectorStore:
        name = name or self.settings.vector_store
        with self._lock:
            if name not in self._stores:
                self._stores[name] = self._build_store(name)
            return self._stores[name]

    def _build_store(self, name: VectorStoreName) -> VectorStore:
        s = self.settings
        if name == "memory":
            return InMemoryVectorStore()
        if name == "pinecone":
            from app.vectorstores.pinecone import PineconeVectorStore

            key = _require(s.pinecone_api_key, "PINECONE_API_KEY", "Pinecone")
            return PineconeVectorStore.from_settings(key, s.pinecone_index_name)
        if name == "pgvector":
            from app.vectorstores.pgvector import PgVectorStore

            dsn = _require(s.database_url, "DATABASE_URL", "pgvector")
            return PgVectorStore.from_dsn(dsn, s.embedding_dim, s.pgvector_auto_migrate)
        raise ProviderNotConfiguredError(f"Unknown vector store: {name}")
