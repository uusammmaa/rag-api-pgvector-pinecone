"""Embedding providers.

``OpenAIEmbedder`` is the production path. ``HashEmbedder`` is a deterministic,
dependency-free stand-in (signed feature hashing of unigrams and bigrams) used
for tests, mock evals and running the stack without API keys. It is lexical,
so it behaves like a weak BM25 rather than a semantic model.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

from app.errors import UpstreamError

logger = logging.getLogger(__name__)

Vector = list[float]


class Embedder(Protocol):
    dim: int
    model: str

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        batch_size: int = 128,
        client: object | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, max_retries=3)
        self.client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size

    def _embed(self, texts: Sequence[str]) -> list[Vector]:
        import openai

        vectors: list[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            try:
                response = self.client.embeddings.create(  # type: ignore[attr-defined]
                    model=self.model, input=batch, dimensions=self.dim
                )
            except openai.OpenAIError as exc:
                logger.error("embedding request failed: %s", exc)
                raise UpstreamError("Embedding provider request failed.") from exc
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(texts)

    def embed_query(self, text: str) -> Vector:
        return self._embed([text])[0]


_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOPWORDS = """a an and are as at be been but by can could did do does for from had has have he
    her his how i if in into is it its may might must no not of on or our shall should
    so such than that the their them then there these they this those to was we were
    what when where which who whom why will with would you your"""
STOPWORDS = frozenset(_STOPWORDS.split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class HashEmbedder:
    model = "hash-bow-v1"

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _vector(self, text: str) -> Vector:
        tokens = tokenize(text)
        features = Counter(tokens)
        features.update(f"{a}_{b}" for a, b in pairwise(tokens))
        vec = [0.0] * self.dim
        for feature, count in features.items():
            idx, sign = self._bucket(feature)
            vec[idx] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)
