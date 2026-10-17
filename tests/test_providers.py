"""Embedding and generation providers, with SDK clients replaced by fakes."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.errors import ProviderNotConfiguredError
from app.rag.embeddings import HashEmbedder, OpenAIEmbedder, tokenize
from app.rag.generation import AnthropicGenerator, ExtractiveGenerator, complete
from app.rag.prompts import NOT_FOUND_ANSWER, build_user_prompt
from app.services import Services
from app.vectorstores.base import RetrievedChunk
from app.vectorstores.memory import cosine

CHUNKS = [
    RetrievedChunk(
        "d-0", "The keeper lit the lamp at dusk. Ships relied on it.", 0.9, 0, 1, "Ch 1"
    ),
    RetrievedChunk("d-1", "Supplies came by boat once a month in winter.", 0.5, 1, 2, None),
]


def test_tokenize_drops_stopwords_and_punctuation() -> None:
    assert tokenize("The Senate shall, by law, be chosen!") == ["senate", "law", "chosen"]


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashEmbedder(dim=64)
    a = embedder.embed_query("powers of congress")
    assert a == HashEmbedder(dim=64).embed_query("powers of congress")
    assert sum(v * v for v in a) == pytest.approx(1.0)
    assert len(a) == 64


def test_hash_embedder_rewards_lexical_overlap() -> None:
    embedder = HashEmbedder(dim=512)
    query = embedder.embed_query("who declares war")
    near, far = embedder.embed_documents(
        ["Congress shall have power to declare war", "Rules for commas in a series"]
    )
    assert cosine(query, near) > cosine(query, far)


def test_openai_embedder_batches_requests() -> None:
    calls = []

    class FakeEmbeddings:
        def create(self, model: str, input: list[str], dimensions: int):
            calls.append((model, len(input), dimensions))
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 3) for _ in input])

    client = SimpleNamespace(embeddings=FakeEmbeddings())
    embedder = OpenAIEmbedder("key", model="m", dim=3, batch_size=2, client=client)
    assert len(embedder.embed_documents(["a", "b", "c"])) == 3
    assert calls == [("m", 2, 3), ("m", 1, 3)]


def test_prompt_numbers_excerpts_and_includes_metadata() -> None:
    prompt = build_user_prompt("When is the lamp lit?", CHUNKS)
    assert '<excerpt id="1" section="Ch 1" page="1">' in prompt
    assert '<excerpt id="2" page="2">' in prompt
    assert prompt.rstrip().endswith("</question>")


def test_extractive_generator_picks_overlapping_sentences() -> None:
    answer = complete(ExtractiveGenerator(max_sentences=1), "When is the lamp lit?", CHUNKS)
    assert answer == "The keeper lit the lamp at dusk."


def test_extractive_generator_says_not_found_without_overlap() -> None:
    assert complete(ExtractiveGenerator(), "quantum chromodynamics", CHUNKS) == NOT_FOUND_ANSWER


class FakeStream:
    def __init__(self, parts: list[str], stop_reason: str) -> None:
        self.text_stream = iter(parts)
        self._stop = stop_reason

    def get_final_message(self) -> SimpleNamespace:
        return SimpleNamespace(stop_reason=self._stop)


def fake_anthropic(parts: list[str], stop_reason: str = "end_turn", sink: dict | None = None):
    @contextmanager
    def stream(**kwargs):
        if sink is not None:
            sink.update(kwargs)
        yield FakeStream(parts, stop_reason)

    return SimpleNamespace(messages=SimpleNamespace(stream=stream))


def test_anthropic_generator_streams_text_with_grounding_prompt() -> None:
    sent: dict = {}
    generator = AnthropicGenerator(
        "key", model="claude-haiku-5-5", client=fake_anthropic(["The ", "lamp."], sink=sent)
    )
    assert list(generator.stream("q?", CHUNKS)) == ["The ", "lamp."]
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["output_config"] == {"effort": "low"}
    assert "only the excerpts" in sent["system"]
    assert "<question>\nq?\n</question>" in sent["messages"][0]["content"]


def test_anthropic_generator_reports_refusals() -> None:
    generator = AnthropicGenerator("key", model="m", client=fake_anthropic([], "refusal"))
    assert "declined" in complete(generator, "q", CHUNKS)


def test_services_require_keys_only_for_selected_providers() -> None:
    settings = Settings(
        _env_file=None,
        embedding_provider="openai",
        llm_provider="anthropic",
        openai_api_key=None,
        anthropic_api_key=None,
        pinecone_api_key=None,
        database_url=None,
    )
    services = Services(settings)
    for attr in ("embedder", "generator"):
        with pytest.raises(ProviderNotConfiguredError):
            getattr(services, attr)
    for store in ("pinecone", "pgvector"):
        with pytest.raises(ProviderNotConfiguredError):
            services.store(store)
    assert services.store("memory") is services.store("memory")


def test_supabase_is_an_alias_for_pgvector() -> None:
    assert Settings(_env_file=None, vector_store="supabase").vector_store == "pgvector"


def test_cors_origins_accept_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    assert Settings(_env_file=None).cors_origins == ["http://a.test", "http://b.test"]
