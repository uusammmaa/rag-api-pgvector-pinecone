"""Answer generators: Claude for real answers, an extractive mock for offline runs."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from typing import Protocol

from app.errors import UpstreamError
from app.rag.embeddings import tokenize
from app.rag.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT, build_user_prompt
from app.vectorstores.base import RetrievedChunk

logger = logging.getLogger(__name__)


class Generator(Protocol):
    model: str

    def stream(self, question: str, chunks: Sequence[RetrievedChunk]) -> Iterator[str]: ...


def complete(generator: Generator, question: str, chunks: Sequence[RetrievedChunk]) -> str:
    return "".join(generator.stream(question, chunks))


class AnthropicGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        effort: str = "low",
        client: object | None = None,
    ) -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def stream(self, question: str, chunks: Sequence[RetrievedChunk]) -> Iterator[str]:
        import anthropic

        try:
            with self.client.messages.stream(  # type: ignore[attr-defined]
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(question, chunks)}],
                output_config={"effort": self.effort},
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.APIError as exc:
            logger.error("generation request failed: %s", exc)
            raise UpstreamError("Language model request failed.") from exc
        if final.stop_reason == "refusal":
            logger.warning("model declined to answer")
            yield "\n\nThe model declined to answer this question."
        elif final.stop_reason == "max_tokens":
            logger.warning("answer truncated at max_tokens=%s", self.max_tokens)


_SENTENCE = re.compile(r"(?<=[.!?;:])\s+|\n+")


class ExtractiveGenerator:
    """Deterministic offline generator: returns the retrieved sentences that best
    overlap the question. Faithful by construction, so mock evals measure whether
    retrieval surfaced the right text rather than model quality."""

    model = "mock-extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def stream(self, question: str, chunks: Sequence[RetrievedChunk]) -> Iterator[str]:
        q_terms = set(tokenize(question))
        candidates: list[tuple[float, int, str]] = []
        for rank, chunk in enumerate(chunks):
            for sentence in _SENTENCE.split(chunk.text):
                sentence = " ".join(sentence.split())
                terms = set(tokenize(sentence))
                if len(terms) < 3:
                    continue
                overlap = len(q_terms & terms)
                if overlap:
                    # Prefer overlap, then higher-ranked chunks.
                    candidates.append((overlap - 0.01 * rank, len(candidates), sentence))
        if not candidates:
            yield NOT_FOUND_ANSWER
            return
        best = sorted(candidates, key=lambda c: (-c[0], c[1]))[: self.max_sentences]
        ordered = [s for _, _, s in sorted(best, key=lambda c: c[1])]
        yield " ".join(ordered)
