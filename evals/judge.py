"""Answer quality judges.

``ClaudeJudge`` grades with a fixed rubric and structured output. ``LexicalJudge``
is the offline stand-in used in mock mode and CI: it maps token overlap onto the
same 1-5 scale so the pipeline and report run end to end without API keys. Its
numbers are proxies, not model judgments.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from app.rag.embeddings import tokenize
from app.rag.prompts import NOT_FOUND_ANSWER, format_context
from app.vectorstores.base import RetrievedChunk

logger = logging.getLogger(__name__)

RUBRIC = """You are grading one answer produced by a retrieval-augmented QA system.
Score it on two independent criteria, each from 1 to 5.

FAITHFULNESS: is every claim in the ANSWER supported by the CONTEXT the system \
retrieved? Judge only against the context, not your own knowledge.
5 = every claim is supported by the context
4 = supported, apart from a minor detail that does not change the meaning
3 = a mix of supported and unsupported claims
2 = mostly unsupported by the context
1 = contradicts the context or is fabricated
An answer that says the context does not contain the information, when it indeed \
does not, scores 5.

RELEVANCE: does the ANSWER respond to the QUESTION and agree with the REFERENCE \
answer written by a human labeller?
5 = fully answers the question and matches the reference
4 = answers it, with a minor omission or imprecision
3 = partially answers it or misses an important part of the reference
2 = touches on the topic but does not answer the question
1 = does not answer, refuses, or contradicts the reference

Grade each criterion separately: a faithful answer can be irrelevant and a \
relevant answer can be unfaithful. Write brief reasoning first, then the scores."""


class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="One to three sentences justifying the scores.")
    faithfulness: Literal[1, 2, 3, 4, 5]
    relevance: Literal[1, 2, 3, 4, 5]


@dataclass(frozen=True)
class Grade:
    faithfulness: float | None
    relevance: float | None
    reasoning: str = ""


class Judge(Protocol):
    name: str

    def grade(
        self, question: str, reference: str, answer: str, context: Sequence[RetrievedChunk]
    ) -> Grade: ...


class ClaudeJudge:
    def __init__(self, model: str, client: object | None = None, effort: str = "medium") -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(max_retries=4)
        self.client = client
        self.model = model
        self.effort = effort
        self.name = model

    def grade(
        self, question: str, reference: str, answer: str, context: Sequence[RetrievedChunk]
    ) -> Grade:
        prompt = (
            f"<question>\n{question}\n</question>\n\n"
            f"<reference>\n{reference}\n</reference>\n\n"
            f"<context>\n{format_context(context)}\n</context>\n\n"
            f"<answer>\n{answer}\n</answer>"
        )
        response = self.client.messages.parse(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=4096,
            system=RUBRIC,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
            output_format=JudgeVerdict,
        )
        verdict = response.parsed_output
        if response.stop_reason == "refusal" or verdict is None:
            logger.warning("judge returned no verdict (stop_reason=%s)", response.stop_reason)
            return Grade(None, None, f"no verdict: {response.stop_reason}")
        return Grade(float(verdict.faithfulness), float(verdict.relevance), verdict.reasoning)


def _scale(fraction: float) -> float:
    return round(1 + 4 * max(0.0, min(1.0, fraction)), 2)


class LexicalJudge:
    """Offline proxy: faithfulness = share of answer terms found in the context;
    relevance = share of reference-answer terms found in the answer."""

    name = "lexical-proxy"

    def grade(
        self, question: str, reference: str, answer: str, context: Sequence[RetrievedChunk]
    ) -> Grade:
        answer_terms = set(tokenize(answer))
        if answer.strip() == NOT_FOUND_ANSWER or not answer_terms:
            return Grade(5.0, 1.0, "abstained")
        context_terms = set(tokenize(" ".join(c.text for c in context)))
        reference_terms = set(tokenize(reference))
        support = len(answer_terms & context_terms) / len(answer_terms)
        recall = len(reference_terms & answer_terms) / len(reference_terms or {""})
        return Grade(_scale(support), _scale(recall))
