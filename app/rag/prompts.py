"""Prompt templates for grounded answering."""

from __future__ import annotations

from collections.abc import Sequence

from app.vectorstores.base import RetrievedChunk

SYSTEM_PROMPT = """You answer questions about a document using only the excerpts provided.

Rules:
- Base every statement on the excerpts. Do not use outside knowledge.
- If the excerpts do not contain the answer, say that the document does not say, \
and stop.
- Cite the excerpts you used with their bracketed numbers, for example [1] or [2][3].
- Be concise: answer in a few sentences unless the question asks for a list."""

NOT_FOUND_ANSWER = "The document does not say."


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        attrs = f' section="{chunk.section}"' if chunk.section else ""
        attrs += f' page="{chunk.page}"' if chunk.page else ""
        parts.append(f'<excerpt id="{i}"{attrs}>\n{chunk.text}\n</excerpt>')
    return "\n".join(parts)


def build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    return (
        f"<excerpts>\n{format_context(chunks)}\n</excerpts>\n\n<question>\n{question}\n</question>"
    )
