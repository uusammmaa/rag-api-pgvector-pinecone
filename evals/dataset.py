"""Load the labelled dataset and anchor its evidence spans in the extracted PDF text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from app.rag.pdf import ExtractedDocument, extract_pdf

EVALS_DIR = Path(__file__).parent
CORPUS_DIR = EVALS_DIR / "corpus"
DATASET_PATH = EVALS_DIR / "dataset.jsonl"

Span = tuple[int, int]


@dataclass(frozen=True)
class Example:
    id: str
    doc: str
    question: str
    expected_answer: str
    evidence: tuple[str, ...]
    evidence_spans: tuple[Span, ...]


@cache
def load_document(name: str) -> ExtractedDocument:
    """Extract a corpus PDF with the same code path the API uses."""
    return extract_pdf((CORPUS_DIR / f"{name}.pdf").read_bytes())


def locate(evidence: str, text: str) -> Span:
    """Find ``evidence`` in ``text`` ignoring whitespace differences (PDF extraction
    re-wraps lines). Raises if the span is missing."""
    pattern = r"\s+".join(re.escape(word) for word in evidence.split())
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"evidence not found in document: {evidence[:80]!r}")
    return match.span()


def load_dataset(path: Path = DATASET_PATH, limit: int | None = None) -> list[Example]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = load_document(row["doc"]).text
        examples.append(
            Example(
                id=row["id"],
                doc=row["doc"],
                question=row["question"],
                expected_answer=row["expected_answer"],
                evidence=tuple(row["evidence"]),
                evidence_spans=tuple(locate(e, text) for e in row["evidence"]),
            )
        )
    return examples[:limit] if limit else examples
