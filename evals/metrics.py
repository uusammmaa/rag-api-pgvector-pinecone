"""Retrieval metrics scored against character-offset evidence spans.

Evidence is labelled as verbatim text, not chunk IDs, so the same labels work for
every chunking strategy. Definitions, for one question with evidence spans E and
the top-k retrieved chunks C:

* a chunk is **relevant** if it overlaps some evidence span by at least half of
  the shorter of the two (the chunk mostly sits inside the evidence, or the
  evidence mostly sits inside the chunk);
* an evidence span is **found** if the top-k chunks together cover at least half
  of its characters;
* precision@k = relevant chunks in top k / k;
* recall@k = found evidence spans / |E|;
* reciprocal rank = 1 / rank of the first relevant chunk (0 if none in top k).

Precision@k is capped below 1 whenever a question has fewer relevant chunks than
k, so compare it across strategies rather than reading it in absolute terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Span = tuple[int, int]

THRESHOLD = 0.5


def overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def is_relevant(chunk: Span, evidence: Sequence[Span], threshold: float = THRESHOLD) -> bool:
    for span in evidence:
        shorter = min(chunk[1] - chunk[0], span[1] - span[0])
        if shorter > 0 and overlap(chunk, span) / shorter >= threshold:
            return True
    return False


def coverage(span: Span, chunks: Sequence[Span]) -> float:
    """Fraction of ``span`` covered by the union of ``chunks``."""
    covered = [False] * (span[1] - span[0])
    for chunk in chunks:
        for i in range(max(chunk[0], span[0]), min(chunk[1], span[1])):
            covered[i - span[0]] = True
    return sum(covered) / len(covered) if covered else 0.0


@dataclass(frozen=True)
class RetrievalScores:
    precision: float
    recall: float
    reciprocal_rank: float


def score_retrieval(
    retrieved: Sequence[Span], evidence: Sequence[Span], k: int, threshold: float = THRESHOLD
) -> RetrievalScores:
    top = list(retrieved[:k])
    relevant = [is_relevant(chunk, evidence, threshold) for chunk in top]
    found = sum(coverage(span, top) >= threshold for span in evidence)
    first = next((rank for rank, hit in enumerate(relevant, start=1) if hit), None)
    return RetrievalScores(
        precision=sum(relevant) / k,
        recall=found / len(evidence) if evidence else 0.0,
        reciprocal_rank=1.0 / first if first else 0.0,
    )


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
