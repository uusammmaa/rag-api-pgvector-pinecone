"""Evaluate retrieval and answer quality across chunking strategies and k values.

    python -m evals.run                 # mock mode: no API keys, deterministic
    python -m evals.run --mode live     # OpenAI embeddings, Claude answers, Claude judge

Writes ``evals/RESULTS.md`` (summary tables) and ``evals/results.json`` (per question).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

from app.config import Settings
from app.rag.chunking import Chunk, build_chunker
from app.rag.embeddings import Embedder, HashEmbedder, OpenAIEmbedder, Vector
from app.rag.generation import AnthropicGenerator, ExtractiveGenerator, Generator, complete
from app.rag.pipeline import chunk_document
from app.vectorstores.base import RetrievedChunk
from app.vectorstores.memory import InMemoryVectorStore
from evals.dataset import EVALS_DIR, Example, load_dataset, load_document
from evals.judge import ClaudeJudge, Judge, LexicalJudge
from evals.metrics import mean, score_retrieval

logger = logging.getLogger("evals")

STRATEGIES = ("fixed", "recursive", "heading")
K_VALUES = (3, 5)
ERROR_PREFIX = "error: "
DOC_TITLES = {
    "us_constitution": "US Constitution",
    "declaration_of_independence": "Declaration of Independence",
    "elements_of_style": "The Elements of Style",
}


class CachedEmbedder:
    """Disk cache keyed by model and text so repeated live runs only pay once."""

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.model = inner.model
        self.path = cache_dir / f"embeddings-{inner.model}-{inner.dim}.json"
        self.cache: dict[str, Vector] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        missing = list(dict.fromkeys(t for t in texts if self._key(t) not in self.cache))
        if missing:
            for text, vector in zip(missing, self.inner.embed_documents(missing), strict=True):
                self.cache[self._key(text)] = vector
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache))
        return [self.cache[self._key(t)] for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]


@dataclass
class QuestionResult:
    id: str
    doc: str
    strategy: str
    k: int
    precision: float
    recall: float
    reciprocal_rank: float
    retrieved: list[int]
    answer: str = ""
    faithfulness: float | None = None
    relevance: float | None = None
    judge_reasoning: str = ""


@dataclass
class ConfigSummary:
    strategy: str
    k: int
    chunks: int
    avg_chars: float
    precision: float
    recall: float
    mrr: float
    faithfulness: float | None
    relevance: float | None
    per_doc_recall: dict[str, float] = field(default_factory=dict)


@dataclass
class Components:
    mode: str
    embedder: Embedder
    generator: Generator
    judge: Judge


def build_components(mode: str, settings: Settings) -> Components:
    if mode == "mock":
        return Components(mode, HashEmbedder(dim=1024), ExtractiveGenerator(), LexicalJudge())
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", settings.openai_api_key),
            ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
        )
        if value is None
    ]
    if missing:
        sys.exit(f"live mode needs {', '.join(missing)} (or run with --mode mock)")
    embedder = OpenAIEmbedder(
        settings.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
        model=settings.embedding_model,
        dim=settings.embedding_dim,
    )
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value(), max_retries=4)  # type: ignore[union-attr]
    generator = AnthropicGenerator(
        "", model=settings.chat_model, effort=settings.chat_effort, client=client
    )
    return Components(
        mode,
        CachedEmbedder(embedder, EVALS_DIR / ".cache"),
        generator,
        ClaudeJudge(settings.judge_model, client=client),
    )


def answer_and_grade(
    components: Components,
    example: Example,
    context: list[RetrievedChunk],
    result: QuestionResult,
) -> None:
    """Answer and grade one question. A provider failure is recorded on the result
    (scores stay None and are excluded from means) instead of aborting the run."""
    try:
        result.answer = complete(components.generator, example.question, context)
        grade = components.judge.grade(
            example.question, example.expected_answer, result.answer, context
        )
    except Exception as exc:
        logger.warning("%s (%s, k=%d) failed: %s", example.id, result.strategy, result.k, exc)
        result.judge_reasoning = f"{ERROR_PREFIX}{type(exc).__name__}: {exc}"
        return
    result.faithfulness, result.relevance = grade.faithfulness, grade.relevance
    result.judge_reasoning = grade.reasoning


def run(
    components: Components,
    examples: list[Example],
    strategies: Sequence[str],
    k_values: Sequence[int],
    chunk_size: int,
    overlap: int,
    concurrency: int,
) -> tuple[list[ConfigSummary], list[QuestionResult]]:
    docs = sorted({e.doc for e in examples})
    max_k = max(k_values)
    question_vectors = components.embedder.embed_documents([e.question for e in examples])
    summaries: list[ConfigSummary] = []
    results: list[QuestionResult] = []

    for strategy in strategies:
        chunker = build_chunker(strategy, chunk_size, overlap)  # type: ignore[arg-type]
        store = InMemoryVectorStore()
        chunks_by_doc: dict[str, list[Chunk]] = {}
        for doc in docs:
            chunks = chunk_document(load_document(doc), chunker)
            vectors = components.embedder.embed_documents([c.embed_text for c in chunks])
            store.upsert(doc, chunks, vectors)
            chunks_by_doc[doc] = chunks
        all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]

        retrieved = [
            store.query(vector, doc_id=example.doc, top_k=max_k)
            for example, vector in zip(examples, question_vectors, strict=True)
        ]

        for k in k_values:
            config_results = []
            for example, hits in zip(examples, retrieved, strict=True):
                spans = [
                    (
                        chunks_by_doc[example.doc][h.chunk_index].start,
                        chunks_by_doc[example.doc][h.chunk_index].end,
                    )
                    for h in hits
                ]
                scores = score_retrieval(spans, example.evidence_spans, k)
                config_results.append(
                    QuestionResult(
                        id=example.id,
                        doc=example.doc,
                        strategy=strategy,
                        k=k,
                        precision=scores.precision,
                        recall=scores.recall,
                        reciprocal_rank=scores.reciprocal_rank,
                        retrieved=[h.chunk_index for h in hits[:k]],
                    )
                )

            started = time.perf_counter()
            contexts = [hits[:k] for hits in retrieved]
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                list(
                    pool.map(
                        partial(answer_and_grade, components),
                        examples,
                        contexts,
                        config_results,
                    )
                )
            logger.info(
                "%s k=%d: answered and graded %d questions in %.1fs",
                strategy,
                k,
                len(examples),
                time.perf_counter() - started,
            )

            faith = [r.faithfulness for r in config_results if r.faithfulness is not None]
            rel = [r.relevance for r in config_results if r.relevance is not None]
            summaries.append(
                ConfigSummary(
                    strategy=strategy,
                    k=k,
                    chunks=len(all_chunks),
                    avg_chars=mean([len(c.text) for c in all_chunks]),
                    precision=mean([r.precision for r in config_results]),
                    recall=mean([r.recall for r in config_results]),
                    mrr=mean([r.reciprocal_rank for r in config_results]),
                    faithfulness=mean(faith) if faith else None,
                    relevance=mean(rel) if rel else None,
                    per_doc_recall={
                        doc: mean([r.recall for r in config_results if r.doc == doc])
                        for doc in docs
                    },
                )
            )
            results.extend(config_results)
    return summaries, results


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(
    components: Components,
    summaries: list[ConfigSummary],
    results: list[QuestionResult],
    examples: list[Example],
    chunk_size: int,
    overlap: int,
    settings: Settings,
) -> str:
    evidence_count = sum(len(e.evidence) for e in examples)
    docs = sorted({e.doc for e in examples})
    if components.mode == "mock":
        mode_line = (
            "**Mode: mock.** Hash embeddings (lexical, 1024-d), an extractive generator and a "
            "lexical judge. Runs offline and in CI. Faithfulness and relevance here are "
            "token-overlap proxies on the 1-5 scale, not model judgments; the extractive "
            "generator only quotes retrieved text, so its faithfulness is near 5 by construction. "
            "Run `make eval-live` for real numbers."
        )
    else:
        mode_line = (
            f"**Mode: live.** Embeddings `{settings.embedding_model}` "
            f"({settings.embedding_dim}-d), "
            f"answers `{settings.chat_model}` (effort `{settings.chat_effort}`), "
            f"judge `{components.judge.name}` with the fixed rubric in `evals/judge.py`."
        )
    best = max(summaries, key=lambda s: (s.recall, s.mrr))
    lines = [
        "# Eval results",
        "",
        "Generated by `python -m evals.run"
        + ("" if components.mode == "mock" else " --mode live")
        + "`. Do not edit by hand.",
        "",
        mode_line,
        "",
        f"Corpus: {len(docs)} public-domain PDFs, {len(examples)} questions, "
        f"{evidence_count} labelled evidence spans. Chunk size {chunk_size} characters, "
        f"overlap {overlap}. Retrieval is scoped to the question's document, as in the API.",
        "",
        "## Summary",
        "",
        "| Strategy | k | Chunks | Avg chars | Precision@k | Recall@k | MRR | Faithfulness (1-5) "
        "| Relevance (1-5) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.strategy} | {s.k} | {s.chunks} | {s.avg_chars:.0f} | {_fmt(s.precision)} | "
            f"{_fmt(s.recall)} | {_fmt(s.mrr)} | {_fmt(s.faithfulness, 2)} | "
            f"{_fmt(s.relevance, 2)} |"
        )
    lines += [
        "",
        f"Best retrieval: **{best.strategy}, k={best.k}** (recall@k {best.recall:.3f}, "
        f"MRR {best.mrr:.3f}).",
        "",
    ]
    failed = sum(r.judge_reasoning.startswith(ERROR_PREFIX) for r in results)
    if failed:
        lines += [
            f"**{failed} of {len(results)} answers failed** (provider errors) and are excluded "
            "from the faithfulness and relevance means; see `evals/results.json`.",
            "",
        ]
    lines += [
        "## Recall@k by document",
        "",
        "| Strategy | k | " + " | ".join(DOC_TITLES.get(d, d) for d in docs) + " |",
        "|---|---:|" + "---:|" * len(docs),
    ]
    for s in summaries:
        lines.append(
            f"| {s.strategy} | {s.k} | "
            + " | ".join(_fmt(s.per_doc_recall[d]) for d in docs)
            + " |"
        )

    max_k = max(s.k for s in summaries)
    by_question: dict[str, list[float]] = {}
    for r in results:
        if r.k == max_k:
            by_question.setdefault(r.id, []).append(r.recall)
    hardest = sorted(by_question.items(), key=lambda kv: (mean(kv[1]), kv[0]))[:5]
    questions = {e.id: e.question for e in examples}
    lines += [
        "",
        f"## Hardest questions (mean recall@{max_k} across strategies)",
        "",
        "| ID | Question | Recall |",
        "|---|---|---:|",
    ]
    lines += [f"| {qid} | {questions[qid]} | {mean(vals):.2f} |" for qid, vals in hardest]
    lines += [
        "",
        "## Method",
        "",
        "- Evidence is labelled as verbatim passages and anchored to character offsets in the "
        "extracted PDF text, so one set of labels scores every chunking strategy.",
        "- A retrieved chunk is relevant when it overlaps an evidence span by at least half of "
        "the shorter of the two. An evidence span is found when the top-k chunks cover at "
        "least half of it.",
        "- Precision@k = relevant chunks / k (capped below 1 when a question has fewer "
        "relevant chunks than k). Recall@k = found spans / labelled spans. MRR uses the first "
        "relevant chunk within the top k.",
        "- Faithfulness: are the answer's claims supported by the retrieved context? Relevance: "
        "does it answer the question and agree with the reference answer? Both 1-5, rubric in "
        "`evals/judge.py`.",
        "- Per-question rows, including answers and judge reasoning, are in `evals/results.json`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--k", nargs="+", type=int, default=list(K_VALUES), dest="k_values")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke runs)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, default=EVALS_DIR / "RESULTS.md")
    parser.add_argument("--json", type=Path, default=EVALS_DIR / "results.json")
    parser.add_argument(
        "--fail-under-recall",
        type=float,
        help="exit non-zero if the best configuration's recall@k is below this value",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = Settings()
    components = build_components(args.mode, settings)
    examples = load_dataset(limit=args.limit)
    logger.info("loaded %d questions; mode=%s", len(examples), args.mode)

    summaries, results = run(
        components,
        examples,
        args.strategies,
        sorted(args.k_values),
        args.chunk_size,
        args.overlap,
        args.concurrency if args.mode == "live" else 1,
    )
    report = render_report(
        components, summaries, results, examples, args.chunk_size, args.overlap, settings
    )
    args.output.write_text(report, encoding="utf-8")
    args.json.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "summaries": [asdict(s) for s in summaries],
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report)

    best = max(s.recall for s in summaries)
    if args.fail_under_recall is not None and best < args.fail_under_recall:
        print(f"best recall@k {best:.3f} is below --fail-under-recall {args.fail_under_recall}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
