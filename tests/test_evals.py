"""Eval harness: metric definitions, dataset integrity and an end-to-end mock run."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.rag.embeddings import HashEmbedder
from evals.dataset import CORPUS_DIR, load_dataset, load_document, locate
from evals.judge import ClaudeJudge, LexicalJudge
from evals.metrics import coverage, is_relevant, score_retrieval
from evals.run import CachedEmbedder, build_components, main, run

# --- metrics -------------------------------------------------------------------------


def test_relevance_uses_the_shorter_span() -> None:
    evidence = [(100, 200)]
    assert is_relevant((90, 210), evidence)  # chunk contains the evidence
    assert is_relevant((120, 180), evidence)  # chunk sits inside the evidence
    assert is_relevant((150, 400), evidence)  # half the evidence
    assert not is_relevant((160, 400), evidence)  # 40% of the evidence
    assert not is_relevant((300, 400), evidence)


def test_coverage_unions_chunks() -> None:
    assert coverage((0, 100), [(0, 30), (20, 60)]) == pytest.approx(0.6)
    assert coverage((0, 100), []) == 0.0


def test_score_retrieval() -> None:
    evidence = [(100, 200), (500, 600)]
    retrieved = [(900, 1000), (90, 210), (480, 620), (0, 50)]
    at2 = score_retrieval(retrieved, evidence, k=2)
    assert at2.precision == pytest.approx(0.5)
    assert at2.recall == pytest.approx(0.5)
    assert at2.reciprocal_rank == pytest.approx(0.5)
    at4 = score_retrieval(retrieved, evidence, k=4)
    assert at4.recall == pytest.approx(1.0)
    assert at4.precision == pytest.approx(0.5)


def test_score_retrieval_with_no_hits() -> None:
    scores = score_retrieval([(0, 10)], [(100, 200)], k=3)
    assert (scores.precision, scores.recall, scores.reciprocal_rank) == (0.0, 0.0, 0.0)


# --- dataset -------------------------------------------------------------------------


def test_dataset_is_well_formed() -> None:
    examples = load_dataset()
    assert 25 <= len(examples) <= 40
    assert len({e.id for e in examples}) == len(examples)
    assert len({e.doc for e in examples}) == 3
    for example in examples:
        assert example.question.strip() and example.expected_answer.strip()
        assert example.evidence
        text = load_document(example.doc).text
        for evidence, (start, end) in zip(example.evidence, example.evidence_spans, strict=True):
            assert " ".join(text[start:end].split()) == " ".join(evidence.split())


def test_corpus_sources_match_pdfs() -> None:
    for pdf in CORPUS_DIR.glob("*.pdf"):
        source = (CORPUS_DIR / "source" / f"{pdf.stem}.txt").read_text()
        extracted = load_document(pdf.stem).text
        assert " ".join(source.split()) == " ".join(extracted.split())


def test_locate_ignores_whitespace_and_rejects_missing() -> None:
    assert locate("two   Senators\nfrom", "the two Senators from each") == (4, 21)
    with pytest.raises(ValueError):
        locate("not present", "some text")


# --- judges and caching ----------------------------------------------------------------


def test_lexical_judge() -> None:
    from app.vectorstores.base import RetrievedChunk

    context = [RetrievedChunk("d-0", "Congress shall have power to declare war.", 1.0, 0)]
    good = LexicalJudge().grade("Who declares war?", "Congress declares war.", "Congress.", context)
    assert good.faithfulness == 5.0
    assert good.relevance == pytest.approx(1 + 4 / 3, abs=0.01)  # 1 of 3 reference terms
    abstain = LexicalJudge().grade("q", "Congress", "The document does not say.", context)
    assert (abstain.faithfulness, abstain.relevance) == (5.0, 1.0)


def test_claude_judge_sends_rubric_and_parses_verdict() -> None:
    sent: dict = {}

    def parse(**kwargs):
        sent.update(kwargs)
        verdict = kwargs["output_format"](reasoning="ok", faithfulness=4, relevance=5)
        return SimpleNamespace(parsed_output=verdict, stop_reason="end_turn")

    client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    grade = ClaudeJudge("claude-opus-5-5", client=client).grade("q", "ref", "ans", [])
    assert (grade.faithfulness, grade.relevance) == (4.0, 5.0)
    assert sent["model"] == "claude-opus-5-5"
    assert "FAITHFULNESS" in sent["system"]
    assert "<reference>\nref\n</reference>" in sent["messages"][0]["content"]


def test_claude_judge_handles_refusal() -> None:
    client = SimpleNamespace(
        messages=SimpleNamespace(
            parse=lambda **_: SimpleNamespace(parsed_output=None, stop_reason="refusal")
        )
    )
    grade = ClaudeJudge("m", client=client).grade("q", "r", "a", [])
    assert grade.faithfulness is None and grade.relevance is None


def test_cached_embedder_only_embeds_new_texts(tmp_path) -> None:
    calls = []

    class Counting(HashEmbedder):
        def embed_documents(self, texts):
            calls.append(list(texts))
            return super().embed_documents(texts)

    cached = CachedEmbedder(Counting(dim=16), tmp_path)
    first = cached.embed_documents(["a", "b", "a"])
    again = CachedEmbedder(Counting(dim=16), tmp_path).embed_documents(["a", "b", "c"])
    assert calls == [["a", "b"], ["c"]]
    assert first[0] == again[0]


# --- end to end ------------------------------------------------------------------------


def test_mock_run_produces_all_configurations() -> None:
    components = build_components("mock", Settings(_env_file=None))
    summaries, results = run(
        components, load_dataset(limit=6), ["fixed", "heading"], [2, 4], 600, 60, 1
    )
    assert [(s.strategy, s.k) for s in summaries] == [
        ("fixed", 2),
        ("fixed", 4),
        ("heading", 2),
        ("heading", 4),
    ]
    assert len(results) == 6 * 4
    for s in summaries:
        assert 0 <= s.precision <= 1 and 0 <= s.recall <= 1 and 0 <= s.mrr <= 1
        assert 1 <= s.faithfulness <= 5 and 1 <= s.relevance <= 5


def test_cli_writes_report_and_enforces_threshold(tmp_path) -> None:
    out, js = tmp_path / "RESULTS.md", tmp_path / "results.json"
    args = ["--limit", "5", "--output", str(out), "--json", str(js)]
    assert main(args) == 0
    assert "| Strategy | k |" in out.read_text()
    assert json.loads(js.read_text())["mode"] == "mock"
    assert main([*args, "--fail-under-recall", "1.01"]) == 1


def test_provider_failure_is_recorded_not_fatal(tmp_path) -> None:
    from evals.run import Components, render_report

    class FailingJudge(LexicalJudge):
        def grade(self, question, reference, answer, context):
            if "Senator" in question:
                raise RuntimeError("judge unavailable")
            return super().grade(question, reference, answer, context)

    base = build_components("mock", Settings(_env_file=None))
    components = Components("mock", base.embedder, base.generator, FailingJudge())
    examples = load_dataset(limit=4)
    summaries, results = run(components, examples, ["recursive"], [3], 800, 100, 1)
    failed = [r for r in results if r.judge_reasoning.startswith("error: ")]
    assert failed and all(r.faithfulness is None for r in failed)
    assert summaries[0].faithfulness is not None
    report = render_report(
        components, summaries, results, examples, 800, 100, Settings(_env_file=None)
    )
    assert f"{len(failed)} of {len(results)} answers failed" in report
