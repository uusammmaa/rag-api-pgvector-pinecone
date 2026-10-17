from __future__ import annotations

import itertools

import pytest

from app.rag.chunking import (
    FixedSizeChunker,
    HeadingAwareChunker,
    RecursiveChunker,
    build_chunker,
    detect_heading,
)

TEXT = (
    "ARTICLE ONE\n"
    "Section 1. All legislative powers herein granted shall be vested in a Congress, "
    "which shall consist of a Senate and a House of Representatives.\n"
    "Section 2. The House shall be composed of members chosen every second year by the "
    "people of the several states. No person shall be a representative who has not "
    "attained the age of twenty five years.\n"
    "ARTICLE TWO\n"
    "Section 1. The executive power shall be vested in a President. He shall hold his "
    "office during the term of four years.\n"
) * 3

STRATEGIES = ["fixed", "recursive", "heading"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunks_are_exact_slices_of_the_source(strategy: str) -> None:
    chunks = build_chunker(strategy, 200, 30).split(TEXT)
    assert chunks
    for i, chunk in enumerate(chunks):
        assert chunk.index == i
        assert TEXT[chunk.start : chunk.end] == chunk.text
        assert chunk.text == chunk.text.strip()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunks_respect_size_limit(strategy: str) -> None:
    for chunk in build_chunker(strategy, 200, 30).split(TEXT):
        assert len(chunk.text) <= 200


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_non_space_character_is_covered(strategy: str) -> None:
    covered = [False] * len(TEXT)
    for chunk in build_chunker(strategy, 200, 30).split(TEXT):
        for i in range(chunk.start, chunk.end):
            covered[i] = True
    missing = [i for i, ch in enumerate(TEXT) if not ch.isspace() and not covered[i]]
    assert missing == []


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_empty_and_whitespace_text(strategy: str) -> None:
    chunker = build_chunker(strategy, 200, 30)
    assert chunker.split("") == []
    assert chunker.split("   \n\n  ") == []


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_short_text_is_a_single_chunk(strategy: str) -> None:
    chunks = build_chunker(strategy, 200, 30).split("  A short sentence.  ")
    assert [c.text for c in chunks] == ["A short sentence."]


def test_fixed_chunker_does_not_split_words() -> None:
    words = set(TEXT.split())
    for chunk in FixedSizeChunker(120, 20).split(TEXT):
        first, last = chunk.text.split()[0], chunk.text.split()[-1]
        assert first in words
        assert last in words


def test_fixed_chunker_overlaps_neighbours() -> None:
    chunks = FixedSizeChunker(150, 40).split(TEXT)
    assert all(b.start < a.end for a, b in itertools.pairwise(chunks))


def test_recursive_chunker_prefers_line_boundaries() -> None:
    chunks = RecursiveChunker(400, 0).split(TEXT)
    # With room to spare every chunk should end where a line ends.
    assert all(TEXT[c.end : c.end + 1] in ("\n", "") for c in chunks)


def test_recursive_chunker_overlap_repeats_text() -> None:
    chunks = RecursiveChunker(120, 60).split(TEXT)
    overlapping = sum(1 for a, b in itertools.pairwise(chunks) if b.start < a.end)
    assert overlapping > 0


def test_recursive_chunker_hard_splits_unbroken_text() -> None:
    text = "x" * 1000
    chunks = RecursiveChunker(300, 0).split(text)
    assert [len(c.text) for c in chunks] == [300, 300, 300, 100]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Article 1", (1, "Article 1")),
        ("ARTICLE THREE", (1, "ARTICLE THREE")),
        ("II. ELEMENTARY RULES OF USAGE", (1, "II. ELEMENTARY RULES OF USAGE")),
        ("THE DECLARATION OF INDEPENDENCE", (1, "THE DECLARATION OF INDEPENDENCE")),
        ("Section 8.  The Congress shall have Power to lay taxes", (2, "Section 8")),
        ("10. Use the active voice.", (2, "10. Use the active voice.")),
        ("The Congress shall have Power to lay and collect Taxes,", None),
        ("", None),
        ("OK", None),
    ],
)
def test_detect_heading(line: str, expected: tuple[int, str] | None) -> None:
    assert detect_heading(line) == expected


def test_heading_chunker_attaches_section_paths() -> None:
    chunks = HeadingAwareChunker(200, 20, min_section=20).split(TEXT)
    sections = {c.section for c in chunks}
    assert "ARTICLE ONE > Section 1" in sections
    assert "ARTICLE ONE > Section 2" in sections
    assert "ARTICLE TWO > Section 1" in sections


def test_heading_chunker_never_mixes_sections() -> None:
    chunker = HeadingAwareChunker(500, 50, min_section=20)
    bounds = [(s, e) for s, e, _ in chunker.sections(TEXT)]
    for chunk in chunker.split(TEXT):
        assert any(s <= chunk.start and chunk.end <= e for s, e in bounds)


def test_heading_chunker_folds_heading_only_sections_forward() -> None:
    sections = HeadingAwareChunker(500, 50, min_section=40).sections(TEXT)
    # "ARTICLE ONE" alone is shorter than min_section, so it joins "Section 1".
    assert all(label != "ARTICLE ONE" for _, _, label in sections)
    assert TEXT[sections[0][0] :].startswith("ARTICLE ONE")


def test_embed_text_includes_section() -> None:
    chunk = HeadingAwareChunker(200, 20, min_section=20).split(TEXT)[0]
    assert chunk.embed_text.startswith(f"{chunk.section}\n")
    plain = RecursiveChunker(200, 20).split(TEXT)[0]
    assert plain.embed_text == plain.text


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(size, overlap)


def test_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        build_chunker("semantic", 100, 10)  # type: ignore[arg-type]
