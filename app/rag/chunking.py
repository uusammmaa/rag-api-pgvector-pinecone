"""Chunking strategies.

Every chunker returns ``Chunk`` objects whose ``text`` is an exact slice
``source[start:end]`` of the input. Keeping offsets lets the eval harness score
retrieval against labelled evidence spans independently of the strategy, and
lets the API map chunks back to PDF pages.

Strategies:

* ``fixed``: sliding character windows, nudged to the nearest word boundary.
* ``recursive``: split on paragraph, line, sentence then word boundaries until
  pieces fit, then greedily merge neighbours up to ``chunk_size``.
* ``heading``: detect headings (articles, chapters, numbered rules, ALL CAPS
  titles), split into sections first, then run the recursive splitter inside
  each section. Chunks carry their section path, which is prepended to the
  text that gets embedded.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.config import ChunkStrategy


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    start: int
    end: int
    section: str | None = None
    page: int | None = None

    @property
    def embed_text(self) -> str:
        """Text sent to the embedding model: section path + body."""
        return f"{self.section}\n{self.text}" if self.section else self.text


class Chunker(Protocol):
    name: str

    def split(self, text: str) -> list[Chunk]: ...


Span = tuple[int, int]


def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")


def _to_chunks(text: str, spans: Sequence[Span], section: str | None = None) -> list[Chunk]:
    chunks = []
    for start, end in spans:
        start, end = _trim(text, start, end)
        if end > start:
            chunks.append(Chunk(text[start:end], len(chunks), start, end, section))
    return chunks


class FixedSizeChunker:
    name = "fixed"

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        _validate(chunk_size, overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        spans: list[Span] = []
        start, n = 0, len(text)
        slack = self.chunk_size // 5
        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                # Back off to whitespace so words are not cut, within 20% of the window.
                cut = text.rfind(" ", end - slack, end)
                cut = max(cut, text.rfind("\n", end - slack, end))
                if cut > start:
                    end = cut
            spans.append((start, end))
            if end >= n:
                break
            nxt = end - self.overlap
            if self.overlap:
                space = text.find(" ", nxt, end)
                nxt = space + 1 if space != -1 else nxt
            start = max(nxt, start + 1)
        return _to_chunks(text, spans)


class RecursiveChunker:
    name = "recursive"
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        _validate(chunk_size, overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        return _to_chunks(text, self.split_spans(text, 0, len(text)))

    def split_spans(self, text: str, start: int, end: int) -> list[Span]:
        pieces = self._pieces(text, start, end, self.separators)
        return self._merge(pieces)

    def _pieces(self, text: str, start: int, end: int, seps: Sequence[str]) -> list[Span]:
        """Cut [start, end) into contiguous spans no longer than chunk_size."""
        if end - start <= self.chunk_size:
            return [(start, end)]
        for i, sep in enumerate(seps):
            if sep == "":
                return [
                    (s, min(s + self.chunk_size, end)) for s in range(start, end, self.chunk_size)
                ]
            if text.find(sep, start, end) == -1:
                continue
            out: list[Span] = []
            cursor = start
            while cursor < end:
                hit = text.find(sep, cursor, end)
                piece_end = end if hit == -1 else hit + len(sep)  # separator stays on the left
                if piece_end - cursor <= self.chunk_size:
                    out.append((cursor, piece_end))
                else:
                    out.extend(self._pieces(text, cursor, piece_end, seps[i + 1 :]))
                cursor = piece_end
            return out
        return [(start, end)]

    def _merge(self, pieces: list[Span]) -> list[Span]:
        """Greedily pack contiguous pieces into windows, carrying ``overlap`` chars back."""
        merged: list[Span] = []
        window: list[Span] = []
        for piece in pieces:
            if window and piece[1] - window[0][0] > self.chunk_size:
                merged.append((window[0][0], window[-1][1]))
                # Keep trailing pieces that fit inside the overlap budget.
                carry: list[Span] = []
                for prev in reversed(window):
                    if window[-1][1] - prev[0] > self.overlap:
                        break
                    carry.insert(0, prev)
                window = carry
                while window and piece[1] - window[0][0] > self.chunk_size:
                    window.pop(0)
            window.append(piece)
        if window:
            merged.append((window[0][0], window[-1][1]))
        return merged


# Level 1: document divisions. Level 2: subsections inside them.
_LEVEL1 = re.compile(
    r"^(?:(?:article|chapter|part)\s+[\w-]+\.?|[IVXLC]+\.\s+[A-Z][A-Z ,;:'\-]+)\s*$",
    re.IGNORECASE,
)
_LEVEL2 = re.compile(r"^(?:section\s+\d+\.?|\d{1,2}\.\s+[A-Z])", re.IGNORECASE)
_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9 ,.;:'\-]{3,79}$")


def detect_heading(line: str) -> tuple[int, str] | None:
    """Return (level, label) when ``line`` looks like a heading."""
    stripped = line.strip()
    if not stripped:
        return None
    is_caps = _CAPS.match(stripped) and sum(c.isalpha() for c in stripped) > 3
    if len(stripped) <= 160 and (_LEVEL1.match(stripped) or is_caps):
        return 1, stripped
    match = _LEVEL2.match(stripped)
    if match:
        label = stripped
        if stripped.lower().startswith("section"):
            label = match.group(0).rstrip(". ")
        elif len(label) > 80:
            label = label[:77].rstrip() + "..."
        return 2, label
    return None


class HeadingAwareChunker:
    name = "heading"

    def __init__(self, chunk_size: int = 800, overlap: int = 100, min_section: int = 120) -> None:
        self.inner = RecursiveChunker(chunk_size, overlap)
        self.min_section = min_section

    def sections(self, text: str) -> list[tuple[int, int, str | None]]:
        """Split text at heading lines into (start, end, section_path) triples."""
        bounds: list[tuple[int, str | None]] = [(0, None)]
        path: dict[int, str] = {}
        offset = 0
        for line in text.splitlines(keepends=True):
            heading = detect_heading(line)
            if heading:
                level, label = heading
                path[level] = label
                if level == 1:
                    path.pop(2, None)
                bounds.append((offset, " > ".join(path[k] for k in sorted(path))))
            offset += len(line)

        sections = []
        for i, (start, label) in enumerate(bounds):
            end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
            if end > start:
                sections.append((start, end, label))

        # A heading followed directly by another heading (e.g. "Article 1" then
        # "Section 1.") would produce a near-empty section: fold it forward.
        merged: list[tuple[int, int, str | None]] = []
        pending: int | None = None
        for i, (start, end, label) in enumerate(sections):
            start = pending if pending is not None else start
            pending = None
            if end - start < self.min_section and i < len(sections) - 1:
                pending = start
                continue
            merged.append((start, end, label))
        return merged

    def split(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for start, end, label in self.sections(text):
            for s, e in self.inner.split_spans(text, start, end):
                s, e = _trim(text, s, e)
                if e > s:
                    chunks.append(Chunk(text[s:e], len(chunks), s, e, label))
        return chunks


def build_chunker(strategy: ChunkStrategy, chunk_size: int, overlap: int) -> Chunker:
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size, overlap)
    if strategy == "recursive":
        return RecursiveChunker(chunk_size, overlap)
    if strategy == "heading":
        return HeadingAwareChunker(chunk_size, overlap)
    raise ValueError(f"Unknown chunk strategy: {strategy!r}")
