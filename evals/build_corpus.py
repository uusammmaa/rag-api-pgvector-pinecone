"""Render the plain-text corpus sources in ``evals/corpus/source`` into PDFs.

The eval runs the same PDF extraction path as the API, so the committed
PDFs are the real inputs. This script exists so the corpus is reproducible:

    pip install fpdf2
    python -m evals.build_corpus
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fpdf import FPDF

CORPUS_DIR = Path(__file__).parent / "corpus"
SOURCE_DIR = CORPUS_DIR / "source"

TITLES = {
    "us_constitution": "The Constitution of the United States (1787)",
    "declaration_of_independence": "The Declaration of Independence (1776)",
    "elements_of_style": "The Elements of Style, William Strunk Jr. (1918)",
}


def _is_heading(paragraph: str) -> bool:
    lines = paragraph.splitlines()
    return len(lines) == 1 and len(lines[0]) < 70 and not lines[0].startswith(" ")


def render(source: Path, target: Path) -> None:
    pdf = FPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_title(TITLES.get(source.stem, source.stem))
    pdf.set_author("Public domain")
    pdf.set_creation_date(datetime(2026, 1, 1, tzinfo=UTC))  # reproducible bytes
    pdf.add_page()

    for paragraph in source.read_text(encoding="utf-8").split("\n\n"):
        paragraph = paragraph.rstrip()
        if not paragraph:
            continue
        if _is_heading(paragraph):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, paragraph.strip())
            pdf.ln(2)
            continue
        pdf.set_font("Helvetica", "", 11)
        if any(line.startswith(" ") for line in paragraph.splitlines()):
            # Indented examples keep their line breaks.
            body = "\n".join(line.strip() for line in paragraph.splitlines())
        else:
            body = " ".join(line.strip() for line in paragraph.splitlines())
        pdf.multi_cell(0, 5.5, body)
        pdf.ln(3)

    pdf.output(str(target))


def main() -> None:
    for source in sorted(SOURCE_DIR.glob("*.txt")):
        target = CORPUS_DIR / f"{source.stem}.pdf"
        render(source, target)
        print(f"wrote {target.relative_to(CORPUS_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
