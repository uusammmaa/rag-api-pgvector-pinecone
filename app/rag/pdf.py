"""PDF text extraction."""

from __future__ import annotations

import io
import logging
from bisect import bisect_right
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.errors import UnprocessableDocumentError

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"

NO_TEXT_MESSAGE = (
    "This PDF has no extractable text. It is probably a scan, a screenshot or a "
    "print-to-PDF saved as images. Upload a text-based PDF (one where you can select "
    "and copy text in a reader)."
)


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    page_starts: tuple[int, ...]  # character offset where each page begins

    @property
    def pages(self) -> int:
        return len(self.page_starts)

    def page_for_offset(self, offset: int) -> int:
        """1-based page number containing ``offset``."""
        return max(1, bisect_right(self.page_starts, offset))


def extract_pdf(data: bytes) -> ExtractedDocument:
    """Extract text page by page, keeping page boundaries for citations."""
    try:
        reader = PdfReader(io.BytesIO(data))
        page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        logger.warning("pdf parse failed: %s", exc)
        raise UnprocessableDocumentError("The file could not be parsed as a PDF.") from exc

    parts: list[str] = []
    starts: list[int] = []
    offset = 0
    for text in page_texts:
        starts.append(offset)
        parts.append(text)
        offset += len(text) + 1  # joined with "\n"
    document = ExtractedDocument(text="\n".join(parts), page_starts=tuple(starts))
    if not document.text.strip():
        raise UnprocessableDocumentError(NO_TEXT_MESSAGE)
    return document
