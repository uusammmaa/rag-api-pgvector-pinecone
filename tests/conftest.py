from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

from app.config import Settings
from app.main import create_app

SAMPLE_PARAGRAPHS = [
    ("CHAPTER ONE", None),
    (None, "The lighthouse keeper lit the lamp every evening at dusk without fail."),
    (None, "Ships passing the northern reef relied on the beam to avoid the rocks."),
    ("CHAPTER TWO", None),
    (None, "In winter the keeper rowed to the mainland once a month for supplies."),
    (None, "The supply boat carried flour, lamp oil, coal and letters from home."),
]


def make_pdf(paragraphs: list[tuple[str | None, str | None]] | None = None) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    for heading, body in paragraphs if paragraphs is not None else SAMPLE_PARAGRAPHS:
        if heading:
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        if body:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return make_pdf()


@pytest.fixture(scope="session")
def blank_pdf() -> bytes:
    return make_pdf([])


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        vector_store="memory",
        embedding_provider="hash",
        embedding_dim=256,
        llm_provider="mock",
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
        max_upload_mb=1,
        anthropic_api_key=None,
        openai_api_key=None,
        pinecone_api_key=None,
        database_url=None,
    )


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def uploaded_doc(client: TestClient, sample_pdf: bytes) -> str:
    response = client.post("/upload", files={"file": ("book.pdf", sample_pdf, "application/pdf")})
    assert response.status_code == 200, response.text
    return response.json()["doc_id"]
