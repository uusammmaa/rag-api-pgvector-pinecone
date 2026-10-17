"""HTTP routes, run against offline providers (memory store, hash embedder, mock LLM)."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.errors import UpstreamError
from app.middleware import REQUEST_ID_HEADER


def upload(client: TestClient, data: bytes, name: str = "book.pdf", ctype: str = "application/pdf"):
    return client.post("/upload", files={"file": (name, data, ctype)})


def assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == code
    assert error["request_id"] == response.headers[REQUEST_ID_HEADER]
    return error


# --- health and request IDs ----------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert len(response.headers[REQUEST_ID_HEADER]) == 32


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-abc.123"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-abc.123"


def test_malformed_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "bad id with spaces"})
    assert response.headers[REQUEST_ID_HEADER] != "bad id with spaces"


def test_logs_carry_the_request_id(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.request"):
        client.get("/health", headers={REQUEST_ID_HEADER: "rid-42"})
    record = next(r for r in caplog.records if r.name == "app.request")
    assert record.request_id == "rid-42"
    assert record.status == 200
    assert record.duration_ms >= 0


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    assert_error(client.get("/nope"), 404, "not_found")


def test_unhandled_exception_returns_500_envelope(client: TestClient, app) -> None:
    @app.get("/explode")
    def explode() -> None:
        raise RuntimeError("secret internals")

    response = client.get("/explode")
    error = assert_error(response, 500, "internal_error")
    assert "secret" not in error["message"]


# --- upload ------------------------------------------------------------------------


def test_upload_pdf(client: TestClient, sample_pdf: bytes) -> None:
    response = upload(client, sample_pdf)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["filename"] == "book.pdf"
    assert body["chunks"] >= 2
    assert body["pages"] == 1
    assert body["vector_store"] == "memory"
    assert body["chunk_strategy"] == "heading"


def test_upload_accepts_octet_stream_when_bytes_are_pdf(client: TestClient, sample_pdf: bytes):
    assert upload(client, sample_pdf, ctype="application/octet-stream").status_code == 200


def test_upload_strips_directories_from_filename(client: TestClient, sample_pdf: bytes) -> None:
    assert upload(client, sample_pdf, name="../../etc/x.pdf").json()["filename"] == "x.pdf"


def test_upload_rejects_non_pdf_extension(client: TestClient) -> None:
    assert_error(upload(client, b"hello", "notes.txt", "text/plain"), 415, "unsupported_media_type")


def test_upload_rejects_disallowed_content_type(client: TestClient, sample_pdf: bytes) -> None:
    assert_error(upload(client, sample_pdf, ctype="image/png"), 415, "unsupported_media_type")


def test_upload_rejects_fake_pdf(client: TestClient) -> None:
    error = assert_error(upload(client, b"<html>not a pdf</html>"), 415, "unsupported_media_type")
    assert "%PDF" in error["message"]


def test_upload_rejects_empty_file(client: TestClient) -> None:
    assert_error(upload(client, b""), 400, "bad_request")


def test_upload_rejects_oversized_file(client: TestClient) -> None:
    big = b"%PDF-1.7\n" + b"0" * (2 * 1024 * 1024)  # limit is 1 MB in tests
    assert_error(upload(client, big), 413, "payload_too_large")


def test_upload_rejects_pdf_without_text(client: TestClient, blank_pdf: bytes) -> None:
    error = assert_error(upload(client, blank_pdf), 422, "unprocessable_document")
    assert "no extractable text" in error["message"]


def test_upload_rejects_corrupt_pdf(client: TestClient) -> None:
    assert_error(upload(client, b"%PDF-1.7\ngarbage"), 422, "unprocessable_document")


def test_upload_rejects_unknown_store(client: TestClient, sample_pdf: bytes) -> None:
    response = client.post(
        "/upload",
        files={"file": ("a.pdf", sample_pdf, "application/pdf")},
        data={"vector_store": "faiss"},
    )
    assert_error(response, 422, "validation_error")


def test_upload_to_unconfigured_store_is_503(client: TestClient, sample_pdf: bytes) -> None:
    response = client.post(
        "/upload",
        files={"file": ("a.pdf", sample_pdf, "application/pdf")},
        data={"vector_store": "supabase"},
    )
    error = assert_error(response, 503, "provider_not_configured")
    assert "DATABASE_URL" in error["message"]


# --- chat and query ------------------------------------------------------------------


def test_chat_streams_grounded_text(client: TestClient, uploaded_doc: str) -> None:
    response = client.post(
        "/chat",
        json={"question": "How often did the keeper row for supplies?", "doc_id": uploaded_doc},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "once a month" in response.text
    assert int(response.headers["x-retrieved-chunks"]) >= 1


def test_query_returns_answer_and_sources(client: TestClient, uploaded_doc: str) -> None:
    response = client.post(
        "/query", json={"question": "What did ships rely on?", "doc_id": uploaded_doc, "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert "beam" in body["answer"]
    assert body["model"] == "mock-extractive"
    assert 1 <= len(body["sources"]) <= 2
    assert body["sources"][0]["chunk_id"].startswith(uploaded_doc)
    assert body["sources"][0]["page"] == 1


def test_chat_unknown_document_is_404(client: TestClient) -> None:
    response = client.post("/chat", json={"question": "anything", "doc_id": "missing"})
    assert_error(response, 404, "document_not_found")


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "doc_id": "d"},
        {"question": "   ", "doc_id": "d"},
        {"question": "q", "doc_id": ""},
        {"question": "q"},
        {"question": "q", "doc_id": "d", "top_k": 0},
        {"question": "q", "doc_id": "d", "vector_store": "faiss"},
    ],
)
def test_chat_validation_errors(client: TestClient, payload: dict) -> None:
    error = assert_error(client.post("/chat", json=payload), 422, "validation_error")
    assert error["details"]


def test_question_length_limit(client: TestClient, uploaded_doc: str) -> None:
    response = client.post("/query", json={"question": "x" * 2001, "doc_id": uploaded_doc})
    assert_error(response, 400, "bad_request")


def test_chat_with_unconfigured_pinecone_is_503(client: TestClient) -> None:
    response = client.post(
        "/chat", json={"question": "q", "doc_id": "d", "vector_store": "pinecone"}
    )
    assert_error(response, 503, "provider_not_configured")


def test_upstream_failure_before_streaming_is_502(client: TestClient, app, uploaded_doc: str):
    class BrokenEmbedder:
        dim, model = 256, "broken"

        def embed_query(self, text: str):
            raise UpstreamError("Embedding provider request failed.")

    app.state.services.__dict__["embedder"] = BrokenEmbedder()
    response = client.post("/chat", json={"question": "q", "doc_id": uploaded_doc})
    assert_error(response, 502, "upstream_error")


def test_upstream_failure_mid_stream_ends_with_error_line(
    client: TestClient, app, uploaded_doc: str
) -> None:
    class FlakyGenerator:
        model = "flaky"

        def stream(self, question, chunks) -> Iterator[str]:
            yield "Partial answer"
            raise UpstreamError("Language model request failed.")

    app.state.services.__dict__["generator"] = FlakyGenerator()
    response = client.post(
        "/chat", json={"question": "q", "doc_id": uploaded_doc}, headers={REQUEST_ID_HEADER: "r1"}
    )
    assert response.status_code == 200
    assert response.text.startswith("Partial answer")
    assert "[error] Language model request failed. (request_id=r1)" in response.text
