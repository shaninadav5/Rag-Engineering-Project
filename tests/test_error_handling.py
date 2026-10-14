"""Every edge case in spec.md, through the HTTP API.

Where each edge case is tested ("api" = test_documents_api.py, "here" = this file):

1. Empty document: api test_invalid_upload_is_rejected, test_image_only_pdf_is_rejected_as_empty
2. Over the size or page limit: api test_file_just_over_limit_is_rejected_by_the_exact_check,
   test_oversized_body_is_rejected_before_the_handler_runs, test_pdf_over_page_limit_is_rejected
3. No relevant passage: here test_no_relevant_passage_refuses_without_calling_the_llm
4. Unknown document id: here test_unknown_document_id_is_a_clear_error
5. Unsupported file type: api test_invalid_upload_is_rejected
6. LLM unavailable / erroring: here test_llm_failure_is_reported_and_the_service_keeps_working,
   test_unreachable_ollama_is_reported_by_the_real_client
7. Embedding fails: api test_embedding_failure_marks_document_failed_with_reason (processing);
   here test_question_embedding_failure_is_a_clear_error (question)

Additional cases:

- Document not ready: here test_document_not_ready_is_a_clear_error
- No documents processed yet: here test_no_documents_processed_yet_refuses_without_calling_the_llm
- Empty question, top_k out of range: here test_invalid_ask_request_is_a_validation_error
- Get / delete a document that does not exist: api test_unknown_document_is_not_found
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import AppError, EmbeddingUnavailableError, LLMError, LLMUnavailableError
from app.main import create_app
from app.services.prompts import REFUSAL
from tests.fakes import FakeEmbedder, FakeLLM

MakeClient = Callable[..., TestClient]
REPORT = b"Q3 revenue grew 12 percent."


class QueryFailingEmbedder(FakeEmbedder):
    """Embeds documents normally but fails on questions."""

    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingUnavailableError("The embedding model failed to process the text.")


def upload(client: TestClient, content: bytes = REPORT) -> str:
    response = client.post("/documents", files={"file": ("report.txt", content)})
    assert response.status_code == 202
    return response.json()["id"]


def assert_error(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.json()
    error = response.json()["error"]
    assert error["code"] == code
    return error


def test_no_relevant_passage_refuses_without_calling_the_llm(
    client: TestClient, llm: FakeLLM, settings: Settings
) -> None:
    upload(client)

    response = client.post("/ask", json={"question": "Kubernetes cluster autoscaling"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": REFUSAL,
        "sources": [],
        "model": settings.llm_model,
        "tokens_used": 0,
    }
    assert llm.prompts == []


def test_no_documents_processed_yet_refuses_without_calling_the_llm(
    client: TestClient, llm: FakeLLM
) -> None:
    response = client.post("/ask", json={"question": "Q3 revenue grew percent"})

    assert (response.json()["answer"], response.json()["sources"]) == (REFUSAL, [])
    assert llm.prompts == []


def test_unknown_document_id_is_a_clear_error(client: TestClient) -> None:
    report_id = upload(client)

    response = client.post(
        "/ask", json={"question": "Q3 revenue", "document_ids": [report_id, "no-such-id"]}
    )

    error = assert_error(response, 404, "DOCUMENT_NOT_FOUND")
    assert error["message"] == "Unknown document id(s): no-such-id"
    assert error["details"] == {"missing_ids": ["no-such-id"]}


def test_document_not_ready_is_a_clear_error(make_client: MakeClient) -> None:
    client = make_client(embedder=FakeEmbedder(fail=True))
    failed_id = upload(client)

    response = client.post("/ask", json={"question": "Q3 revenue", "document_ids": [failed_id]})

    error = assert_error(response, 409, "DOCUMENT_NOT_READY")
    assert error["details"] == {"not_ready": {failed_id: "failed"}}


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMUnavailableError("The language model service is unreachable."), 503, "LLM_UNAVAILABLE"),
        (LLMError("The language model returned an invalid response."), 502, "LLM_ERROR"),
    ],
)
def test_llm_failure_is_reported_and_the_service_keeps_working(
    make_client: MakeClient, error: AppError, status: int, code: str
) -> None:
    client = make_client(llm=FakeLLM(error=error))
    upload(client)

    response = client.post("/ask", json={"question": "Q3 revenue grew percent"})

    assert assert_error(response, status, code)["message"] == error.message
    assert client.get("/health").status_code == 200
    assert client.post("/documents", files={"file": ("more.txt", b"More text.")}).status_code == 202


def test_unreachable_ollama_is_reported_by_the_real_client(tmp_path: Path) -> None:
    # No llm argument: the app builds its real OllamaClient, pointed at a port nothing listens on.
    settings = Settings(_env_file=None, data_dir=tmp_path, ollama_base_url="http://127.0.0.1:9")

    with TestClient(create_app(settings, embedder=FakeEmbedder())) as client:
        upload(client)
        response = client.post("/ask", json={"question": "Q3 revenue grew percent"})

    error = assert_error(response, 503, "LLM_UNAVAILABLE")
    assert "http://127.0.0.1:9 is unreachable" in error["message"]


def test_question_embedding_failure_is_a_clear_error(make_client: MakeClient) -> None:
    client = make_client(embedder=QueryFailingEmbedder())
    upload(client)

    response = client.post("/ask", json={"question": "Q3 revenue"})

    assert_error(response, 503, "EMBEDDING_UNAVAILABLE")


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({}, "body.question"),
        ({"question": "   "}, "body.question"),
        ({"question": "x" * 2001}, "body.question"),
        ({"question": "q", "top_k": 0}, "body.top_k"),
        ({"question": "q", "top_k": 21}, "body.top_k"),
        ({"question": "q", "document_ids": []}, "body.document_ids"),
    ],
)
def test_invalid_ask_request_is_a_validation_error(
    client: TestClient, body: dict[str, Any], field: str
) -> None:
    error = assert_error(client.post("/ask", json=body), 422, "VALIDATION_ERROR")

    assert [item["field"] for item in error["details"]["errors"]] == [field]


def test_ask_limits_follow_the_settings(make_client: MakeClient) -> None:
    client = make_client(top_k=2, top_k_max=3, max_question_chars=10)

    too_many = client.post("/ask", json={"question": "q", "top_k": 4})
    too_long = client.post("/ask", json={"question": "x" * 11})

    assert (
        "less than or equal to 3"
        in assert_error(too_many, 422, "VALIDATION_ERROR")["details"]["errors"][0]["message"]
    )
    assert (
        "at most 10 characters"
        in assert_error(too_long, 422, "VALIDATION_ERROR")["details"]["errors"][0]["message"]
    )
