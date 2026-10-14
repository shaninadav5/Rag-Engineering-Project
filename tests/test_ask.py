"""End-to-end question answering: upload -> background processing -> POST /ask (fake LLM)."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.services.prompts import REFUSAL, SYSTEM_PROMPT
from tests.fakes import FakeLLM
from tests.pdf_factory import write_pdf

# With chunk_size=8 and no overlap, each sentence below becomes its own chunk.
REPORT = "Q3 revenue grew 12 percent.\n\nThe Berlin office opened in March."
PARIS_REPORT = "Q3 revenue grew 20 percent in Paris."
QUESTION = "Q3 revenue grew percent"

MakeClient = Callable[..., TestClient]


@pytest.fixture
def client(make_client: MakeClient, llm: FakeLLM) -> TestClient:
    return make_client(llm=llm, chunk_size=8, chunk_overlap=0)


def upload(client: TestClient, filename: str, content: bytes, name: str | None = None) -> str:
    data = {"name": name} if name is not None else {}
    response = client.post("/documents", files={"file": (filename, content)}, data=data)
    assert response.status_code == 202
    document_id = response.json()["id"]
    assert client.get(f"/documents/{document_id}").json()["status"] == "completed"
    return document_id


def ask(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/ask", json=body)
    assert response.status_code == 200, response.json()
    return response.json()


def test_question_is_answered_from_the_uploaded_document_with_sources(
    client: TestClient, llm: FakeLLM
) -> None:
    report_id = upload(client, "report.txt", REPORT.encode())

    result = ask(client, question=QUESTION)

    assert result["answer"] == "Revenue grew 12% [1]."
    assert (result["model"], result["tokens_used"]) == (FakeLLM.MODEL, FakeLLM.TOKENS)
    [source] = result["sources"]
    assert source == {
        "document_id": report_id,
        "document_name": "report.txt",
        "chunk_text": "Q3 revenue grew 12 percent.",
        "page": None,
        "page_end": None,
        "relevance_score": pytest.approx(source["relevance_score"]),
    }
    assert source["relevance_score"] > 0.5
    # The LLM saw the rules, the retrieved chunk as source [1], and the question.
    [(system, user)] = llm.prompts
    assert system == SYSTEM_PROMPT
    assert (
        user == f"Sources:\n\n[1] report.txt\nQ3 revenue grew 12 percent.\n\nQuestion: {QUESTION}"
    )


def test_pdf_sources_carry_display_name_and_pages(client: TestClient, tmp_path: Path) -> None:
    # Page 1 alone fills a chunk (8 words), so page 2 starts a new chunk.
    pdf = write_pdf(
        tmp_path / "q3.pdf",
        ["Revenue grew 12 percent in the third quarter.", "Berlin office opened."],
    )
    upload(client, "q3.pdf", pdf.read_bytes(), name="Q3 Report")

    [source] = ask(client, question="Berlin office opened")["sources"]

    assert (source["document_name"], source["page"], source["page_end"]) == ("Q3 Report", 2, 2)


def test_one_question_searches_all_documents(client: TestClient) -> None:
    report_id = upload(client, "report.txt", REPORT.encode())
    paris_id = upload(client, "paris.txt", PARIS_REPORT.encode())

    sources = ask(client, question=QUESTION)["sources"]

    assert {source["document_id"] for source in sources} == {report_id, paris_id}


def test_document_filter_limits_sources_to_those_documents(client: TestClient) -> None:
    upload(client, "report.txt", REPORT.encode())
    paris_id = upload(client, "paris.txt", PARIS_REPORT.encode())

    sources = ask(client, question=QUESTION, document_ids=[paris_id])["sources"]

    assert sources
    assert {source["document_id"] for source in sources} == {paris_id}


def test_top_k_limits_the_number_of_sources(client: TestClient) -> None:
    upload(client, "report.txt", REPORT.encode())
    upload(client, "paris.txt", PARIS_REPORT.encode())

    assert len(ask(client, question=QUESTION, top_k=1)["sources"]) == 1


def test_sources_are_ordered_by_relevance(client: TestClient) -> None:
    upload(client, "report.txt", REPORT.encode())
    upload(client, "paris.txt", PARIS_REPORT.encode())

    scores = [source["relevance_score"] for source in ask(client, question=QUESTION)["sources"]]

    assert scores == sorted(scores, reverse=True)


def test_model_refusal_returns_the_exact_sentence_and_no_sources(
    make_client: MakeClient,
) -> None:
    client = make_client(llm=FakeLLM(answer="  i don't have information about that  "))
    upload(client, "report.txt", REPORT.encode())

    result = ask(client, question=QUESTION)

    assert (result["answer"], result["sources"]) == (REFUSAL, [])
    assert result["tokens_used"] == FakeLLM.TOKENS


def test_out_of_range_citations_are_stripped_from_the_answer(make_client: MakeClient) -> None:
    client = make_client(llm=FakeLLM(answer="Revenue grew 12% [1] and also [9]."))
    upload(client, "report.txt", REPORT.encode())

    result = ask(client, question=QUESTION)

    assert result["answer"] == "Revenue grew 12% [1] and also."
    assert result["sources"]


def test_prompt_overhead_that_leaves_no_room_for_sources_refuses_without_the_llm(
    make_client: MakeClient,
) -> None:
    llm = FakeLLM()
    # chunk_size equals max_context_tokens; the system prompt alone exceeds that budget.
    client = make_client(llm=llm, chunk_size=8, chunk_overlap=0, max_context_tokens=8)
    upload(client, "report.txt", REPORT.encode())

    result = ask(client, question=QUESTION)

    assert (result["answer"], result["sources"]) == (REFUSAL, [])
    assert llm.prompts == []


def test_deleted_document_is_never_a_source(client: TestClient) -> None:
    report_id = upload(client, "report.txt", REPORT.encode())
    paris_id = upload(client, "paris.txt", PARIS_REPORT.encode())
    assert client.delete(f"/documents/{report_id}").status_code == 204

    sources = ask(client, question=QUESTION)["sources"]

    assert {source["document_id"] for source in sources} == {paris_id}
