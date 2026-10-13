import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import Settings
from app.db.database import create_db_engine, create_session_factory, init_db
from app.db.models import Chunk, Document, DocumentStatus, FileType
from app.main import create_app
from app.services.documents import INTERRUPTED_REASON, DocumentService
from app.services.vector_store import ChromaVectorStore, VectorRecord
from tests.fakes import FakeEmbedder
from tests.pdf_factory import write_pdf

LONG_TEXT = "\n\n".join(
    " ".join(f"p{paragraph}w{word}" for word in range(12)) + "." for paragraph in range(10)
)


@pytest.fixture
def make_client(tmp_path: Path) -> Iterator[Any]:
    """Builds a client with setting overrides; for tests that need non-default limits."""
    clients: list[TestClient] = []

    def build(embedder: FakeEmbedder | None = None, **overrides: Any) -> TestClient:
        settings = Settings(_env_file=None, data_dir=tmp_path, **overrides)
        client = TestClient(create_app(settings, embedder=embedder or FakeEmbedder()))
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)


def upload(client: TestClient, filename: str, content: bytes, name: str | None = None) -> Any:
    data = {"name": name} if name is not None else {}
    return client.post("/documents", files={"file": (filename, content)}, data=data)


def chunk_rows(client: TestClient, document_id: str) -> list[Chunk]:
    with client.app.state.session_factory() as session:
        query = select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        return list(session.scalars(query))


def stored_files(settings: Settings) -> list[Path]:
    return list(settings.uploads_dir.iterdir())


def assert_nothing_stored(client: TestClient) -> None:
    assert stored_files(client.app.state.settings) == []
    assert client.get("/documents").json() == []


# --- Happy paths -------------------------------------------------------------------


def test_text_upload_is_accepted_then_processed(client: TestClient, settings: Settings) -> None:
    response = upload(client, "notes.txt", b"Q3 revenue grew 12 percent.\n\nCosts were flat.")

    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status"] == "pending"
    assert accepted["name"] == "notes.txt"
    assert accepted["file_type"] == "txt"
    assert accepted["pages"] is None

    document = client.get(f"/documents/{accepted['id']}").json()
    assert document["status"] == "completed"
    assert document["chunks"] == 1
    assert document["error"] is None
    assert document["processed_at"] is not None
    [row] = chunk_rows(client, accepted["id"])
    assert row.embedding_id == row.id
    assert (row.page_start, row.page_end) == (None, None)
    assert client.app.state.vector_store.count(accepted["id"]) == 1
    assert stored_files(settings) == [settings.uploads_dir / f"{accepted['id']}.txt"]


def test_pdf_upload_records_pages_and_page_ranges(client: TestClient, tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "report.pdf", ["Revenue grew.", "Costs fell.", "Outlook is good."])

    response = upload(client, "report.pdf", pdf.read_bytes(), name="  Q3 Report  ")

    assert response.status_code == 202
    assert response.json()["name"] == "Q3 Report"
    document = client.get(f"/documents/{response.json()['id']}").json()
    assert (document["status"], document["pages"], document["chunks"]) == ("completed", 3, 1)
    [row] = chunk_rows(client, document["id"])
    assert (row.page_start, row.page_end) == (1, 3)


def test_long_document_is_stored_in_several_batches(make_client: Any) -> None:
    client = make_client(chunk_size=30, chunk_overlap=5, embedding_batch_size=2)

    document_id = upload(client, "long.md", LONG_TEXT.encode()).json()["id"]

    document = client.get(f"/documents/{document_id}").json()
    rows = chunk_rows(client, document_id)
    assert document["status"] == "completed"
    assert document["chunks"] == len(rows) > 4
    assert [row.chunk_index for row in rows] == list(range(len(rows)))
    assert client.app.state.vector_store.count(document_id) == len(rows)


def test_list_returns_newest_first_with_summary_fields(client: TestClient) -> None:
    first = upload(client, "a.txt", b"Alpha.").json()
    second = upload(client, "b.md", b"# Beta", name="Beta notes").json()

    listed = client.get("/documents").json()

    assert [item["id"] for item in listed] == [second["id"], first["id"]]
    assert set(listed[0]) == {"id", "name", "filename", "chunks", "status", "uploaded_at"}
    assert (listed[0]["name"], listed[0]["filename"]) == ("Beta notes", "b.md")


def test_delete_removes_row_chunks_vectors_and_file(client: TestClient) -> None:
    document_id = upload(client, "notes.txt", b"Q3 revenue grew.").json()["id"]

    response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/documents/{document_id}").status_code == 404
    assert chunk_rows(client, document_id) == []
    assert client.app.state.vector_store.count(document_id) == 0
    assert_nothing_stored(client)


@pytest.mark.parametrize("method", ["get", "delete"])
def test_unknown_document_is_not_found(client: TestClient, method: str) -> None:
    response = getattr(client, method)("/documents/no-such-id")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert response.json()["error"]["details"] == {"document_id": "no-such-id"}


# --- Rejections at upload: each leaves no file and no row --------------------------


@pytest.mark.parametrize(
    ("filename", "content", "status", "code"),
    [
        ("notes.docx", b"Some text", 415, "UNSUPPORTED_FILE_TYPE"),
        ("no_extension", b"Some text", 415, "UNSUPPORTED_FILE_TYPE"),
        ("empty.txt", b"", 400, "EMPTY_DOCUMENT"),
        ("blank.md", b"  \n\t \r\n", 400, "EMPTY_DOCUMENT"),
        ("broken.pdf", b"%PDF-1.4 this is not a real pdf", 400, "UNREADABLE_DOCUMENT"),
    ],
)
def test_invalid_upload_is_rejected(
    client: TestClient, filename: str, content: bytes, status: int, code: str
) -> None:
    response = upload(client, filename, content)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert_nothing_stored(client)


def test_image_only_pdf_is_rejected_as_empty(client: TestClient, tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "scan.pdf", ["", ""])

    response = upload(client, "scan.pdf", pdf.read_bytes())

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_DOCUMENT"
    assert_nothing_stored(client)


def test_pdf_over_page_limit_is_rejected(make_client: Any, tmp_path: Path) -> None:
    client = make_client(max_pages=2)
    pdf = write_pdf(tmp_path / "long.pdf", ["One.", "Two.", "Three."])

    response = upload(client, "long.pdf", pdf.read_bytes())

    assert response.status_code == 413
    assert response.json()["error"] == {
        "code": "TOO_MANY_PAGES",
        "message": "PDF has 3 pages; the maximum is 2.",
        "details": {"pages": 3, "max_pages": 2},
    }
    assert_nothing_stored(client)


def test_file_just_over_limit_is_rejected_by_the_exact_check(make_client: Any) -> None:
    client = make_client(max_upload_mb=1)
    # Within the middleware's multipart allowance, so only the handler's exact check catches it.
    content = b"a" * (1024 * 1024 + 1)

    response = upload(client, "big.txt", content)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert response.json()["error"]["details"] == {"max_bytes": 1024 * 1024}
    assert_nothing_stored(client)


def test_file_at_the_limit_is_accepted(make_client: Any) -> None:
    client = make_client(max_upload_mb=1)

    response = upload(client, "big.txt", b"word " * (1024 * 1024 // 5))

    assert response.status_code == 202


def test_oversized_body_is_rejected_before_the_handler_runs(
    make_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(max_upload_mb=1)

    def handler_must_not_run(*_args: Any) -> None:
        raise AssertionError("the upload handler ran for an oversized body")

    monkeypatch.setattr(DocumentService, "create", handler_must_not_run)

    response = upload(client, "huge.txt", b"a" * (2 * 1024 * 1024))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


@pytest.mark.parametrize(
    ("data", "files", "field"),
    [
        ({}, None, "body.file"),
        ({"name": "   "}, {"file": ("a.txt", b"Text.")}, "body.name"),
        ({"name": "x" * 256}, {"file": ("a.txt", b"Text.")}, "body.name"),
    ],
)
def test_malformed_upload_request_is_a_validation_error(
    client: TestClient, data: dict[str, str], files: Any, field: str
) -> None:
    response = client.post("/documents", data=data, files=files)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert [error["field"] for error in response.json()["error"]["details"]["errors"]] == [field]
    assert_nothing_stored(client)


# --- Failures during background processing -----------------------------------------


def test_embedding_failure_marks_document_failed_with_reason(make_client: Any) -> None:
    client = make_client(embedder=FakeEmbedder(fail=True))

    document_id = upload(client, "notes.txt", b"Q3 revenue grew.").json()["id"]

    document = client.get(f"/documents/{document_id}").json()
    assert document["status"] == "failed"
    assert document["error"] == "Fake embedding failure."
    assert document["chunks"] == 0
    assert document["processed_at"] is not None
    assert chunk_rows(client, document_id) == []
    assert client.app.state.vector_store.count(document_id) == 0


def test_unexpected_processing_error_is_logged_and_marks_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken_chunker(*_args: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("app.services.documents.chunk_pages", broken_chunker)

    with caplog.at_level(logging.ERROR, logger="app.services.documents"):
        document_id = upload(client, "notes.txt", b"Q3 revenue grew.").json()["id"]

    document = client.get(f"/documents/{document_id}").json()
    assert document["status"] == "failed"
    assert document["error"] == "Processing failed due to an unexpected error (RuntimeError)."
    assert "Unexpected error while processing document" in caplog.text


# --- Startup recovery ---------------------------------------------------------------


def test_startup_fails_documents_interrupted_by_a_restart(settings: Settings) -> None:
    settings.chroma_dir.mkdir(parents=True)
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    with create_session_factory(engine)() as session:
        session.add(
            Document(
                id="interrupted",
                name="a",
                filename="a.txt",
                file_type=FileType.TXT,
                size_bytes=10,
                status=DocumentStatus.PROCESSING,
            )
        )
        session.add(
            Chunk(
                id="c1",
                document_id="interrupted",
                chunk_index=0,
                text="Partial.",
                token_count=1,
                embedding_id="c1",
            )
        )
        session.commit()
    engine.dispose()
    ChromaVectorStore(settings.chroma_dir).add(
        [
            VectorRecord(
                "c1", FakeEmbedder().embed_documents(["Partial."])[0], "interrupted", 0, None, None
            )
        ]
    )

    with TestClient(create_app(settings, embedder=FakeEmbedder())) as client:
        document = client.get("/documents/interrupted").json()

        assert (document["status"], document["error"]) == ("failed", INTERRUPTED_REASON)
        assert chunk_rows(client, "interrupted") == []
        assert client.app.state.vector_store.count("interrupted") == 0
