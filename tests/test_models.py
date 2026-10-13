import uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentStatus, FileType
from app.schemas import DocumentDetail, DocumentSummary


def make_document(**overrides: Any) -> Document:
    fields: dict[str, Any] = {
        "name": "Q3 Report",
        "filename": "q3_report.pdf",
        "file_type": FileType.PDF,
        "size_bytes": 1024,
        "page_count": 3,
    }
    return Document(**(fields | overrides))


def make_chunk(**overrides: Any) -> Chunk:
    chunk_id = str(uuid.uuid4())
    fields: dict[str, Any] = {
        "id": chunk_id,
        "embedding_id": chunk_id,
        "chunk_index": 0,
        "text": "Text of a chunk.",
        "page_start": 1,
        "page_end": 1,
        "token_count": 6,
    }
    return Chunk(**(fields | overrides))


def count_chunks(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Chunk))


def test_document_with_chunks_round_trip(db_session: Session) -> None:
    document = make_document()
    document.chunks = [make_chunk(chunk_index=1, page_start=2, page_end=3), make_chunk()]
    db_session.add(document)
    db_session.commit()
    db_session.expire_all()

    loaded = db_session.get(Document, document.id)

    assert loaded is not None
    assert uuid.UUID(loaded.id).version == 4
    assert loaded.status == DocumentStatus.PENDING
    assert loaded.chunk_count == 0
    assert loaded.uploaded_at.tzinfo == UTC
    assert loaded.processed_at is None
    assert [chunk.chunk_index for chunk in loaded.chunks] == [0, 1]
    assert (loaded.chunks[1].page_start, loaded.chunks[1].page_end) == (2, 3)


def test_enums_are_stored_as_lowercase_values(db_session: Session) -> None:
    db_session.add(make_document())
    db_session.commit()

    row = db_session.execute(text("SELECT status, file_type FROM documents")).one()

    assert tuple(row) == ("pending", "pdf")


def test_database_rejects_unknown_status(db_session: Session) -> None:
    db_session.add(make_document())
    db_session.commit()

    with pytest.raises(IntegrityError):
        db_session.execute(text("UPDATE documents SET status = 'PENDING'"))


def test_timestamps_are_normalized_to_utc(db_session: Session) -> None:
    israel_time = timezone(timedelta(hours=3))
    document = make_document(uploaded_at=datetime(2026, 10, 13, 15, 0, tzinfo=israel_time))
    db_session.add(document)
    db_session.commit()
    db_session.expire_all()

    loaded = db_session.get(Document, document.id)

    assert loaded is not None
    assert loaded.uploaded_at == datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    assert loaded.uploaded_at.tzinfo == UTC


def test_naive_timestamps_are_rejected(db_session: Session) -> None:
    db_session.add(make_document(uploaded_at=datetime(2026, 10, 13, 12, 0)))

    with pytest.raises(StatementError):
        db_session.commit()


def test_deleting_document_through_orm_removes_its_chunks(db_session: Session) -> None:
    document = make_document()
    document.chunks = [make_chunk(), make_chunk(chunk_index=1)]
    db_session.add(document)
    db_session.commit()

    db_session.delete(document)
    db_session.commit()

    assert count_chunks(db_session) == 0


def test_database_cascade_removes_chunks_without_orm(db_session: Session) -> None:
    document = make_document()
    document.chunks = [make_chunk(), make_chunk(chunk_index=1)]
    db_session.add(document)
    db_session.commit()

    db_session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document.id})
    db_session.commit()

    assert count_chunks(db_session) == 0


def test_chunk_index_is_unique_per_document(db_session: Session) -> None:
    document = make_document()
    document.chunks = [make_chunk(), make_chunk()]
    db_session.add(document)

    with pytest.raises(IntegrityError):
        db_session.commit()


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"page_start": 3, "page_end": 2}, id="end-before-start"),
        pytest.param({"page_start": 1, "page_end": None}, id="only-start-set"),
        pytest.param({"page_start": None, "page_end": 1}, id="only-end-set"),
        pytest.param({"page_start": 0, "page_end": 0}, id="page-zero"),
        pytest.param({"chunk_index": -1}, id="negative-index"),
        pytest.param({"token_count": 0}, id="zero-tokens"),
        pytest.param({"text": ""}, id="empty-text"),
    ],
)
def test_invalid_chunk_is_rejected(db_session: Session, overrides: dict[str, Any]) -> None:
    document = make_document()
    document.chunks = [make_chunk(**overrides)]
    db_session.add(document)

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_text_document_chunks_have_no_pages(db_session: Session) -> None:
    document = make_document(filename="notes.txt", file_type=FileType.TXT, page_count=None)
    document.chunks = [make_chunk(page_start=None, page_end=None)]
    db_session.add(document)
    db_session.commit()

    assert count_chunks(db_session) == 1


def test_chunk_requires_existing_document(db_session: Session) -> None:
    db_session.add(make_chunk(document_id=str(uuid.uuid4())))

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_document_schemas_match_api_contract(db_session: Session) -> None:
    document = make_document()
    db_session.add(document)
    db_session.commit()

    summary = DocumentSummary.model_validate(document).model_dump(mode="json")
    detail = DocumentDetail.model_validate(document).model_dump(mode="json")

    assert set(summary) == {"id", "name", "filename", "chunks", "status", "uploaded_at"}
    assert set(detail) == set(summary) | {
        "file_type",
        "size_bytes",
        "pages",
        "error",
        "processed_at",
    }
    assert detail["chunks"] == 0
    assert detail["pages"] == 3
    assert detail["status"] == "pending"
    assert detail["file_type"] == "pdf"
    assert detail["uploaded_at"].endswith("Z")
