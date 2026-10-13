from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from pydantic import StringConstraints
from sqlalchemy import select

from app.db.models import Document
from app.dependencies import DocumentServiceDep, SessionDep
from app.errors import DocumentNotFoundError
from app.schemas import DocumentDetail, DocumentSummary

router = APIRouter(prefix="/documents", tags=["documents"])

DisplayName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]


# Plain `def` endpoints run in a worker thread, so copying the file, validating it, and the
# database calls do not block the event loop.
@router.post("", status_code=202)
def upload_document(
    service: DocumentServiceDep,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF, TXT, or Markdown file")],
    name: Annotated[DisplayName | None, Form(description="Defaults to the filename")] = None,
) -> DocumentDetail:
    document = service.create(file.file, file.filename or "", name)
    background_tasks.add_task(service.process, document.id)
    return DocumentDetail.model_validate(document)


@router.get("")
def list_documents(session: SessionDep) -> list[DocumentSummary]:
    documents = session.scalars(select(Document).order_by(Document.uploaded_at.desc())).all()
    return [DocumentSummary.model_validate(document) for document in documents]


@router.get("/{document_id}")
def get_document(document_id: str, session: SessionDep) -> DocumentDetail:
    document = session.get(Document, document_id)
    if document is None:
        raise DocumentNotFoundError.for_id(document_id)
    return DocumentDetail.model_validate(document)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, service: DocumentServiceDep) -> None:
    service.delete(document_id)
