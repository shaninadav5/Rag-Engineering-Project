from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.config import Settings
from app.services.documents import DocumentService
from app.services.qa import QAService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


def get_qa_service(request: Request) -> QAService:
    return request.app.state.qa_service


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[Session, Depends(get_session)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
