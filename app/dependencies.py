from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.services.documents import DocumentService


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service


SessionDep = Annotated[Session, Depends(get_session)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
