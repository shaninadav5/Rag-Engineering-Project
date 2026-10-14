from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.db.models import DocumentStatus, FileType


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    filename: str
    chunks: int = Field(validation_alias="chunk_count")
    status: DocumentStatus
    uploaded_at: datetime


class DocumentDetail(DocumentSummary):
    file_type: FileType
    size_bytes: int
    pages: int | None = Field(validation_alias="page_count")
    error: str | None
    processed_at: datetime | None


class AskRequest(BaseModel):
    # Upper limits (question length, top_k) come from settings and are checked in the endpoint.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    document_ids: list[str] | None = Field(default=None, min_length=1)
    top_k: int | None = Field(default=None, ge=1)


class Source(BaseModel):
    document_id: str
    document_name: str
    chunk_text: str
    page: int | None
    page_end: int | None
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    tokens_used: int
