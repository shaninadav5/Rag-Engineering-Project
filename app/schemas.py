from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

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
