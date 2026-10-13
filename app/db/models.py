import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Dialect,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.db.database import Base


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(enum.StrEnum):
    PDF = "pdf"
    TXT = "txt"
    MD = "md"


class UTCDateTime(TypeDecorator[datetime]):
    """Stores datetimes as naive UTC and returns them timezone-aware.

    SQLite has no timezone type, so a plain DateTime column would read values back naive.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetimes are not allowed; pass a timezone-aware value.")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None


def _enum_type(enum_class: type[enum.StrEnum], name: str) -> Enum:
    # Store the enum *values* ("pending"), not SQLAlchemy's default of member names
    # ("PENDING"), and add a CHECK constraint so the database rejects anything else.
    return Enum(
        enum_class,
        name=name,
        native_enum=False,
        create_constraint=True,
        values_callable=lambda members: [member.value for member in members],
    )


def _new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("size_bytes > 0", name="ck_documents_size_positive"),
        CheckConstraint("chunk_count >= 0", name="ck_documents_chunk_count_non_negative"),
        CheckConstraint(
            "page_count IS NULL OR page_count >= 1", name="ck_documents_page_count_positive"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(255))
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[FileType] = mapped_column(_enum_type(FileType, "file_type"))
    size_bytes: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[DocumentStatus] = mapped_column(
        _enum_type(DocumentStatus, "document_status"),
        default=DocumentStatus.PENDING,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text)
    uploaded_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    # passive_deletes lets the database's ON DELETE CASCADE remove chunks in one
    # statement instead of the ORM loading and deleting each chunk individually.
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
        CheckConstraint("chunk_index >= 0", name="ck_chunks_index_non_negative"),
        CheckConstraint("token_count > 0", name="ck_chunks_token_count_positive"),
        CheckConstraint("length(text) > 0", name="ck_chunks_text_non_empty"),
        # The IS NOT NULL checks are required: a CHECK only rejects a row when the
        # expression is FALSE, and comparisons with NULL evaluate to NULL (unknown).
        CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) "
            "OR (page_start IS NOT NULL AND page_end IS NOT NULL "
            "AND page_start >= 1 AND page_end >= page_start)",
            name="ck_chunks_page_range_valid",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    # No default: the ingestion code must set it explicitly to the vector record id.
    embedding_id: Mapped[str] = mapped_column(String(36), unique=True)

    document: Mapped[Document] = relationship(back_populates="chunks")
