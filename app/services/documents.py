import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import Chunk, Document, DocumentStatus, FileType, utcnow
from app.errors import AppError, DocumentNotFoundError, FileTooLargeError
from app.services.chunking import ChunkData, chunk_pages
from app.services.embeddings import Embedder
from app.services.extraction import detect_file_type, extract, validate
from app.services.vector_store import ChromaVectorStore, VectorRecord

logger = logging.getLogger(__name__)

INTERRUPTED_REASON = "Processing was interrupted by a server restart."


class DocumentService:
    """Creates, processes, and deletes documents across SQLite, Chroma, and the uploads folder.

    Batch writes and deletes share one lock, so a delete never interleaves with a write and
    cannot leave orphan chunks or vectors behind. The slow steps (extraction, embedding) run
    outside the lock. The lock only covers one process, which is why the app runs one worker.
    """

    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        embedder: Embedder,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._embedder = embedder
        self._vector_store = vector_store
        self._lock = threading.Lock()

    def create(self, source: BinaryIO, filename: str, name: str | None) -> Document:
        """Stores the upload and runs the lightweight checks; processing is scheduled separately."""
        file_type = detect_file_type(filename)
        document_id = str(uuid.uuid4())
        path = self._file_path(document_id, file_type)
        try:
            with path.open("wb") as target:
                shutil.copyfileobj(source, target)
            size = path.stat().st_size
            if size > self._settings.max_upload_bytes:
                raise FileTooLargeError.for_limit(self._settings.max_upload_mb)
            validation = validate(path, file_type, self._settings.max_pages)
            document = Document(
                id=document_id,
                name=name or filename[:255],
                filename=filename,
                file_type=file_type,
                size_bytes=size,
                page_count=validation.page_count,
            )
            with self._session_factory() as session:
                session.add(document)
                session.commit()
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return document

    def process(self, document_id: str) -> None:
        """Background task: extract, chunk, embed, and store, then mark completed or failed."""
        with self._lock, self._session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                return
            document.status = DocumentStatus.PROCESSING
            session.commit()
            path = self._file_path(document.id, document.file_type)
            file_type = document.file_type

        try:
            pages = extract(path, file_type)
            chunks = chunk_pages(
                pages,
                self._embedder.count_tokens,
                self._settings.chunk_size,
                self._settings.chunk_overlap,
            )
            batch_size = self._settings.embedding_batch_size
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                embeddings = self._embedder.embed_documents([chunk.text for chunk in batch])
                if not self._store_batch(document_id, batch, embeddings):
                    return
            self._mark_completed(document_id, len(chunks))
        except Exception as exc:
            # A document deleted mid-processing (e.g. its file vanished) is a clean stop, not
            # an error worth a traceback.
            if self._mark_failed(document_id, _failure_reason(exc)) and not isinstance(
                exc, AppError
            ):
                logger.exception("Unexpected error while processing document %s", document_id)

    def delete(self, document_id: str) -> None:
        with self._lock, self._session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise DocumentNotFoundError.for_id(document_id)
            path = self._file_path(document.id, document.file_type)
            # Vectors first: a crash midway leaves a document without vectors, never vectors
            # without a document.
            self._vector_store.delete_document(document_id)
            session.delete(document)
            session.commit()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete stored file %s", path, exc_info=True)

    def fail_interrupted(self) -> None:
        """At startup: documents left pending or processing by a restart become failed."""
        with self._lock, self._session_factory() as session:
            interrupted = session.scalars(
                select(Document).where(
                    Document.status.in_([DocumentStatus.PENDING, DocumentStatus.PROCESSING])
                )
            ).all()
            for document in interrupted:
                self._remove_content(session, document.id)
                document.status = DocumentStatus.FAILED
                document.error = INTERRUPTED_REASON
                document.processed_at = utcnow()
            session.commit()

    def _store_batch(
        self, document_id: str, batch: list[ChunkData], embeddings: list[list[float]]
    ) -> bool:
        """Writes one batch of chunks and vectors; False if the document was deleted meanwhile."""
        with self._lock, self._session_factory() as session:
            if session.get(Document, document_id) is None:
                return False
            rows = [_chunk_row(document_id, chunk) for chunk in batch]
            session.add_all(rows)
            session.flush()
            self._vector_store.add(
                [
                    VectorRecord(
                        row.embedding_id,
                        embedding,
                        document_id,
                        row.chunk_index,
                        row.page_start,
                        row.page_end,
                    )
                    for row, embedding in zip(rows, embeddings, strict=True)
                ]
            )
            session.commit()
        return True

    def _mark_completed(self, document_id: str, chunk_count: int) -> None:
        with self._lock, self._session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                return
            document.status = DocumentStatus.COMPLETED
            document.chunk_count = chunk_count
            document.processed_at = utcnow()
            session.commit()

    def _mark_failed(self, document_id: str, reason: str) -> bool:
        """Removes partial chunks and vectors; False if the document no longer exists."""
        with self._lock, self._session_factory() as session:
            document = session.get(Document, document_id)
            if document is None:
                return False
            self._remove_content(session, document_id)
            document.status = DocumentStatus.FAILED
            document.error = reason
            document.chunk_count = 0
            document.processed_at = utcnow()
            session.commit()
        return True

    def _remove_content(self, session: Session, document_id: str) -> None:
        self._vector_store.delete_document(document_id)
        session.execute(delete(Chunk).where(Chunk.document_id == document_id))

    def _file_path(self, document_id: str, file_type: FileType) -> Path:
        return self._settings.uploads_dir / f"{document_id}.{file_type}"


def _chunk_row(document_id: str, chunk: ChunkData) -> Chunk:
    chunk_id = str(uuid.uuid4())
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        token_count=chunk.token_count,
        embedding_id=chunk_id,
    )


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, AppError):
        return exc.message
    return f"Processing failed due to an unexpected error ({type(exc).__name__})."
