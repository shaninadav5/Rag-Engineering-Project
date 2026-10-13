from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import Chunk, Document, DocumentStatus
from app.errors import DocumentNotFoundError, DocumentNotReadyError
from app.services.embeddings import Embedder
from app.services.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_start: int | None
    page_end: int | None
    score: float


class Retriever:
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

    def retrieve(
        self, question: str, top_k: int, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Most relevant chunks above the similarity threshold, best first; may be empty."""
        with self._session_factory() as session:
            scope = _searchable_ids(session, document_ids)
        if not scope:
            return []

        embedding = self._embedder.embed_query(question)
        hits = self._vector_store.search(embedding, top_k, scope)
        scores = {
            hit.id: hit.score for hit in hits if hit.score >= self._settings.similarity_threshold
        }
        if not scores:
            return []

        with self._session_factory() as session:
            rows = session.execute(
                select(Chunk, Document.name)
                .join(Document)
                .where(Chunk.embedding_id.in_(list(scores)))
            ).all()
        # A hit without a row belongs to a document deleted after the search; it is skipped.
        chunks = [_to_retrieved(chunk, name, scores[chunk.embedding_id]) for chunk, name in rows]
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)


def _searchable_ids(session: Session, document_ids: list[str] | None) -> list[str]:
    if document_ids is None:
        return list(
            session.scalars(select(Document.id).where(Document.status == DocumentStatus.COMPLETED))
        )

    requested = list(dict.fromkeys(document_ids))
    rows = session.execute(select(Document.id, Document.status).where(Document.id.in_(requested)))
    statuses = {document_id: status for document_id, status in rows}
    missing = [document_id for document_id in requested if document_id not in statuses]
    if missing:
        raise DocumentNotFoundError(
            f"Unknown document id(s): {', '.join(missing)}", {"missing_ids": missing}
        )
    not_ready = {
        document_id: str(statuses[document_id])
        for document_id in requested
        if statuses[document_id] is not DocumentStatus.COMPLETED
    }
    if not_ready:
        listed = ", ".join(f"{document_id} ({status})" for document_id, status in not_ready.items())
        raise DocumentNotReadyError(
            f"Document(s) not ready for questions: {listed}", {"not_ready": not_ready}
        )
    return requested


def _to_retrieved(chunk: Chunk, document_name: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_name=document_name,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        score=score,
    )
