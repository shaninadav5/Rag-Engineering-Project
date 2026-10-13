import io
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.database import create_db_engine, create_session_factory, init_db
from app.db.models import Chunk, Document, DocumentStatus, FileType
from app.errors import EmbeddingUnavailableError
from app.services.documents import INTERRUPTED_REASON, DocumentService
from app.services.vector_store import ChromaVectorStore, VectorRecord
from tests.fakes import FakeEmbedder

# Ten paragraphs of 12 words; with chunk_size=30 and batch size 2 this needs several batches.
LONG_TEXT = "\n\n".join(
    " ".join(f"p{paragraph}w{word}" for word in range(12)) + "." for paragraph in range(10)
)


class ScriptedEmbedder(FakeEmbedder):
    """Runs `on_call(n)` before the n-th embed_documents call (1-based)."""

    def __init__(self, on_call: Callable[[int], None]) -> None:
        super().__init__()
        self.calls = 0
        self._on_call = on_call

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self._on_call(self.calls)
        return super().embed_documents(texts)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    settings = Settings(
        _env_file=None, data_dir=tmp_path, chunk_size=30, chunk_overlap=5, embedding_batch_size=2
    )
    settings.uploads_dir.mkdir(parents=True)
    return settings


@pytest.fixture
def session_factory(settings: Settings) -> Iterator[sessionmaker[Session]]:
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def vector_store(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(settings.chroma_dir)


def make_service(
    settings: Settings,
    session_factory: sessionmaker[Session],
    vector_store: ChromaVectorStore,
    embedder: FakeEmbedder,
) -> DocumentService:
    return DocumentService(settings, session_factory, embedder, vector_store)


def chunk_count(session_factory: sessionmaker[Session], document_id: str) -> int:
    with session_factory() as session:
        return len(session.scalars(select(Chunk).where(Chunk.document_id == document_id)).all())


def test_delete_during_processing_stops_cleanly_and_leaves_nothing(
    settings: Settings, session_factory: sessionmaker[Session], vector_store: ChromaVectorStore
) -> None:
    def delete_before_second_batch(call: int) -> None:
        if call == 2:
            assert vector_store.count(document.id) > 0  # the first batch was already stored
            service.delete(document.id)

    embedder = ScriptedEmbedder(delete_before_second_batch)
    service = make_service(settings, session_factory, vector_store, embedder)
    document = service.create(io.BytesIO(LONG_TEXT.encode()), "long.txt", None)

    service.process(document.id)

    assert embedder.calls == 2
    with session_factory() as session:
        assert session.get(Document, document.id) is None
    assert chunk_count(session_factory, document.id) == 0
    assert vector_store.count(document.id) == 0
    assert list(settings.uploads_dir.iterdir()) == []


class PausingVectorStore(ChromaVectorStore):
    """Calls `on_first_add` in the middle of the first batch write (after the SQLite flush)."""

    def __init__(self, path: Path, on_first_add: Callable[[], None]) -> None:
        super().__init__(path)
        self._on_first_add: Callable[[], None] | None = on_first_add

    def add(self, records: list[VectorRecord]) -> None:
        if self._on_first_add is not None:
            callback, self._on_first_add = self._on_first_add, None
            callback()
        super().add(records)


def test_delete_racing_a_batch_write_waits_for_it_and_leaves_nothing(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    deleters: list[threading.Thread] = []

    def delete_from_another_thread() -> None:
        deleter = threading.Thread(target=service.delete, args=(document.id,))
        deleter.start()
        # Give the delete time to run; with the lock it waits until this batch is committed.
        deleter.join(timeout=0.3)
        deleters.append(deleter)

    store = PausingVectorStore(settings.chroma_dir, delete_from_another_thread)
    service = make_service(settings, session_factory, store, FakeEmbedder())
    document = service.create(io.BytesIO(LONG_TEXT.encode()), "long.txt", None)

    service.process(document.id)
    deleters[0].join()

    with session_factory() as session:
        assert session.get(Document, document.id) is None
    assert chunk_count(session_factory, document.id) == 0
    assert store.count(document.id) == 0


def test_failure_after_some_batches_removes_the_partial_content(
    settings: Settings, session_factory: sessionmaker[Session], vector_store: ChromaVectorStore
) -> None:
    def fail_on_third_batch(call: int) -> None:
        if call == 3:
            raise EmbeddingUnavailableError("Embedding model crashed.")

    service = make_service(
        settings, session_factory, vector_store, ScriptedEmbedder(fail_on_third_batch)
    )
    document = service.create(io.BytesIO(LONG_TEXT.encode()), "long.txt", None)

    service.process(document.id)

    with session_factory() as session:
        stored = session.get(Document, document.id)
        assert stored is not None
        assert (stored.status, stored.error) == (DocumentStatus.FAILED, "Embedding model crashed.")
    assert chunk_count(session_factory, document.id) == 0
    assert vector_store.count(document.id) == 0


def test_processing_a_deleted_document_does_nothing(
    settings: Settings, session_factory: sessionmaker[Session], vector_store: ChromaVectorStore
) -> None:
    embedder = ScriptedEmbedder(lambda _call: None)
    service = make_service(settings, session_factory, vector_store, embedder)

    service.process("no-such-id")

    assert embedder.calls == 0
    assert vector_store.count() == 0


def test_fail_interrupted_only_touches_pending_and_processing_documents(
    settings: Settings, session_factory: sessionmaker[Session], vector_store: ChromaVectorStore
) -> None:
    statuses = {
        "pending": DocumentStatus.PENDING,
        "processing": DocumentStatus.PROCESSING,
        "completed": DocumentStatus.COMPLETED,
    }
    embedding = FakeEmbedder().embed_documents(["Some text."])[0]
    with session_factory() as session:
        for document_id, status in statuses.items():
            session.add(
                Document(
                    id=document_id,
                    name=document_id,
                    filename=f"{document_id}.txt",
                    file_type=FileType.TXT,
                    size_bytes=10,
                    status=status,
                    chunk_count=1 if status is DocumentStatus.COMPLETED else 0,
                )
            )
        for document_id in ("processing", "completed"):
            chunk_id = f"{document_id}-chunk"
            session.add(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=0,
                    text="Some text.",
                    token_count=2,
                    embedding_id=chunk_id,
                )
            )
            vector_store.add([VectorRecord(chunk_id, embedding, document_id, 0, None, None)])
        session.commit()

    make_service(settings, session_factory, vector_store, FakeEmbedder()).fail_interrupted()

    with session_factory() as session:
        for document_id in ("pending", "processing"):
            document = session.get(Document, document_id)
            assert document is not None
            assert (document.status, document.error) == (DocumentStatus.FAILED, INTERRUPTED_REASON)
            assert document.processed_at is not None
        completed = session.get(Document, "completed")
        assert completed is not None
        assert (completed.status, completed.error) == (DocumentStatus.COMPLETED, None)
    assert chunk_count(session_factory, "processing") == 0
    assert vector_store.count("processing") == 0
    assert chunk_count(session_factory, "completed") == 1
    assert vector_store.count("completed") == 1
