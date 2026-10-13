import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.database import create_db_engine, create_session_factory, init_db
from app.db.models import Document, DocumentStatus
from app.errors import DocumentNotFoundError, DocumentNotReadyError, EmbeddingUnavailableError
from app.services.documents import DocumentService
from app.services.embeddings import Embedder, SentenceTransformerEmbedder
from app.services.retrieval import Retriever
from app.services.vector_store import ChromaVectorStore, VectorRecord
from tests.fakes import FakeEmbedder
from tests.pdf_factory import write_pdf

# With chunk_size=12 and no overlap, every paragraph below becomes its own chunk.
REPORT = (
    "Q3 revenue grew 12 percent year over year.\n\n"
    "The new office opened in Berlin in March.\n\n"
    "Employee headcount reached 450 people."
)
HANDBOOK = "Vacation policy allows 25 days per year.\n\nRemote work is allowed two days a week."


class SpyEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return super().embed_query(text)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        chunk_size=12,
        chunk_overlap=0,
        similarity_threshold=0.3,
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


@pytest.fixture
def embedder() -> SpyEmbedder:
    return SpyEmbedder()


@pytest.fixture
def service(
    settings: Settings,
    session_factory: sessionmaker[Session],
    vector_store: ChromaVectorStore,
    embedder: SpyEmbedder,
) -> DocumentService:
    return DocumentService(settings, session_factory, embedder, vector_store)


@pytest.fixture
def retriever(
    settings: Settings,
    session_factory: sessionmaker[Session],
    vector_store: ChromaVectorStore,
    embedder: SpyEmbedder,
) -> Retriever:
    return Retriever(settings, session_factory, embedder, vector_store)


def make_retriever(
    settings: Settings,
    session_factory: sessionmaker[Session],
    vector_store: ChromaVectorStore,
    embedder: Embedder,
    threshold: float,
) -> Retriever:
    settings = settings.model_copy(update={"similarity_threshold": threshold})
    return Retriever(settings, session_factory, embedder, vector_store)


def ingest(service: DocumentService, text: str, filename: str, name: str | None = None) -> str:
    document = service.create(io.BytesIO(text.encode()), filename, name)
    service.process(document.id)
    return document.id


def set_status(
    session_factory: sessionmaker[Session], document_id: str, status: DocumentStatus
) -> None:
    with session_factory() as session:
        document = session.get(Document, document_id)
        assert document is not None
        document.status = status
        session.commit()


# --- Ranking, limits, threshold ------------------------------------------------------


def test_passage_with_the_known_fact_ranks_first(
    service: DocumentService, retriever: Retriever
) -> None:
    report_id = ingest(service, REPORT, "report.txt")
    ingest(service, HANDBOOK, "handbook.md")

    results = retriever.retrieve("Q3 revenue grew percent", top_k=5)

    best = results[0]
    assert best.text == "Q3 revenue grew 12 percent year over year."
    assert (best.document_id, best.document_name) == (report_id, "report.txt")
    assert (best.page_start, best.page_end) == (None, None)
    assert best.score > 0.5
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_pdf_results_carry_display_name_and_pages(
    service: DocumentService, retriever: Retriever, tmp_path: Path
) -> None:
    pdf = write_pdf(
        tmp_path / "q3.pdf",
        ["Revenue grew 12 percent in the third quarter.", "The new Berlin office opened in March."],
    )
    document = service.create(io.BytesIO(pdf.read_bytes()), "q3.pdf", "Q3 Report")
    service.process(document.id)

    [best, *_] = retriever.retrieve("Berlin office opened March", top_k=5)

    assert best.document_name == "Q3 Report"
    assert (best.page_start, best.page_end) == (2, 2)


def test_returns_at_most_top_k(
    service: DocumentService,
    settings: Settings,
    session_factory: sessionmaker[Session],
    vector_store: ChromaVectorStore,
    embedder: SpyEmbedder,
) -> None:
    ingest(service, REPORT, "report.txt")
    keep_everything = make_retriever(settings, session_factory, vector_store, embedder, -1.0)

    assert len(keep_everything.retrieve("revenue office headcount", top_k=2)) == 2


def test_off_topic_question_returns_nothing(service: DocumentService, retriever: Retriever) -> None:
    ingest(service, REPORT, "report.txt")

    assert retriever.retrieve("Kubernetes cluster autoscaling configuration", top_k=5) == []


def test_threshold_drops_weak_matches_and_keeps_strong_ones(
    service: DocumentService,
    retriever: Retriever,
    settings: Settings,
    session_factory: sessionmaker[Session],
    vector_store: ChromaVectorStore,
    embedder: SpyEmbedder,
) -> None:
    ingest(service, REPORT, "report.txt")
    question = "Q3 revenue grew percent office"
    keep_everything = make_retriever(settings, session_factory, vector_store, embedder, -1.0)

    unfiltered = [r.text for r in keep_everything.retrieve(question, top_k=5)]
    filtered = [r.text for r in retriever.retrieve(question, top_k=5)]

    assert "The new office opened in Berlin in March." in unfiltered
    assert filtered == ["Q3 revenue grew 12 percent year over year."]


# --- Scope: document filter and readiness ----------------------------------------


def test_filter_restricts_results_to_requested_documents(
    service: DocumentService,
    settings: Settings,
    session_factory: sessionmaker[Session],
    vector_store: ChromaVectorStore,
    embedder: SpyEmbedder,
) -> None:
    ingest(service, REPORT, "report.txt")
    handbook_id = ingest(service, HANDBOOK, "handbook.md")
    keep_everything = make_retriever(settings, session_factory, vector_store, embedder, -1.0)

    results = keep_everything.retrieve(
        "revenue per year days", top_k=10, document_ids=[handbook_id, handbook_id]
    )

    assert len(results) == 2
    assert {r.document_id for r in results} == {handbook_id}


def test_unknown_document_ids_are_reported(service: DocumentService, retriever: Retriever) -> None:
    report_id = ingest(service, REPORT, "report.txt")

    with pytest.raises(DocumentNotFoundError) as error:
        retriever.retrieve("revenue", top_k=5, document_ids=["nope-1", report_id, "nope-2"])

    assert error.value.message == "Unknown document id(s): nope-1, nope-2"
    assert error.value.details == {"missing_ids": ["nope-1", "nope-2"]}


@pytest.mark.parametrize(
    "status", [DocumentStatus.PENDING, DocumentStatus.PROCESSING, DocumentStatus.FAILED]
)
def test_documents_that_are_not_completed_are_rejected(
    service: DocumentService,
    retriever: Retriever,
    session_factory: sessionmaker[Session],
    status: DocumentStatus,
) -> None:
    report_id = ingest(service, REPORT, "report.txt")
    set_status(session_factory, report_id, status)

    with pytest.raises(DocumentNotReadyError) as error:
        retriever.retrieve("revenue", top_k=5, document_ids=[report_id])

    assert error.value.message == f"Document(s) not ready for questions: {report_id} ({status})"
    assert error.value.details == {"not_ready": {report_id: str(status)}}


def test_unknown_ids_take_precedence_over_not_ready(
    service: DocumentService, retriever: Retriever, session_factory: sessionmaker[Session]
) -> None:
    report_id = ingest(service, REPORT, "report.txt")
    set_status(session_factory, report_id, DocumentStatus.PROCESSING)

    with pytest.raises(DocumentNotFoundError):
        retriever.retrieve("revenue", top_k=5, document_ids=[report_id, "nope"])


def test_without_filter_documents_still_processing_are_hidden(
    service: DocumentService,
    settings: Settings,
    session_factory: sessionmaker[Session],
    vector_store: ChromaVectorStore,
    embedder: SpyEmbedder,
) -> None:
    report_id = ingest(service, REPORT, "report.txt")
    partial_id = ingest(service, "Q3 revenue grew percent again.", "partial.txt")
    set_status(session_factory, partial_id, DocumentStatus.PROCESSING)
    keep_everything = make_retriever(settings, session_factory, vector_store, embedder, -1.0)

    results = keep_everything.retrieve("Q3 revenue grew percent", top_k=20)

    assert {r.document_id for r in results} == {report_id}


def test_nothing_processed_yet_returns_nothing_without_embedding_the_question(
    service: DocumentService, retriever: Retriever, embedder: SpyEmbedder
) -> None:
    pending = service.create(io.BytesIO(REPORT.encode()), "report.txt", None)
    assert pending.status is DocumentStatus.PENDING

    assert retriever.retrieve("Q3 revenue grew percent", top_k=5) == []
    assert embedder.queries == []


# --- Robustness ---------------------------------------------------------------------


def test_vectors_without_a_chunk_row_are_skipped(
    service: DocumentService,
    retriever: Retriever,
    vector_store: ChromaVectorStore,
    embedder: SpyEmbedder,
) -> None:
    report_id = ingest(service, REPORT, "report.txt")
    ghost = embedder.embed_documents(["Q3 revenue grew percent"])[0]
    vector_store.add([VectorRecord("ghost", ghost, report_id, 99, None, None)])

    results = retriever.retrieve("Q3 revenue grew percent", top_k=5)

    assert "ghost" not in {r.chunk_id for r in results}
    assert results[0].text == "Q3 revenue grew 12 percent year over year."


def test_question_embedding_failure_is_raised(
    service: DocumentService,
    settings: Settings,
    session_factory: sessionmaker[Session],
    vector_store: ChromaVectorStore,
) -> None:
    ingest(service, REPORT, "report.txt")
    failing = Retriever(settings, session_factory, FakeEmbedder(fail=True), vector_store)

    with pytest.raises(EmbeddingUnavailableError):
        failing.retrieve("Q3 revenue", top_k=5)


@pytest.mark.model
def test_real_model_ranks_the_passage_with_the_fact_first(
    tmp_path: Path, real_embedder: SentenceTransformerEmbedder
) -> None:
    # chunk_size=25 keeps each of the three paragraphs in its own chunk.
    settings = Settings(_env_file=None, data_dir=tmp_path, chunk_size=25, chunk_overlap=0)
    settings.uploads_dir.mkdir(parents=True)
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    session_factory = create_session_factory(engine)
    store = ChromaVectorStore(settings.chroma_dir)
    service = DocumentService(settings, session_factory, real_embedder, store)
    text = (
        "Third-quarter revenue increased by 12% compared with the same period last year, "
        "driven mainly by cloud subscriptions.\n\n"
        "The company opened a new engineering office in Berlin and hired 40 engineers.\n\n"
        "Employees receive 25 days of paid vacation per year, plus public holidays."
    )
    ingest(service, text, "report.txt")

    results = Retriever(settings, session_factory, real_embedder, store).retrieve(
        "How much did revenue grow in Q3?", top_k=3
    )
    engine.dispose()

    assert results[0].text.startswith("Third-quarter revenue increased by 12%")
