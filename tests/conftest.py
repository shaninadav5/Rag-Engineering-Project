from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.database import create_db_engine, create_session_factory, init_db
from app.main import create_app
from app.services.embeddings import Embedder, SentenceTransformerEmbedder
from app.services.llm import LLM
from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(scope="session")
def real_embedder() -> SentenceTransformerEmbedder:
    """The real bge model, loaded once per test run; only for tests marked `model`."""
    defaults = Settings(_env_file=None)
    return SentenceTransformerEmbedder(
        defaults.embedding_model, defaults.embedding_batch_size, defaults.embedding_query_prefix
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings: Settings, embedder: FakeEmbedder, llm: FakeLLM) -> Iterator[TestClient]:
    # The `with` block runs the app's lifespan (data dirs, database, embedder); without it
    # startup never happens.
    with TestClient(create_app(settings, embedder=embedder, llm=llm)) as test_client:
        yield test_client


@pytest.fixture
def make_client(tmp_path: Path) -> Iterator[Callable[..., TestClient]]:
    """Builds a client with setting overrides, for tests that need non-default limits."""
    clients: list[TestClient] = []

    def build(
        embedder: Embedder | None = None, llm: LLM | None = None, **overrides: Any
    ) -> TestClient:
        settings = Settings(_env_file=None, data_dir=tmp_path, **overrides)
        app = create_app(settings, embedder=embedder or FakeEmbedder(), llm=llm or FakeLLM())
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def db_session(settings: Settings) -> Iterator[Session]:
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    session = create_session_factory(engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
