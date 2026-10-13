from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.database import create_db_engine, create_session_factory, init_db
from app.main import create_app
from tests.fakes import FakeEmbedder


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def client(settings: Settings, embedder: FakeEmbedder) -> Iterator[TestClient]:
    # The `with` block runs the app's lifespan (data dirs, database, embedder); without it
    # startup never happens.
    with TestClient(create_app(settings, embedder=embedder)) as test_client:
        yield test_client


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
