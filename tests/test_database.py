from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import Settings
from app.db.database import create_db_engine


def test_sqlite_pragmas_are_set_on_every_connection(settings: Settings) -> None:
    engine = create_db_engine(settings.database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
            assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
            assert connection.execute(text("PRAGMA busy_timeout")).scalar() == 5000
    finally:
        engine.dispose()


def test_startup_creates_data_dirs_and_database(client: TestClient, settings: Settings) -> None:
    assert settings.uploads_dir.is_dir()
    assert settings.chroma_dir.is_dir()
    assert settings.database_path.is_file()
