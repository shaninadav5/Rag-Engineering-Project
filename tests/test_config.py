from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults_are_valid() -> None:
    settings = make_settings()

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.embedding_dim == 384
    assert settings.max_upload_bytes == 20 * 1024 * 1024


def test_paths_are_derived_from_data_dir(tmp_path: Path) -> None:
    settings = make_settings(data_dir=tmp_path)

    assert settings.uploads_dir == tmp_path / "uploads"
    assert settings.chroma_dir == tmp_path / "chroma"
    assert settings.database_url == f"sqlite:///{(tmp_path / 'rag.db').as_posix()}"


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"chunk_size": 100, "chunk_overlap": 100}, id="overlap-not-smaller"),
        pytest.param({"chunk_size": 0}, id="chunk-size-zero"),
        pytest.param({"top_k": 25}, id="top-k-above-max"),
        pytest.param({"max_context_tokens": 8000}, id="context-exceeds-num-ctx"),
        pytest.param({"max_context_tokens": 400}, id="context-smaller-than-one-chunk"),
        pytest.param({"similarity_threshold": 1.5}, id="threshold-out-of-range"),
    ],
)
def test_invalid_settings_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)
