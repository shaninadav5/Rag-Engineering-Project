import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import ConfigurationError
from app.main import create_app
from app.services.embeddings import SentenceTransformerEmbedder, check_embedder_compatible
from tests.fakes import FakeEmbedder


def test_compatible_embedder_passes(settings: Settings) -> None:
    check_embedder_compatible(FakeEmbedder(dimension=384, max_input_tokens=510), settings)


def test_dimension_mismatch_is_rejected(settings: Settings) -> None:
    with pytest.raises(ConfigurationError, match="768-dimensional.*RAG_EMBEDDING_DIM is 384"):
        check_embedder_compatible(FakeEmbedder(dimension=768), settings)


def test_chunk_size_above_model_input_limit_is_rejected(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, chunk_size=600)

    with pytest.raises(ConfigurationError, match="silently truncated"):
        check_embedder_compatible(FakeEmbedder(max_input_tokens=510), settings)


def test_app_refuses_to_start_with_incompatible_embedder(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, embedding_dim=999)
    app = create_app(settings, embedder=FakeEmbedder(dimension=384))

    with pytest.raises(ConfigurationError), TestClient(app):
        pass


class FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer and records how it is used."""

    loaded: list["FakeSentenceTransformer"] = []

    max_seq_length = 512

    def __init__(self, model_name: str) -> None:
        self.encode_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.tokenizer = types.SimpleNamespace(encode=lambda text, **kwargs: text.split())
        FakeSentenceTransformer.loaded.append(self)

    def get_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls.append((texts, kwargs))
        return np.full((len(texts), 384), 1 / math.sqrt(384))


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    FakeSentenceTransformer.loaded = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeSentenceTransformer


def test_model_is_loaded_once_and_every_encode_requests_normalized_vectors(
    fake_sentence_transformers: type[FakeSentenceTransformer],
) -> None:
    embedder = SentenceTransformerEmbedder("some-model", batch_size=8, query_prefix="query: ")

    for _ in range(20):
        embedder.embed_documents(["Revenue grew.", "Costs fell."])
        embedder.embed_query("How did revenue change?")
        embedder.count_tokens("Revenue grew.")

    [model] = fake_sentence_transformers.loaded
    assert len(model.encode_calls) == 40
    assert all(kwargs["normalize_embeddings"] is True for _, kwargs in model.encode_calls)
    assert model.encode_calls[1][0] == ["query: How did revenue change?"]


def test_app_loads_the_model_once_at_startup_and_keeps_it(
    fake_sentence_transformers: type[FakeSentenceTransformer], settings: Settings
) -> None:
    app = create_app(settings)

    with TestClient(app) as client:
        for _ in range(5):
            client.get("/health")

        [model] = fake_sentence_transformers.loaded
        assert app.state.embedder._model is model


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    defaults = Settings(_env_file=None)
    return SentenceTransformerEmbedder(
        defaults.embedding_model, defaults.embedding_batch_size, defaults.embedding_query_prefix
    )


@pytest.mark.model
def test_real_model_matches_configuration(
    real_embedder: SentenceTransformerEmbedder, settings: Settings
) -> None:
    assert real_embedder.dimension == 384
    assert real_embedder.max_input_tokens == 510
    check_embedder_compatible(real_embedder, settings)


@pytest.mark.model
def test_real_model_vectors_are_normalized_and_rank_relevant_text_first(
    real_embedder: SentenceTransformerEmbedder,
) -> None:
    relevant, unrelated = real_embedder.embed_documents(
        ["Revenue grew 12% in the third quarter.", "The office cafeteria serves lunch at noon."]
    )
    query = real_embedder.embed_query("How much did revenue grow?")

    for vector in (relevant, unrelated, query):
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0, abs=1e-4)

    def similarity(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert similarity(query, relevant) > similarity(query, unrelated)


@pytest.mark.model
def test_real_model_counts_tokens_without_special_tokens(
    real_embedder: SentenceTransformerEmbedder,
) -> None:
    assert real_embedder.count_tokens("hello world") == 2
    assert real_embedder.embed_documents([]) == []


@pytest.mark.model
@pytest.mark.parametrize("text", ["", " ", "  \n\t \r\n  ", "\x00\x07\u200b"])
def test_real_tokenizer_counts_blank_text_as_zero(
    real_embedder: SentenceTransformerEmbedder, text: str
) -> None:
    assert real_embedder.count_tokens(text) == 0


@pytest.mark.model
def test_real_tokenizer_counts_text_longer_than_model_limit_without_truncating(
    real_embedder: SentenceTransformerEmbedder,
) -> None:
    long_text = "revenue " * 5000

    assert real_embedder.count_tokens(long_text) == 5000
    [vector] = real_embedder.embed_documents([long_text])
    assert len(vector) == 384


@pytest.mark.model
def test_unknown_model_fails_with_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="does-not-exist"):
        SentenceTransformerEmbedder("rag-tests/does-not-exist-model", batch_size=8)
