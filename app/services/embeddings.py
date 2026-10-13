from typing import Protocol

from app.config import Settings
from app.errors import ConfigurationError, EmbeddingUnavailableError


class Embedder(Protocol):
    dimension: int
    max_input_tokens: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int, query_prefix: str = "") -> None:
        # Imported here rather than at module level: it loads PyTorch, which takes seconds,
        # and tests run with a fake embedder.
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise ConfigurationError(
                f"Could not load embedding model '{model_name}': {exc}"
            ) from exc

        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ConfigurationError(f"Model '{model_name}' does not report its embedding size.")
        self.dimension: int = dimension
        # max_seq_length includes the [CLS] and [SEP] tokens added around every input.
        self.max_input_tokens: int = self._model.max_seq_length - 2
        self._tokenizer = self._model.tokenizer
        self._batch_size = batch_size
        self._query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingUnavailableError(
                "The embedding model failed to process the text."
            ) from exc
        return vectors.tolist()


def check_embedder_compatible(embedder: Embedder, settings: Settings) -> None:
    if embedder.dimension != settings.embedding_dim:
        raise ConfigurationError(
            f"The embedding model produces {embedder.dimension}-dimensional vectors, "
            f"but RAG_EMBEDDING_DIM is {settings.embedding_dim}."
        )
    if settings.chunk_size > embedder.max_input_tokens:
        raise ConfigurationError(
            f"RAG_CHUNK_SIZE={settings.chunk_size} exceeds the embedding model's input limit of "
            f"{embedder.max_input_tokens} tokens; longer chunks would be silently truncated."
        )
