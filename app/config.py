from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    data_dir: Path = Path("data")
    max_upload_mb: int = 20

    # Chunk sizes are counted with the embedding model's tokenizer, so a chunk
    # must stay below the model's max sequence length or it gets truncated.
    chunk_size: int = 500
    chunk_overlap: int = 50

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 32

    top_k: int = 5
    similarity_threshold: float = 0.5

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_timeout_seconds: float = 120.0
    max_context_tokens: int = 3000

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'rag.db'}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
