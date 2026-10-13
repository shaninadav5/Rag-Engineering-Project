from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    data_dir: Path = Path("data")
    max_upload_mb: int = Field(20, gt=0)
    max_pages: int = Field(1000, gt=0)

    # Chunk sizes are counted with the embedding model's tokenizer, so a chunk
    # must stay below the model's max sequence length or it gets truncated.
    chunk_size: int = Field(500, gt=0)
    chunk_overlap: int = Field(50, ge=0)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(384, gt=0)
    embedding_batch_size: int = Field(32, gt=0)
    # bge models expect this instruction in front of queries (not documents) for retrieval.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    top_k: int = Field(5, gt=0)
    top_k_max: int = Field(20, gt=0)
    similarity_threshold: float = Field(0.5, ge=-1.0, le=1.0)
    max_question_chars: int = Field(2000, gt=0)

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_timeout_seconds: float = Field(120.0, gt=0)
    llm_temperature: float = Field(0.1, ge=0.0, le=2.0)
    llm_num_ctx: int = Field(8192, gt=0)
    llm_num_predict: int = Field(512, gt=0)
    max_context_tokens: int = Field(3000, gt=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k > self.top_k_max:
            raise ValueError("top_k must not exceed top_k_max")
        if self.max_context_tokens < self.chunk_size:
            raise ValueError(
                "max_context_tokens must be at least chunk_size, "
                "otherwise no source fits in the prompt"
            )
        if self.max_context_tokens + self.llm_num_predict >= self.llm_num_ctx:
            raise ValueError(
                "max_context_tokens + llm_num_predict must be smaller than llm_num_ctx, "
                "otherwise Ollama silently truncates the prompt"
            )
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "rag.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path.as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
