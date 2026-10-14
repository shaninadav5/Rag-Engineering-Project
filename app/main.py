from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import ask, documents
from app.config import Settings, get_settings
from app.db.database import create_db_engine, create_session_factory, init_db
from app.errors import register_exception_handlers
from app.middleware import BodySizeLimitMiddleware
from app.services.documents import DocumentService
from app.services.embeddings import Embedder, SentenceTransformerEmbedder, check_embedder_compatible
from app.services.llm import LLM, OllamaClient
from app.services.qa import QAService
from app.services.retrieval import Retriever
from app.services.vector_store import ChromaVectorStore


def create_app(
    settings: Settings | None = None, embedder: Embedder | None = None, llm: LLM | None = None
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_embedder = embedder
        if active_embedder is None:
            active_embedder = SentenceTransformerEmbedder(
                settings.embedding_model,
                settings.embedding_batch_size,
                settings.embedding_query_prefix,
            )
        check_embedder_compatible(active_embedder, settings)

        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)

        engine = create_db_engine(settings.database_url)
        init_db(engine)
        session_factory = create_session_factory(engine)
        vector_store = ChromaVectorStore(settings.chroma_dir)
        document_service = DocumentService(settings, session_factory, active_embedder, vector_store)
        document_service.fail_interrupted()
        retriever = Retriever(settings, session_factory, active_embedder, vector_store)
        # Ollama is not contacted at startup: uploads keep working while it is down.
        ollama_client: OllamaClient | None = None
        active_llm: LLM
        if llm is None:
            ollama_client = OllamaClient(settings)
            active_llm = ollama_client
        else:
            active_llm = llm

        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.embedder = active_embedder
        app.state.vector_store = vector_store
        app.state.document_service = document_service
        app.state.retriever = retriever
        app.state.llm = active_llm
        app.state.qa_service = QAService(
            settings, retriever, active_llm, active_embedder.count_tokens
        )
        yield
        if ollama_client is not None:
            ollama_client.close()
        engine.dispose()

    app = FastAPI(title="RAG Document Q&A", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)
    app.add_middleware(BodySizeLimitMiddleware, max_upload_mb=settings.max_upload_mb)
    app.include_router(documents.router)
    app.include_router(ask.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
