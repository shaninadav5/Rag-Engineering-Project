from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.db.database import create_db_engine, create_session_factory, init_db
from app.errors import register_exception_handlers
from app.services.embeddings import Embedder, SentenceTransformerEmbedder, check_embedder_compatible
from app.services.vector_store import ChromaVectorStore


def create_app(settings: Settings | None = None, embedder: Embedder | None = None) -> FastAPI:
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

        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        app.state.embedder = active_embedder
        app.state.vector_store = ChromaVectorStore(settings.chroma_dir)
        yield
        engine.dispose()

    app = FastAPI(title="RAG Document Q&A", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
