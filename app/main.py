"""Main entrypoint for SpecRAG FastAPI application."""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import get_engine, set_engine
from app.api.routes import router as api_router
from app.config import get_settings
from app.core.engine import SpecRAGEngine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager to initialize and cleanup engine resources."""
    settings = get_settings()
    engine = SpecRAGEngine(settings=settings)
    set_engine(engine)
    yield
    set_engine(None)


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI instance."""
    settings = get_settings()

    application = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description=(
            "SpecRAG: Low-Latency Entropy-Gated Speculative Retrieval-Augmented "
            "Generation Engine running natively on Windows."
        ),
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API versioned routes
    application.include_router(api_router, prefix=settings.API_PREFIX)

    @application.get("/health", tags=["System"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint indicating service status and cache metrics."""
        engine = get_engine()
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": "0.1.0",
            "storage_backend": settings.STORAGE_BACKEND,
            "cache_entries": engine.cache.size() if engine else 0,
        }

    @application.get("/", tags=["System"])
    async def root() -> Dict[str, str]:
        """Root landing endpoint with system status."""
        return {
            "message": f"Welcome to {settings.APP_NAME} Speculative Retrieval Engine",
            "docs": "/docs",
            "health": "/health",
        }

    return application


app = create_app()
