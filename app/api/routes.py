"""FastAPI router endpoints for SpecRAG."""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_engine
from app.api.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.core.engine import SpecRAGEngine

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Execute speculative retrieval query",
    description=(
        "Executes end-to-end speculative query pipeline with semantic caching, "
        "hybrid retrieval (Dense + BM25 with RRF), and dynamic confidence gating."
    ),
)
async def query_pipeline(
    request: QueryRequest,
    engine: SpecRAGEngine = Depends(get_engine),
) -> QueryResponse:
    """Handle speculative search query."""
    try:
        return await engine.query(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest document chunks",
    description="Persist chunks to storage, vector store, and BM25 index.",
)
async def ingest_documents(
    request: IngestRequest,
    engine: SpecRAGEngine = Depends(get_engine),
) -> IngestResponse:
    """Ingest document chunks into storage and indices."""
    try:
        chunks = [c.to_chunk() for c in request.chunks]
        count = engine.ingest_chunks(chunks)
        return IngestResponse(
            count=count,
            message=f"Successfully ingested {count} chunk(s).",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.get(
    "/cache/stats",
    response_model=Dict[str, Any],
    summary="Get semantic cache statistics",
)
async def get_cache_stats(
    engine: SpecRAGEngine = Depends(get_engine),
) -> Dict[str, Any]:
    """Retrieve cache diagnostic statistics (hits, misses, hit rate, size)."""
    return engine.cache.stats()


@router.delete(
    "/cache",
    summary="Clear semantic cache",
)
async def clear_cache(
    engine: SpecRAGEngine = Depends(get_engine),
) -> Dict[str, str]:
    """Purge all entries from the semantic cache."""
    engine.cache.clear()
    return {"status": "success", "message": "Semantic cache purged."}


@router.get(
    "/gating/config",
    response_model=Dict[str, Any],
    summary="Get dynamic gating parameters",
)
async def get_gating_config(
    engine: SpecRAGEngine = Depends(get_engine),
) -> Dict[str, Any]:
    """Retrieve active entropy gate configuration."""
    return {
        "tau": engine.gating.tau,
        "theta_gap": engine.gating.theta_gap,
        "theta_entropy": engine.gating.theta_entropy,
        "theta_min": engine.gating.theta_min,
        "top_k": engine.gating.top_k,
    }
