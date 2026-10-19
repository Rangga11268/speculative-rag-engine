"""FastAPI router endpoints for SpecRAG."""

import logging
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import TenantContext, get_current_tenant, get_engine
from app.api.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.core.engine import SpecRAGEngine

logger = logging.getLogger(__name__)
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
    tenant: TenantContext = Depends(get_current_tenant),
    engine: SpecRAGEngine = Depends(get_engine),
) -> QueryResponse:
    """Handle speculative search query with verified tenant isolation."""
    # Enforce tenant isolation: clients cannot spoof namespaces
    if (
        request.namespace
        and request.namespace != "default"
        and request.namespace != tenant.tenant_id
        and not tenant.is_admin
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Tenant '{tenant.tenant_id}' cannot query foreign namespace '{request.namespace}'.",
        )
    request.namespace = (
        tenant.tenant_id
        if not tenant.is_admin
        else (request.namespace if request.namespace != "default" else tenant.tenant_id)
    )

    try:
        return await engine.query(request)
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Validation error during query execution: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid query request parameters.",
        )
    except PermissionError as e:
        logger.warning("Permission denied during query execution: %s", e)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to requested resource.",
        )
    except Exception:
        logger.exception("Internal error during query execution.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred.",
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
    tenant: TenantContext = Depends(get_current_tenant),
    engine: SpecRAGEngine = Depends(get_engine),
) -> IngestResponse:
    """Ingest document chunks into storage and indices with verified tenant isolation."""
    # Enforce tenant authorization: clients cannot ingest into other tenants' corpora
    for chunk in request.chunks:
        if chunk.namespace and chunk.namespace != tenant.tenant_id and chunk.namespace != "default" and not tenant.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Tenant '{tenant.tenant_id}' cannot ingest into foreign namespace '{chunk.namespace}'.",
            )
        chunk.namespace = tenant.tenant_id

    try:
        chunks = [c.to_chunk() for c in request.chunks]
        count = engine.ingest_chunks(chunks)
        return IngestResponse(
            count=count,
            message=f"Successfully ingested {count} chunk(s).",
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Validation error during chunk ingestion: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chunk ingestion payload.",
        )
    except Exception:
        logger.exception("Internal error during chunk ingestion.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during ingestion.",
        )


@router.get(
    "/cache/stats",
    response_model=Dict[str, Any],
    summary="Get semantic cache statistics",
)
async def get_cache_stats(
    tenant: TenantContext = Depends(get_current_tenant),
    engine: SpecRAGEngine = Depends(get_engine),
) -> Dict[str, Any]:
    """Retrieve cache diagnostic statistics (scoped strictly to tenant unless admin)."""
    if tenant.is_admin:
        return engine.cache.stats()

    # Non-admin tenant ONLY receives metrics scoped to their own tenant/namespace
    # Zero cross-tenant data leakage: do NOT return global hits, misses, or other tenants' sizes
    return {
        "tenant_id": tenant.tenant_id,
        "tenant_entries": engine.cache.size(namespace=tenant.tenant_id),
        "backend": "redis" if getattr(engine.cache, "redis_client", None) else "memory",
        "status": "active",
    }


@router.delete(
    "/cache",
    summary="Clear semantic cache",
)
async def clear_cache(
    tenant: TenantContext = Depends(get_current_tenant),
    engine: SpecRAGEngine = Depends(get_engine),
) -> Dict[str, str]:
    """Purge semantic cache entries (scoped to tenant unless admin)."""
    if tenant.is_admin:
        engine.cache.clear()
        return {"status": "success", "message": "Semantic cache purged globally by admin."}
    else:
        engine.cache.clear(namespace=tenant.tenant_id)
        return {
            "status": "success",
            "message": f"Semantic cache purged for tenant '{tenant.tenant_id}'.",
        }


@router.get(
    "/gating/config",
    response_model=Dict[str, Any],
    summary="Get dynamic gating parameters",
)
async def get_gating_config(
    tenant: TenantContext = Depends(get_current_tenant),
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
