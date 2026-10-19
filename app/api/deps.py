"""Dependency injection providers for FastAPI routes."""

from dataclasses import dataclass
from typing import Optional
from fastapi import Header, HTTPException, status
from app.config import Settings, get_settings
from app.core.engine import SpecRAGEngine

# Global singleton engine instance initialized during application startup
_global_engine: Optional[SpecRAGEngine] = None


@dataclass(frozen=True)
class TenantContext:
    """Security context holding authenticated tenant identity and permissions."""
    tenant_id: str
    is_admin: bool = False


def get_current_tenant(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> TenantContext:
    """Validate tenant identity and administrative credentials from request headers.

    - If X-Admin-Key matches Settings.ADMIN_API_KEY, grants full administrative access.
    - If X-Tenant-ID is supplied, verifies and sanitizes the tenant identifier.
    - If omitted, defaults safely to 'default' tenant context for backward compatibility.
    """
    settings = get_settings()
    is_admin = bool(x_admin_key and x_admin_key.strip() == settings.ADMIN_API_KEY)

    if x_tenant_id is not None:
        tenant_id = x_tenant_id.strip()
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tenant identity: X-Tenant-ID header cannot be empty.",
            )
        return TenantContext(tenant_id=tenant_id, is_admin=is_admin)

    return TenantContext(tenant_id="default", is_admin=is_admin)


def get_engine() -> SpecRAGEngine:
    """Provide singleton instance of SpecRAGEngine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = SpecRAGEngine()
    return _global_engine


def set_engine(engine: Optional[SpecRAGEngine]) -> None:
    """Set or reset singleton engine instance (useful for testing and lifespan)."""
    global _global_engine
    _global_engine = engine
