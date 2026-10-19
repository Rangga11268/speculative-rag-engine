"""Dependency injection providers for FastAPI routes with cryptographic authentication."""

from dataclasses import dataclass
from typing import Optional
from fastapi import Header, HTTPException, status
from app.config import Settings, get_settings
from app.core.engine import SpecRAGEngine
from app.core.security import verify_admin_key, verify_tenant_token

# Global singleton engine instance initialized during application startup
_global_engine: Optional[SpecRAGEngine] = None


@dataclass(frozen=True)
class TenantContext:
    """Security context holding authenticated tenant identity and permissions."""
    tenant_id: str
    is_admin: bool = False


def get_current_tenant(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> TenantContext:
    """Validate tenant identity and administrative credentials from cryptographic tokens.

    Authentication hierarchy:
    1. Admin Token / Key: If matches verified ADMIN_API_KEY, grants full administrative access.
       Admin callers can optionally specify 'X-Tenant-ID' to scope operations to a specific tenant.
    2. Signed Tenant Token: Cryptographically verifies HMAC signature in Bearer token or API key.
       Extracts tenant_id securely. Disallows header identity spoofing if X-Tenant-ID does not match.
    3. Missing / Invalid Credentials:
       - In production: Raises 401 Unauthorized immediately.
       - In development/test (if ALLOW_UNAUTHENTICATED_DEV is enabled): Falls back to dev tenant.
    """
    settings = get_settings()

    # 1. Resolve incoming candidate token from headers
    token: Optional[str] = None
    if authorization and authorization.strip():
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        else:
            token = authorization.strip()
    elif x_api_key and x_api_key.strip():
        token = x_api_key.strip()
    elif x_admin_key and x_admin_key.strip():
        token = x_admin_key.strip()

    # 2. Check Administrative Credentials
    if token and verify_admin_key(token):
        target_tenant = x_tenant_id.strip() if (x_tenant_id and x_tenant_id.strip()) else "admin"
        return TenantContext(tenant_id=target_tenant, is_admin=True)

    # 3. Check Signed Tenant Credentials
    if token:
        verified_tenant = verify_tenant_token(token)
        if verified_tenant is not None:
            # Token is cryptographically authentic
            # Ensure client is not attempting to spoof a different tenant via header
            if x_tenant_id and x_tenant_id.strip() and x_tenant_id.strip() != verified_tenant:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        f"Tenant identity mismatch: Authenticated token is issued for '{verified_tenant}', "
                        f"cannot assume '{x_tenant_id.strip()}'."
                    ),
                )
            return TenantContext(tenant_id=verified_tenant, is_admin=False)

        # Token was provided but failed verification
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token or cryptographic signature verification failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Handle Missing Credentials
    if settings.ENVIRONMENT != "production" and settings.ALLOW_UNAUTHENTICATED_DEV:
        dev_tenant = x_tenant_id.strip() if (x_tenant_id and x_tenant_id.strip()) else "default"
        return TenantContext(tenant_id=dev_tenant, is_admin=False)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide a valid Bearer token or API key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


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
