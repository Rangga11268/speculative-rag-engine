"""Configuration module for SpecRAG using Pydantic Settings."""

from functools import lru_cache
import logging
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

INSECURE_SECRETS = {
    "admin-secret",
    "admin",
    "secret",
    "default",
    "changeme",
    "password",
    "postgres",
    "123456",
    "test",
    "secret-key",
}


class Settings(BaseSettings):
    """Application configuration with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General Application Settings
    APP_NAME: str = "SpecRAG"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "test", "production"] = "development"

    # Cryptographic Authentication & RBAC Secrets
    ADMIN_API_KEY: Optional[str] = None
    SECRET_KEY: Optional[str] = None
    # If True, permits unauthenticated requests defaulting to 'default' namespace in development only
    ALLOW_UNAUTHENTICATED_DEV: bool = True

    # Cross-Origin Resource Sharing (CORS) Configuration
    CORS_ALLOW_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list[str] = [
        "Authorization",
        "Content-Type",
        "X-API-Key",
        "X-Admin-Key",
        "X-Tenant-ID",
    ]

    # Storage Repository Settings
    STORAGE_BACKEND: Literal["sqlite", "postgres"] = "sqlite"
    SQLITE_DB_PATH: str = "specrag.db"
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "postgres"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_FALLBACK_TO_SQLITE: bool = True

    # Vector Embedding Dimensions
    VECTOR_DIMENSION: int = 384

    # Retrieval & BM25 Settings
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RRF_K: int = 60
    RRF_WEIGHT_DENSE: float = 1.0
    RRF_WEIGHT_BM25: float = 1.0
    DEFAULT_TOP_K: int = 10

    # Dynamic Confidence Gating Settings (Milestone 2)
    GATING_TAU: float = 0.25
    GATING_THETA_GAP: float = 0.15
    GATING_THETA_ENTROPY: float = 0.45
    GATING_THETA_MIN: float = 0.30
    GATING_TOP_K: int = 10

    # Semantic Cache Settings (Milestone 3)
    REDIS_ENABLED: bool = False
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    CACHE_MAX_ENTRIES: int = 2000
    CACHE_TTL_SECONDS: int = 3600

    def validate_production_security(self) -> None:
        """Enforce strict cryptographic secret standards, database credentials, and storage policies in production."""
        if self.ENVIRONMENT == "production":
            # 1. Cryptographic Authentication & RBAC Secrets
            if not self.ADMIN_API_KEY or self.ADMIN_API_KEY.strip().lower() in INSECURE_SECRETS or len(self.ADMIN_API_KEY.strip()) < 32:
                raise ValueError(
                    "CRITICAL SECURITY: Insecure or insufficiently strong ADMIN_API_KEY in production. "
                    "Configure a cryptographically random secret with at least 32 characters (256-bit entropy)."
                )
            if not self.SECRET_KEY or self.SECRET_KEY.strip().lower() in INSECURE_SECRETS or len(self.SECRET_KEY.strip()) < 32:
                raise ValueError(
                    "CRITICAL SECURITY: Insecure or insufficiently strong SECRET_KEY in production. "
                    "Configure a cryptographically random secret with at least 32 characters (256-bit entropy)."
                )

            # 2. CORS Production Policy
            if "*" in self.CORS_ALLOW_ORIGINS:
                raise ValueError(
                    "CRITICAL SECURITY: Wildcard '*' CORS origin is forbidden in production. "
                    "Specify exact allowed frontend domains."
                )
            insecure_origins = {"localhost", "127.0.0.1", "0.0.0.0"}
            for origin in self.CORS_ALLOW_ORIGINS:
                cleaned = origin.strip().lower()
                if any(bad in cleaned for bad in insecure_origins):
                    raise ValueError(
                        f"CRITICAL SECURITY: Insecure development origin '{origin}' in production CORS_ALLOW_ORIGINS. "
                        "Specify authorized production domain names only."
                    )

            # 3. Production Database Hardening & Fallback Policy
            if self.STORAGE_BACKEND == "postgres":
                if not self.POSTGRES_PASSWORD or self.POSTGRES_PASSWORD.strip().lower() in INSECURE_SECRETS:
                    raise ValueError(
                        "CRITICAL SECURITY: Insecure or default POSTGRES_PASSWORD in production environment. "
                        "A cryptographically secure database password must be explicitly configured."
                    )
                if self.POSTGRES_USER.strip().lower() in {"postgres", "root", "admin"}:
                    logger.warning(
                        "SECURITY WARNING: Production database is configured with default superuser role '%s'. "
                        "A dedicated least-privilege application user is strongly recommended.",
                        self.POSTGRES_USER,
                    )
                if self.POSTGRES_FALLBACK_TO_SQLITE:
                    raise ValueError(
                        "CRITICAL ARCHITECTURE: POSTGRES_FALLBACK_TO_SQLITE is strictly forbidden in production. "
                        "Silent failover from PostgreSQL to local SQLite causes split-brain state, data loss, "
                        "and multi-tenant isolation discrepancies. Set POSTGRES_FALLBACK_TO_SQLITE=False."
                    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton application settings."""
    settings = Settings()
    return settings
