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
        """Enforce strict cryptographic secret standards in production environment."""
        if self.ENVIRONMENT == "production":
            if not self.ADMIN_API_KEY or self.ADMIN_API_KEY.strip().lower() in INSECURE_SECRETS or len(self.ADMIN_API_KEY.strip()) < 16:
                raise ValueError(
                    "CRITICAL SECURITY: Insecure or default ADMIN_API_KEY in production. "
                    "Configure a cryptographically secure key with at least 16 characters."
                )
            if not self.SECRET_KEY or self.SECRET_KEY.strip().lower() in INSECURE_SECRETS or len(self.SECRET_KEY.strip()) < 16:
                raise ValueError(
                    "CRITICAL SECURITY: Insecure or default SECRET_KEY in production. "
                    "Configure a cryptographically secure key with at least 16 characters."
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton application settings."""
    settings = Settings()
    return settings
