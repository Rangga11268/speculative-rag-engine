"""Configuration module for SpecRAG using Pydantic Settings."""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton application settings."""
    return Settings()
