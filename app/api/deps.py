"""Dependency injection providers for FastAPI routes."""

from typing import Optional
from app.config import Settings, get_settings
from app.core.engine import SpecRAGEngine

# Global singleton engine instance initialized during application startup
_global_engine: Optional[SpecRAGEngine] = None


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
