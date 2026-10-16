"""Base interface for semantic caching in SpecRAG."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
from app.api.schemas import RetrievedCandidateSchema


class BaseSemanticCache(ABC):
    """Abstract interface for caching query results with semantic similarity."""

    @abstractmethod
    def get(
        self, query: str, query_vector: Sequence[float]
    ) -> Optional[List[RetrievedCandidateSchema]]:
        """Retrieve cached candidates if exact match or semantic similarity >= threshold."""
        pass

    @abstractmethod
    def set(
        self,
        query: str,
        query_vector: Sequence[float],
        results: List[RetrievedCandidateSchema],
    ) -> None:
        """Store candidates for query and vector."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached entries."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of active cached entries."""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics (hits, misses, hit_rate, etc.)."""
        pass
