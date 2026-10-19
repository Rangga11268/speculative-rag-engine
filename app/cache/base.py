"""Base interface for semantic caching in SpecRAG."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
from app.api.schemas import RetrievedCandidateSchema


class BaseSemanticCache(ABC):
    """Abstract interface for caching query results with semantic similarity."""

    @abstractmethod
    def get(
        self,
        query: str,
        query_vector: Sequence[float],
        namespace: str = "default",
        index_version: str = "v1",
    ) -> Optional[List[RetrievedCandidateSchema]]:
        """Retrieve cached candidates if exact match or semantic similarity >= threshold."""
        pass

    @abstractmethod
    def set(
        self,
        query: str,
        query_vector: Sequence[float],
        results: List[RetrievedCandidateSchema],
        namespace: str = "default",
        index_version: str = "v1",
    ) -> None:
        """Store candidates for query and vector."""
        pass

    @abstractmethod
    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear cached entries, optionally restricted to a specific namespace."""
        pass

    @abstractmethod
    def invalidate(self, namespace: str = "default", index_version: Optional[str] = None) -> None:
        """Invalidate entries for a specific namespace and/or index version."""
        pass

    @abstractmethod
    def size(self, namespace: Optional[str] = None) -> int:
        """Return the number of active cached entries."""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return cache performance statistics (hits, misses, hit_rate, etc.)."""
        pass

