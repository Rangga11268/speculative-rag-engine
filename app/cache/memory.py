"""High-performance in-memory semantic LRU cache with vector similarity search."""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from app.api.schemas import RetrievedCandidateSchema
from app.cache.base import BaseSemanticCache


@dataclass
class CacheEntry:
    """Represents a cached query result item."""
    query: str
    clean_query: str
    vector: np.ndarray
    results: List[RetrievedCandidateSchema]
    created_at: float
    last_accessed: float


class MemorySemanticCache(BaseSemanticCache):
    """In-memory thread-safe LRU semantic cache using cosine similarity."""

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 2000,
        ttl_seconds: int = 3600,
    ):
        self.similarity_threshold = float(similarity_threshold)
        self.max_entries = int(max_entries)
        self.ttl_seconds = int(ttl_seconds)

        self._lock = threading.RLock()
        self._entries: List[CacheEntry] = []
        self._exact_map: Dict[str, CacheEntry] = {}

        # Telemetry metrics
        self._hits: int = 0
        self._exact_hits: int = 0
        self._semantic_hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def _normalize_vector(self, vec: Sequence[float]) -> np.ndarray:
        """Ensure float32 1D numpy array with unit L2 norm."""
        arr = np.asarray(vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(arr))
        if norm > 1e-12:
            arr = arr / norm
        return arr

    def _cleanup_expired_locked(self, now: float) -> None:
        """Purge expired cache entries (internal lock must be held)."""
        if self.ttl_seconds <= 0:
            return

        alive_entries: List[CacheEntry] = []
        for entry in self._entries:
            if (now - entry.created_at) < self.ttl_seconds:
                alive_entries.append(entry)
            else:
                self._exact_map.pop(entry.clean_query, None)

        self._entries = alive_entries

    def get(
        self, query: str, query_vector: Sequence[float]
    ) -> Optional[List[RetrievedCandidateSchema]]:
        """Lookup cached candidates using exact match first, then vector cosine similarity."""
        with self._lock:
            now = time.time()
            self._cleanup_expired_locked(now)

            clean_q = query.strip().lower()

            # 1. Exact string match (O(1) sub-millisecond shortcut)
            if clean_q in self._exact_map:
                entry = self._exact_map[clean_q]
                if self.ttl_seconds <= 0 or (now - entry.created_at) < self.ttl_seconds:
                    entry.last_accessed = now
                    self._hits += 1
                    self._exact_hits += 1
                    return [c.model_copy() for c in entry.results]
                else:
                    self._exact_map.pop(clean_q, None)
                    if entry in self._entries:
                        self._entries.remove(entry)

            # 2. Semantic vector similarity search
            if not self._entries:
                self._misses += 1
                return None

            q_vec = self._normalize_vector(query_vector)
            cached_matrix = np.vstack([e.vector for e in self._entries])
            similarities = np.dot(cached_matrix, q_vec)

            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= self.similarity_threshold:
                matched_entry = self._entries[best_idx]
                matched_entry.last_accessed = now
                self._hits += 1
                self._semantic_hits += 1
                return [c.model_copy() for c in matched_entry.results]

            # Cache miss
            self._misses += 1
            return None

    def set(
        self,
        query: str,
        query_vector: Sequence[float],
        results: List[RetrievedCandidateSchema],
    ) -> None:
        """Store candidates in semantic cache with LRU eviction."""
        with self._lock:
            now = time.time()
            self._cleanup_expired_locked(now)

            clean_q = query.strip().lower()
            q_vec = self._normalize_vector(query_vector)

            # Update existing exact entry if present
            if clean_q in self._exact_map:
                entry = self._exact_map[clean_q]
                entry.vector = q_vec
                entry.results = [c.model_copy() for c in results]
                entry.created_at = now
                entry.last_accessed = now
                return

            # Check LRU capacity constraint
            if len(self._entries) >= self.max_entries:
                # Evict the entry with the oldest last_accessed timestamp
                oldest_entry = min(self._entries, key=lambda e: e.last_accessed)
                self._entries.remove(oldest_entry)
                self._exact_map.pop(oldest_entry.clean_query, None)
                self._evictions += 1

            new_entry = CacheEntry(
                query=query,
                clean_query=clean_q,
                vector=q_vec,
                results=[c.model_copy() for c in results],
                created_at=now,
                last_accessed=now,
            )
            self._entries.append(new_entry)
            self._exact_map[clean_q] = new_entry

    def clear(self) -> None:
        """Clear all cache state and metrics."""
        with self._lock:
            self._entries.clear()
            self._exact_map.clear()
            self._hits = 0
            self._exact_hits = 0
            self._semantic_hits = 0
            self._misses = 0
            self._evictions = 0

    def size(self) -> int:
        """Return number of active entries."""
        with self._lock:
            return len(self._entries)

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = round(self._hits / total, 4) if total > 0 else 0.0
            return {
                "size": len(self._entries),
                "max_entries": self.max_entries,
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl_seconds,
                "total_queries": total,
                "hits": self._hits,
                "exact_hits": self._exact_hits,
                "semantic_hits": self._semantic_hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
            }
