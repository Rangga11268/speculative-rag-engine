"""High-performance in-memory semantic LRU cache with vector similarity search."""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    namespace: str = "default"
    index_version: str = "v1"


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
        self._exact_map: Dict[Tuple[str, str, str], CacheEntry] = {}


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
                self._exact_map.pop((entry.namespace, entry.index_version, entry.clean_query), None)

        self._entries = alive_entries

    def get(
        self,
        query: str,
        query_vector: Sequence[float],
        namespace: str = "default",
        index_version: str = "v1",
    ) -> Optional[List[RetrievedCandidateSchema]]:
        """Lookup cached candidates using exact match first, then vector cosine similarity."""
        with self._lock:
            now = time.time()
            self._cleanup_expired_locked(now)

            clean_q = query.strip().lower()
            map_key = (namespace, index_version, clean_q)

            # 1. Exact string match (O(1) sub-millisecond shortcut)
            if map_key in self._exact_map:
                entry = self._exact_map[map_key]
                if self.ttl_seconds <= 0 or (now - entry.created_at) < self.ttl_seconds:
                    entry.last_accessed = now
                    self._hits += 1
                    self._exact_hits += 1
                    return [c.model_copy() for c in entry.results]
                else:
                    self._exact_map.pop(map_key, None)
                    if entry in self._entries:
                        self._entries.remove(entry)

            # 2. Semantic vector similarity search within target namespace and index version
            matching_entries = [
                e for e in self._entries
                if e.namespace == namespace and e.index_version == index_version
            ]
            if not matching_entries:
                self._misses += 1
                return None

            q_vec = self._normalize_vector(query_vector)
            cached_matrix = np.vstack([e.vector for e in matching_entries])
            similarities = np.dot(cached_matrix, q_vec)

            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= self.similarity_threshold:
                matched_entry = matching_entries[best_idx]
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
        namespace: str = "default",
        index_version: str = "v1",
    ) -> None:
        """Store candidates in semantic cache with LRU eviction."""
        with self._lock:
            now = time.time()
            self._cleanup_expired_locked(now)

            clean_q = query.strip().lower()
            map_key = (namespace, index_version, clean_q)
            q_vec = self._normalize_vector(query_vector)

            # Update existing exact entry if present
            if map_key in self._exact_map:
                entry = self._exact_map[map_key]
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
                self._exact_map.pop(
                    (oldest_entry.namespace, oldest_entry.index_version, oldest_entry.clean_query),
                    None,
                )
                self._evictions += 1

            new_entry = CacheEntry(
                query=query,
                clean_query=clean_q,
                vector=q_vec,
                results=[c.model_copy() for c in results],
                created_at=now,
                last_accessed=now,
                namespace=namespace,
                index_version=index_version,
            )
            self._entries.append(new_entry)
            self._exact_map[map_key] = new_entry

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear cache state, optionally for a specific namespace."""
        with self._lock:
            if namespace is None:
                self._entries.clear()
                self._exact_map.clear()
                self._hits = 0
                self._exact_hits = 0
                self._semantic_hits = 0
                self._misses = 0
                self._evictions = 0
            else:
                self._entries = [e for e in self._entries if e.namespace != namespace]
                self._exact_map = {
                    k: v for k, v in self._exact_map.items() if k[0] != namespace
                }

    def invalidate(self, namespace: str = "default", index_version: Optional[str] = None) -> None:
        """Invalidate entries matching namespace and optionally index version."""
        with self._lock:
            if index_version is None:
                self.clear(namespace=namespace)
            else:
                self._entries = [
                    e for e in self._entries
                    if not (e.namespace == namespace and e.index_version == index_version)
                ]
                self._exact_map = {
                    k: v for k, v in self._exact_map.items()
                    if not (k[0] == namespace and k[1] == index_version)
                }

    def size(self, namespace: Optional[str] = None) -> int:
        """Return number of active entries."""
        with self._lock:
            if namespace is None:
                return len(self._entries)
            return len([e for e in self._entries if e.namespace == namespace])


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
