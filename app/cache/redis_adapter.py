"""Redis-backed semantic cache with automatic in-memory fallback."""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from app.api.schemas import RetrievedCandidateSchema
from app.cache.base import BaseSemanticCache
from app.cache.memory import MemorySemanticCache

logger = logging.getLogger(__name__)


class RedisSemanticCache(BaseSemanticCache):
    """Semantic cache with Redis persistence and automatic MemorySemanticCache fallback."""

    def __init__(
        self,
        redis_url: str = "redis://127.0.0.1:6379/0",
        similarity_threshold: float = 0.92,
        max_entries: int = 2000,
        ttl_seconds: int = 3600,
    ):
        self.redis_url = redis_url
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

        # In-memory workhorse / fallback
        self._memory_cache = MemorySemanticCache(
            similarity_threshold=similarity_threshold,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
        )

        self._redis_client = None
        self._is_redis_connected = False
        self._init_redis()

    def _init_redis(self) -> None:
        """Attempt connection to Redis instance."""
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
            client.ping()
            self._redis_client = client
            self._is_redis_connected = True
            logger.info("Connected to Redis semantic cache at %s", self.redis_url)
        except Exception as e:
            self._redis_client = None
            self._is_redis_connected = False
            logger.info(
                "Redis unavailable (%s); operating seamlessly with in-memory semantic cache.",
                str(e),
            )

    @property
    def is_connected(self) -> bool:
        """Return whether Redis client is connected."""
        return self._is_redis_connected

    def get(
        self, query: str, query_vector: Sequence[float]
    ) -> Optional[List[RetrievedCandidateSchema]]:
        """Retrieve cached candidates.

        First checks Redis for exact query match; delegates vector similarity
        to the in-memory cache layer.
        """
        # Exact lookup in Redis if connected
        if self._is_redis_connected and self._redis_client:
            try:
                key = f"specrag:cache:exact:{query.strip().lower()}"
                val = self._redis_client.get(key)
                if val:
                    raw_list = json.loads(val)
                    return [RetrievedCandidateSchema(**item) for item in raw_list]
            except Exception as e:
                logger.warning("Redis get error, falling back: %s", str(e))

        return self._memory_cache.get(query, query_vector)

    def set(
        self,
        query: str,
        query_vector: Sequence[float],
        results: List[RetrievedCandidateSchema],
    ) -> None:
        """Cache candidates in both in-memory store and Redis."""
        self._memory_cache.set(query, query_vector, results)

        if self._is_redis_connected and self._redis_client:
            try:
                key = f"specrag:cache:exact:{query.strip().lower()}"
                data = json.dumps([r.model_dump() for r in results])
                if self.ttl_seconds > 0:
                    self._redis_client.setex(key, self.ttl_seconds, data)
                else:
                    self._redis_client.set(key, data)
            except Exception as e:
                logger.warning("Redis set error: %s", str(e))

    def clear(self) -> None:
        """Clear cache state."""
        self._memory_cache.clear()
        if self._is_redis_connected and self._redis_client:
            try:
                for k in self._redis_client.scan_iter("specrag:cache:exact:*"):
                    self._redis_client.delete(k)
            except Exception as e:
                logger.warning("Redis clear error: %s", str(e))

    def size(self) -> int:
        """Return entry count."""
        return self._memory_cache.size()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics including backend indicator."""
        st = self._memory_cache.stats()
        st["redis_connected"] = self._is_redis_connected
        st["backend"] = "redis+memory" if self._is_redis_connected else "in_memory"
        return st
