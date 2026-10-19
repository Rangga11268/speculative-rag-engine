"""Redis-backed semantic cache with automatic in-memory fallback."""

import json
import logging
import time
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
        self,
        query: str,
        query_vector: Sequence[float],
        namespace: str = "default",
        index_version: str = "v1",
    ) -> Optional[List[RetrievedCandidateSchema]]:
        """Retrieve cached candidates.

        First checks Redis for exact query match; delegates vector similarity
        to the in-memory cache layer.
        """
        # Exact lookup in Redis if connected
        if self._is_redis_connected and self._redis_client:
            try:
                clean_q = query.strip().lower()
                key = f"specrag:{namespace}:{index_version}:exact:{clean_q}"
                val = self._redis_client.get(key)
                if val:
                    # Refresh accessed timestamp in zset for LRU tracking
                    zset_key = f"specrag:{namespace}:{index_version}:zkeys"
                    self._redis_client.zadd(zset_key, {key: time.time()}, xx=True)
                    raw_list = json.loads(val)
                    return [RetrievedCandidateSchema(**item) for item in raw_list]
            except Exception as e:
                logger.warning("Redis get error, falling back: %s", str(e))

        return self._memory_cache.get(
            query, query_vector, namespace=namespace, index_version=index_version
        )

    def set(
        self,
        query: str,
        query_vector: Sequence[float],
        results: List[RetrievedCandidateSchema],
        namespace: str = "default",
        index_version: str = "v1",
    ) -> None:
        """Cache candidates in both in-memory store and Redis."""
        self._memory_cache.set(
            query, query_vector, results, namespace=namespace, index_version=index_version
        )

        if self._is_redis_connected and self._redis_client:
            try:
                clean_q = query.strip().lower()
                key = f"specrag:{namespace}:{index_version}:exact:{clean_q}"
                zset_key = f"specrag:{namespace}:{index_version}:zkeys"
                data = json.dumps([r.model_dump() for r in results])
                if self.ttl_seconds > 0:
                    self._redis_client.setex(key, self.ttl_seconds, data)
                else:
                    self._redis_client.set(key, data)

                # Track key timestamp for LRU capacity eviction (P2 Redis Bound)
                now_ts = time.time()
                self._redis_client.zadd(zset_key, {key: now_ts})
                if self.ttl_seconds > 0:
                    self._redis_client.expire(zset_key, self.ttl_seconds + 300)

                if self.max_entries > 0:
                    card = self._redis_client.zcard(zset_key)
                    if card > self.max_entries:
                        to_remove = card - self.max_entries
                        oldest = self._redis_client.zpopmin(zset_key, to_remove)
                        for old_k, _ in oldest:
                            self._redis_client.delete(old_k)
            except Exception as e:
                logger.warning("Redis set error: %s", str(e))

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear cache state."""
        self._memory_cache.clear(namespace=namespace)
        if self._is_redis_connected and self._redis_client:
            try:
                pattern = f"specrag:{namespace}:*" if namespace else "specrag:*"
                for k in self._redis_client.scan_iter(pattern):
                    self._redis_client.delete(k)
            except Exception as e:
                logger.warning("Redis clear error: %s", str(e))

    def invalidate(self, namespace: str = "default", index_version: Optional[str] = None) -> None:
        """Invalidate entries matching namespace and optionally index version."""
        self._memory_cache.invalidate(namespace=namespace, index_version=index_version)
        if self._is_redis_connected and self._redis_client:
            try:
                if index_version:
                    pattern = f"specrag:{namespace}:{index_version}:*"
                else:
                    pattern = f"specrag:{namespace}:*"
                for k in self._redis_client.scan_iter(pattern):
                    self._redis_client.delete(k)
            except Exception as e:
                logger.warning("Redis invalidate error: %s", str(e))

    def size(self, namespace: Optional[str] = None) -> int:
        """Return entry count."""
        return self._memory_cache.size(namespace=namespace)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics including backend indicator."""
        st = self._memory_cache.stats()
        st["redis_connected"] = self._is_redis_connected
        st["backend"] = "redis+memory" if self._is_redis_connected else "in_memory"
        return st

