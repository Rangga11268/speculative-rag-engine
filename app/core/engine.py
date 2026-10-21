"""SpecRAGEngine coordinating storage, hybrid retrieval, gating, and reranking."""

import time
from typing import List, Optional, Sequence
import numpy as np

from app.api.schemas import (
    DocumentChunk,
    GatingTelemetrySchema,
    QueryRequest,
    QueryResponse,
    RetrievedCandidate,
    RetrievedCandidateSchema,
)
from app.cache.base import BaseSemanticCache
from app.cache.memory import MemorySemanticCache
from app.cache.redis_adapter import RedisSemanticCache
from app.config import Settings, get_settings
from app.core.embedder import DenseEmbedder
from app.core.gating import DynamicConfidenceGate, GatingTelemetry, RouteDecision
from app.core.reranker import BaseReranker, NativeRescorer
from app.retrieval.bm25 import OkapiBM25Index
from app.retrieval.hybrid import HybridRetriever
from app.storage.base import StorageRepository
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.vector_store import DenseVectorIndex


class SpecRAGEngine:
    """High-performance pipeline engine for SpecRAG."""

    def __init__(
        self,
        storage_repo: Optional[StorageRepository] = None,
        vector_index: Optional[DenseVectorIndex] = None,
        bm25_index: Optional[OkapiBM25Index] = None,
        embedder: Optional[DenseEmbedder] = None,
        gating: Optional[DynamicConfidenceGate] = None,
        reranker: Optional[BaseReranker] = None,
        cache: Optional[BaseSemanticCache] = None,
        settings: Optional[Settings] = None,
        namespace: str = "default",
    ):
        self.namespace = namespace
        self.index_version: int = 1
        self.settings = settings or get_settings()

        if storage_repo is not None:
            self.storage_repo = storage_repo
        elif self.settings.STORAGE_BACKEND == "postgres":
            from app.storage.postgres import PostgresStorageRepository
            self.storage_repo = PostgresStorageRepository(
                host=self.settings.POSTGRES_HOST,
                port=self.settings.POSTGRES_PORT,
                dbname=self.settings.POSTGRES_DB,
                user=self.settings.POSTGRES_USER,
                password=self.settings.POSTGRES_PASSWORD,
                fallback_to_sqlite=self.settings.POSTGRES_FALLBACK_TO_SQLITE,
                sqlite_fallback_path=self.settings.SQLITE_DB_PATH,
                dimension=self.settings.VECTOR_DIMENSION,
            )
        else:
            self.storage_repo = SQLiteStorageRepository(
                db_path=self.settings.SQLITE_DB_PATH,
                dimension=self.settings.VECTOR_DIMENSION,
            )
        self.vector_index = vector_index or DenseVectorIndex(dimension=self.settings.VECTOR_DIMENSION)
        self.bm25_index = bm25_index or OkapiBM25Index(k1=self.settings.BM25_K1, b=self.settings.BM25_B)
        self.embedder = embedder or DenseEmbedder(dimension=self.settings.VECTOR_DIMENSION)
        self.gating = gating or DynamicConfidenceGate(
            tau=self.settings.GATING_TAU,
            theta_gap=self.settings.GATING_THETA_GAP,
            theta_entropy=self.settings.GATING_THETA_ENTROPY,
            theta_min=self.settings.GATING_THETA_MIN,
            top_k=self.settings.GATING_TOP_K,
        )
        self.reranker = reranker or NativeRescorer()

        # Semantic Cache Layer (Milestone 3)
        if cache is not None:
            self.cache = cache
        elif self.settings.REDIS_ENABLED:
            self.cache = RedisSemanticCache(
                redis_url=self.settings.REDIS_URL,
                similarity_threshold=self.settings.CACHE_SIMILARITY_THRESHOLD,
                max_entries=self.settings.CACHE_MAX_ENTRIES,
                ttl_seconds=self.settings.CACHE_TTL_SECONDS,
            )
        else:
            self.cache = MemorySemanticCache(
                similarity_threshold=self.settings.CACHE_SIMILARITY_THRESHOLD,
                max_entries=self.settings.CACHE_MAX_ENTRIES,
                ttl_seconds=self.settings.CACHE_TTL_SECONDS,
            )

        self.hybrid_retriever = HybridRetriever(
            vector_index=self.vector_index,
            bm25_index=self.bm25_index,
            storage_repo=self.storage_repo,
            default_rrf_k=self.settings.RRF_K,
        )

    def ingest_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Batch ingest chunks into storage, vector index, and BM25 index."""
        if not chunks:
            return 0

        prepared_chunks: List[DocumentChunk] = []
        vec_ids: List[str] = []
        vectors: List[List[float]] = []
        texts: List[str] = []

        for chunk in chunks:
            emb = chunk.embedding
            if emb is None or len(emb) != self.settings.VECTOR_DIMENSION:
                emb = self.embedder.embed_text(chunk.text)
                chunk.embedding = emb

            prepared_chunks.append(chunk)
            vec_ids.append(chunk.id)
            vectors.append(emb)
            texts.append(chunk.text)

        # 1. Persist to storage repository
        count = self.storage_repo.add_chunks(prepared_chunks)

        chunk_namespaces = [c.namespace for c in prepared_chunks]
        try:
            # 2. Add to dense vector index
            self.vector_index.add(vec_ids, vectors, namespaces=chunk_namespaces)

            # 3. Add to BM25 index
            self.bm25_index.add(vec_ids, texts, namespaces=chunk_namespaces)
        except Exception as e:
            # Atomic reconciliation: If vector or BM25 index fails, rebuild indices from storage
            self.rebuild_indices_from_storage()
            raise RuntimeError(f"Ingestion failed during in-memory index updates; indices reconciled from storage: {e}") from e

        # Invalidate cache for all affected namespaces on corpus mutation
        if count > 0:
            self.index_version += 1
            unique_namespaces = set(chunk_namespaces) | {self.namespace}
            for ns in unique_namespaces:
                self.cache.invalidate(namespace=ns)

        return count

    def rebuild_indices_from_storage(self) -> int:
        """Reconcile in-memory dense vector and BM25 indices from persistent storage."""
        all_chunks = self.storage_repo.get_all_chunks()
        if not all_chunks:
            self.vector_index.clear()
            self.bm25_index.clear()
            return 0

        vec_ids = [c.id for c in all_chunks]
        vectors = [
            c.embedding if (c.embedding is not None and len(c.embedding) == self.settings.VECTOR_DIMENSION)
            else self.embedder.embed_text(c.text)
            for c in all_chunks
        ]
        texts = [c.text for c in all_chunks]
        chunk_namespaces = [c.namespace for c in all_chunks]

        self.vector_index.clear()
        self.vector_index.add(vec_ids, vectors, namespaces=chunk_namespaces)

        self.bm25_index.clear()
        self.bm25_index.add(vec_ids, texts, namespaces=chunk_namespaces)

        self.index_version += 1
        unique_namespaces = set(chunk_namespaces) | {self.namespace}
        for ns in unique_namespaces:
            self.cache.invalidate(namespace=ns)
        return len(all_chunks)

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Execute end-to-end asynchronous query pipeline."""
        t0 = time.perf_counter()

        # 1. Generate query embedding
        query_vector = self.embedder.embed_text(request.query)

        # 2. Check Semantic Cache with namespace and server-authoritative index version
        ns = request.namespace if request.namespace else self.namespace
        # Server is single source of truth for active index_version. Ignore or validate client requests.
        idx_ver = str(self.index_version)
        if request.index_version is not None and request.index_version != idx_ver:
            # Client provided a stale or mismatched index_version -> bypass cache lookup to prevent serving stale cache
            use_cache = False
        else:
            use_cache = request.use_cache

        if use_cache:
            cached_candidates = self.cache.get(
                request.query, query_vector, namespace=ns, index_version=idx_ver
            )
            if cached_candidates is not None:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return QueryResponse(
                    query=request.query,
                    results=cached_candidates[: request.top_k],
                    route_taken="CACHE_HIT",
                    cache_hit=True,
                    latency_ms=round(latency_ms, 3),
                    gating_telemetry=None,
                )


        # 3. Hybrid retrieval (Dense + BM25 via RRF) with namespace isolation
        fetch_k = max(request.top_k * 2, self.settings.DEFAULT_TOP_K)
        candidates = await self.hybrid_retriever.retrieve(
            query_text=request.query,
            query_vector=query_vector,
            top_k=fetch_k,
            dense_weight=self.settings.RRF_WEIGHT_DENSE,
            bm25_weight=self.settings.RRF_WEIGHT_BM25,
            rrf_k=self.settings.RRF_K,
            namespace=ns,
        )

        if not candidates:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return QueryResponse(
                query=request.query,
                results=[],
                route_taken="FAST_PATH_BYPASS",
                cache_hit=False,
                latency_ms=round(latency_ms, 3),
                gating_telemetry=None,
            )

        # 4. Dynamic Confidence Gating & Speculative Routing
        if request.force_rerank:
            final_candidates = self.reranker.rerank(
                query=request.query,
                candidates=candidates,
                top_k=request.top_k,
            )
            route_taken = "RERANKED"
            telemetry = GatingTelemetry(
                decision=RouteDecision.RERANK_LOW_GAP,
                entropy=0.0,
                normalized_entropy=0.0,
                marginal_gap=0.0,
                top1_score=candidates[0].score if candidates else 0.0,
                probabilities=[],
                bypass_granted=False,
            )
        else:
            telemetry = self.gating.evaluate(candidates)
            if telemetry.bypass_granted:
                # Fast Path: bypass reranker entirely
                final_candidates = candidates[:request.top_k]
                route_taken = "FAST_PATH_BYPASS"
            else:
                # Deliberate Path: deep cross-encoder reranking
                final_candidates = self.reranker.rerank(
                    query=request.query,
                    candidates=candidates,
                    top_k=request.top_k,
                )
                route_taken = "RERANKED"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        schema_results = [
            RetrievedCandidateSchema(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                score=c.score,
                dense_score=c.dense_score,
                bm25_score=c.bm25_score,
                consensus_count=c.consensus_count,
                metadata=c.metadata,
            )
            for c in final_candidates
        ]

        # 5. Populate cache if requested
        if request.use_cache and schema_results:
            self.cache.set(
                request.query, query_vector, schema_results, namespace=ns, index_version=idx_ver
            )


        gating_schema = GatingTelemetrySchema(
            decision=telemetry.decision.value,
            entropy=telemetry.entropy,
            normalized_entropy=telemetry.normalized_entropy,
            marginal_gap=telemetry.marginal_gap,
            top1_score=telemetry.top1_score,
            probabilities=telemetry.probabilities,
            bypass_granted=telemetry.bypass_granted,
        )

        return QueryResponse(
            query=request.query,
            results=schema_results,
            route_taken=route_taken,
            cache_hit=False,
            latency_ms=round(latency_ms, 3),
            gating_telemetry=gating_schema,
        )

    def query_sync(self, request: QueryRequest) -> QueryResponse:
        """Synchronous wrapper for query pipeline."""
        import asyncio
        return asyncio.run(self.query(request))
