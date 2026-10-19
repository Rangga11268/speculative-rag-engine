import asyncio
import concurrent.futures
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import set_engine
from app.api.schemas import (
    DocumentChunk,
    DocumentChunkSchema,
    IngestRequest,
    QueryRequest,
    RetrievedCandidateSchema,
)
from app.cache.memory import MemorySemanticCache
from app.cache.redis_adapter import RedisSemanticCache
from app.config import Settings
from app.core.engine import SpecRAGEngine
from app.core.security import get_active_admin_key
from app.main import app
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.vector_store import DenseVectorIndex


def _mock_candidates():
    return [
        RetrievedCandidateSchema(
            chunk_id=f"c_{i}",
            doc_id="doc_1",
            text=f"Sample text chunk {i}",
            score=0.9 - (i * 0.1),
            dense_score=0.85,
            bm25_score=0.75,
            consensus_count=1,
            metadata={"source": "test"},
        )
        for i in range(3)
    ]


def test_memory_cache_exact_hit():
    """Verify O(1) exact query string match returns identical cached candidates."""
    cache = MemorySemanticCache(similarity_threshold=0.92, max_entries=100, ttl_seconds=300)
    vec = [0.1] * 384
    candidates = _mock_candidates()

    cache.set("What is SpecRAG?", vec, candidates)
    assert cache.size() == 1

    res = cache.get("What is SpecRAG?", vec)
    assert res is not None
    assert len(res) == 3
    assert res[0].chunk_id == "c_0"

    st = cache.stats()
    assert st["hits"] == 1
    assert st["exact_hits"] == 1
    assert st["misses"] == 0
    assert st["hit_rate"] == 1.0


def test_memory_cache_semantic_hit():
    """Verify semantic similarity match when query vector cosine similarity >= 0.92."""
    cache = MemorySemanticCache(similarity_threshold=0.92, max_entries=100, ttl_seconds=300)

    # Base vector
    np.random.seed(42)
    base_vec = np.random.randn(384).astype(np.float32)
    base_vec /= np.linalg.norm(base_vec)

    # Slight perturbation (cosine similarity > 0.95)
    noise = np.random.randn(384).astype(np.float32)
    noise /= np.linalg.norm(noise)
    pert_vec = 0.96 * base_vec + 0.04 * noise
    pert_vec /= np.linalg.norm(pert_vec)

    cos_sim = float(np.dot(base_vec, pert_vec))
    assert cos_sim >= 0.92

    candidates = _mock_candidates()
    cache.set("Original query", base_vec.tolist(), candidates)

    # Lookup with different text but high vector similarity
    res = cache.get("Slightly rephrased query", pert_vec.tolist())
    assert res is not None
    assert len(res) == 3

    st = cache.stats()
    assert st["hits"] == 1
    assert st["semantic_hits"] == 1
    assert st["misses"] == 0


def test_memory_cache_miss_below_threshold():
    """Verify cache miss when vector similarity is strictly below threshold."""
    cache = MemorySemanticCache(similarity_threshold=0.92, max_entries=100, ttl_seconds=300)

    v1 = np.zeros(384, dtype=np.float32)
    v1[0] = 1.0

    v2 = np.zeros(384, dtype=np.float32)
    v2[1] = 1.0  # Orthogonal: cosine similarity = 0.0

    cache.set("Topic A", v1.tolist(), _mock_candidates())
    res = cache.get("Topic B", v2.tolist())

    assert res is None
    st = cache.stats()
    assert st["misses"] == 1
    assert st["hits"] == 0


def test_memory_cache_lru_eviction():
    """Verify least-recently-accessed entry is evicted when capacity is reached."""
    cache = MemorySemanticCache(similarity_threshold=0.92, max_entries=3, ttl_seconds=300)

    for i in range(3):
        v = [0.0] * 384
        v[i] = 1.0
        cache.set(f"query_{i}", v, _mock_candidates())
        time.sleep(0.01)

    assert cache.size() == 3

    # Access query_0 to make it recently used
    v0 = [0.0] * 384
    v0[0] = 1.0
    cache.get(f"query_0", v0)

    # Insert 4th entry; query_1 should be evicted (as query_0 was refreshed)
    v3 = [0.0] * 384
    v3[3] = 1.0
    cache.set("query_3", v3, _mock_candidates())

    assert cache.size() == 3
    st = cache.stats()
    assert st["evictions"] == 1

    # query_1 should be missing
    v1 = [0.0] * 384
    v1[1] = 1.0
    assert cache.get("query_1", v1) is None


def test_memory_cache_ttl_expiration():
    """Verify expired cache entries return None."""
    cache = MemorySemanticCache(similarity_threshold=0.92, max_entries=10, ttl_seconds=1)
    v = [0.1] * 384
    cache.set("Short lived query", v, _mock_candidates())

    assert cache.size() == 1
    time.sleep(1.1)

    res = cache.get("Short lived query", v)
    assert res is None


def test_memory_cache_thread_safety():
    """Verify concurrent reads and writes do not corrupt state."""
    cache = MemorySemanticCache(similarity_threshold=0.92, max_entries=500, ttl_seconds=300)

    def worker(worker_id: int):
        for j in range(25):
            vec = [0.0] * 384
            vec[j % 384] = 1.0
            cache.set(f"thread_{worker_id}_q_{j}", vec, _mock_candidates())
            cache.get(f"thread_{worker_id}_q_{j}", vec)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert cache.size() <= 500
    st = cache.stats()
    assert st["hits"] > 0


def test_redis_adapter_fallback():
    """Verify RedisSemanticCache operates seamlessly in in-memory mode when Redis is offline."""
    cache = RedisSemanticCache(
        redis_url="redis://127.0.0.1:9999/0",  # Unused port
        similarity_threshold=0.92,
        max_entries=100,
        ttl_seconds=300,
    )
    assert not cache.is_connected

    v = [0.1] * 384
    cache.set("Fallback test", v, _mock_candidates())
    res = cache.get("Fallback test", v)
    assert res is not None
    assert len(res) == 3

    st = cache.stats()
    assert st["backend"] == "in_memory"


def test_engine_cache_integration(tmp_path):
    """Verify SpecRAGEngine routes to CACHE_HIT on repeated query."""
    async def _run():
        db_file = str(tmp_path / "test_engine_cache.db")
        storage = SQLiteStorageRepository(db_file)
        vector_idx = DenseVectorIndex(dimension=384)
        engine = SpecRAGEngine(storage_repo=storage, vector_index=vector_idx)

        # Ingest document
        engine.ingest_chunks([
            DocumentChunk(
                id="c_cache_1",
                doc_id="d1",
                text="Quantum computing utilizes quantum superposition and entanglement.",
                embedding=[0.05] * 384,
            )
        ])

        req = QueryRequest(query="Quantum computing superposition", top_k=1, use_cache=True)

        # First query -> Cache Miss (Fast path or rerank)
        resp1 = await engine.query(req)
        assert not resp1.cache_hit
        assert resp1.route_taken in ("FAST_PATH_BYPASS", "RERANKED")

        # Second query -> Cache Hit
        resp2 = await engine.query(req)
        assert resp2.cache_hit
        assert resp2.route_taken == "CACHE_HIT"
        assert resp2.latency_ms < resp1.latency_ms + 10.0  # Typically sub-10ms

    asyncio.run(_run())


def test_fastapi_endpoints(tmp_path):
    """Verify all FastAPI routes via TestClient."""
    db_file = str(tmp_path / "test_api.db")
    storage = SQLiteStorageRepository(db_file)
    vector_idx = DenseVectorIndex(dimension=384)
    engine = SpecRAGEngine(storage_repo=storage, vector_index=vector_idx)
    set_engine(engine)

    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"

    # 2. Ingest endpoint
    chunk_payload = {
        "chunks": [
            {
                "id": "api_chunk_1",
                "doc_id": "doc_api",
                "text": "FastAPI is a modern, high-performance web framework for Python.",
                "metadata": {"category": "tech"},
            }
        ]
    }
    res_ingest = client.post("/api/v1/ingest", json=chunk_payload)
    assert res_ingest.status_code == 201
    assert res_ingest.json()["count"] == 1

    # 3. Query endpoint (Miss first)
    query_payload = {
        "query": "FastAPI web framework",
        "top_k": 1,
        "use_cache": True,
        "force_rerank": False,
    }
    res_query1 = client.post("/api/v1/query", json=query_payload)
    assert res_query1.status_code == 200
    q_data1 = res_query1.json()
    assert q_data1["cache_hit"] is False
    assert len(q_data1["results"]) >= 1

    # 4. Query endpoint (Hit second)
    res_query2 = client.post("/api/v1/query", json=query_payload)
    assert res_query2.status_code == 200
    q_data2 = res_query2.json()
    assert q_data2["cache_hit"] is True
    assert q_data2["route_taken"] == "CACHE_HIT"

    # 5. Cache stats: admin view receives global hits, tenant view receives isolated entries
    res_stats_admin = client.get("/api/v1/cache/stats", headers={"X-Admin-Key": get_active_admin_key()})
    assert res_stats_admin.status_code == 200
    stats_data_admin = res_stats_admin.json()
    assert stats_data_admin["hits"] >= 1

    res_stats_tenant = client.get("/api/v1/cache/stats")
    assert res_stats_tenant.status_code == 200
    stats_data_tenant = res_stats_tenant.json()
    assert stats_data_tenant["tenant_entries"] >= 1
    assert "hits" not in stats_data_tenant

    # 6. Gating config
    res_gating = client.get("/api/v1/gating/config")
    assert res_gating.status_code == 200
    assert "tau" in res_gating.json()

    # 7. Clear cache
    res_clear = client.delete("/api/v1/cache")
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "success"

    # 8. Query again -> should be miss again
    res_query3 = client.post("/api/v1/query", json=query_payload)
    assert res_query3.status_code == 200
    assert res_query3.json()["cache_hit"] is False


def test_cache_namespace_corpus_isolation():
    """Verify identical queries in different tenant/corpus namespaces are strictly isolated."""
    cache = MemorySemanticCache(similarity_threshold=0.92, max_entries=100, ttl_seconds=300)
    vec = [0.1] * 384

    cand_legal = [
        RetrievedCandidateSchema(chunk_id="c_leg", doc_id="doc_leg", text="Legal clause", score=0.9)
    ]
    cand_tech = [
        RetrievedCandidateSchema(chunk_id="c_tech", doc_id="doc_tech", text="Tech manual", score=0.95)
    ]

    cache.set("compliance", vec, cand_legal, namespace="legal_corpus", index_version="v1")
    cache.set("compliance", vec, cand_tech, namespace="tech_corpus", index_version="v1")

    res_leg = cache.get("compliance", vec, namespace="legal_corpus", index_version="v1")
    assert res_leg is not None
    assert res_leg[0].chunk_id == "c_leg"

    res_tech = cache.get("compliance", vec, namespace="tech_corpus", index_version="v1")
    assert res_tech is not None
    assert res_tech[0].chunk_id == "c_tech"

    # Querying a non-existent namespace returns cache miss
    assert cache.get("compliance", vec, namespace="finance_corpus", index_version="v1") is None


def test_cache_index_version_invalidation():
    """Verify cache invalidation when index_version changes."""
    cache = MemorySemanticCache(similarity_threshold=0.92, max_entries=100, ttl_seconds=300)
    vec = [0.1] * 384
    candidates = _mock_candidates()

    cache.set("query", vec, candidates, namespace="default", index_version="v1")
    assert cache.get("query", vec, namespace="default", index_version="v1") is not None

    # Stale version lookup is a miss
    assert cache.get("query", vec, namespace="default", index_version="v2") is None

    # Invalidate v1
    cache.invalidate(namespace="default", index_version="v1")
    assert cache.get("query", vec, namespace="default", index_version="v1") is None


def test_tenant_data_isolation_end_to_end(tmp_path):
    """Verify that Tenant A chunks are NEVER retrieved in Tenant B queries (Dense, BM25, and Cache)."""
    async def _run():
        db_file = str(tmp_path / "test_tenant_isolation.db")
        storage = SQLiteStorageRepository(db_file)
        vector_idx = DenseVectorIndex(dimension=384)
        engine = SpecRAGEngine(storage_repo=storage, vector_index=vector_idx)

        # 1. Ingest confidential document for Tenant A
        chunk_a = DocumentChunk(
            id="tenant_a_doc_1",
            doc_id="secret_a",
            text="Confidential quarterly earnings and financial statements for Tenant Alpha.",
            embedding=[0.05] * 384,
            namespace="tenant_a",
        )
        # 2. Ingest document for Tenant B
        chunk_b = DocumentChunk(
            id="tenant_b_doc_1",
            doc_id="public_b",
            text="General marketing strategies and public roadmap for Tenant Beta.",
            embedding=[-0.05] * 384,
            namespace="tenant_b",
        )
        engine.ingest_chunks([chunk_a, chunk_b])

        # 3. Query as Tenant B asking about financial statements
        req_b = QueryRequest(
            query="financial quarterly statements earnings",
            top_k=5,
            namespace="tenant_b",
            use_cache=True,
        )
        resp_b = await engine.query(req_b)

        # Assert Tenant A document is NEVER leaked to Tenant B
        retrieved_ids_b = [r.chunk_id for r in resp_b.results]
        assert "tenant_a_doc_1" not in retrieved_ids_b

        # 4. Query as Tenant A asking the same query
        req_a = QueryRequest(
            query="financial quarterly statements earnings",
            top_k=5,
            namespace="tenant_a",
            use_cache=True,
        )
        resp_a = await engine.query(req_a)
        retrieved_ids_a = [r.chunk_id for r in resp_a.results]
        assert "tenant_a_doc_1" in retrieved_ids_a

        # 5. Verify caching isolation: subsequent query by Tenant B must still miss Tenant A's cached result
        resp_b_second = await engine.query(req_b)
        assert "tenant_a_doc_1" not in [r.chunk_id for r in resp_b_second.results]

        # 6. Verify client cannot bypass server authority by sending a stale index_version
        stale_req_a = QueryRequest(
            query="financial quarterly statements earnings",
            top_k=5,
            namespace="tenant_a",
            index_version="v0_stale",
            use_cache=True,
        )
        stale_resp_a = await engine.query(stale_req_a)
        assert stale_resp_a.cache_hit is False  # Stale version bypasses cache lookup

        # 7. Ingestion of new chunk in tenant_a invalidates tenant_a cache without polluting tenant_b
        chunk_a_update = DocumentChunk(
            id="tenant_a_doc_2",
            doc_id="secret_a_v2",
            text="Updated audit report for Tenant Alpha.",
            embedding=[0.06] * 384,
            namespace="tenant_a",
        )
        engine.ingest_chunks([chunk_a_update])
        # Engine index_version should have incremented
        assert engine.index_version >= 2
        # Tenant A cache must be invalidated
        cached_after = engine.cache.get(
            "financial quarterly statements earnings",
            engine.embedder.embed_text("financial quarterly statements earnings"),
            namespace="tenant_a",
            index_version=str(engine.index_version - 1),
        )
        assert cached_after is None

    asyncio.run(_run())

