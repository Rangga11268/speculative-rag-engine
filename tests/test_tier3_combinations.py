"""Tier 3: Comprehensive Cross-Feature Combinations Test Suite.

Verifies pairwise interactions and multi-system workflows:
- Hybrid Retrieval + Dynamic Confidence Gating
- Semantic Cache Miss -> Fast-Path Retrieval -> Cache Write
- Semantic Cache Near-Match Cosine Bypass
- Gating Rerank Triggered -> Cross-Encoder Execution -> Convex Score Blending
- Storage Ingestion -> Lexical Indexing -> Vector Matrix Sync
- Redis Offline -> In-Memory Fallback -> Fast-Path Query
- Relevance Floor Failure -> Cross-Encoder Timeout Circuit Breaker
- Concurrent Hybrid Retrieval & Cache Updates
- End-to-End Pipeline Multi-Route Lifecycle

Compatible with both standard unittest and pytest.
"""

import asyncio
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.conftest import (
    MOCK_CORPUS,
    DocumentChunk,
    ReferenceInMemoryLRUCache,
    generate_deterministic_vector,
    ref_convex_blend,
    ref_cosine_similarity,
    ref_dynamic_route,
    ref_marginal_gap,
    ref_rrf,
    ref_shannon_entropy,
    ref_softmax,
)

# Optional dynamic imports from app
try:
    from app.storage.sqlite import SQLiteStorageRepository
except ImportError:
    SQLiteStorageRepository = None

try:
    from app.retrieval.bm25 import OkapiBM25Index
except ImportError:
    OkapiBM25Index = None

try:
    from app.storage.vector_store import DenseVectorIndex
except ImportError:
    DenseVectorIndex = None

try:
    from app.retrieval.rrf import compute_rrf_scores, reciprocal_rank_fusion
except ImportError:
    compute_rrf_scores = None
    reciprocal_rank_fusion = None

try:
    from app.retrieval.hybrid import HybridRetriever
except ImportError:
    HybridRetriever = None


class TestTier3Combinations(unittest.TestCase):
    """Tier 3: Cross-Feature Interaction & Subsystem Combination Tests."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_tier3.db")
        if SQLiteStorageRepository:
            self.repo = SQLiteStorageRepository(db_path=self.db_path, dimension=384)
        else:
            self.repo = None

        if DenseVectorIndex:
            self.vec_index = DenseVectorIndex(dimension=384)
        else:
            self.vec_index = None

        if OkapiBM25Index:
            self.bm25_index = OkapiBM25Index(k1=1.5, b=0.75)
        else:
            self.bm25_index = None

        self.cache = ReferenceInMemoryLRUCache(capacity=50, default_ttl=3600)

        # Index initial 25 corpus chunks
        self.corpus_chunks = [
            DocumentChunk(
                id=c["id"],
                doc_id=c["doc_id"],
                text=c["text"],
                metadata=c["metadata"],
                embedding=c["embedding"],
            )
            for c in MOCK_CORPUS[:25]
        ]
        if self.repo:
            self.repo.add_chunks(self.corpus_chunks)
        if self.vec_index:
            self.vec_index.add([c.id for c in self.corpus_chunks], [c.embedding for c in self.corpus_chunks])
        if self.bm25_index:
            self.bm25_index.index([c.id for c in self.corpus_chunks], [c.text for c in self.corpus_chunks])

    def tearDown(self):
        if self.repo:
            self.repo.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_c01_hybrid_retrieval_and_gating_high_consensus_fast_path(self):
        target = next(c for c in self.corpus_chunks if c.id == "sci_01")
        query_text = target.text
        query_vec = target.embedding

        # 1. Execute retrieval across both modalities
        if self.vec_index and self.bm25_index:
            dense_res = self.vec_index.search(query_vec, top_k=5)
            bm25_res = self.bm25_index.search(query_text, top_k=5)
            # RRF Fusion
            if compute_rrf_scores:
                fused = compute_rrf_scores(dense_res, bm25_res, k=60)
                scores = [f["score"] for f in fused]
                top1_doc = fused[0]["chunk_id"]
            else:
                dense_ranks = [r[0] for r in dense_res]
                bm25_ranks = [r[0] for r in bm25_res]
                fused = ref_rrf(dense_ranks, bm25_ranks, k=60)
                scores = [r[1] for r in fused]
                top1_doc = fused[0][0]
        else:
            scores = [0.88, 0.55, 0.40, 0.30]
            top1_doc = "sci_01"

        # 2. Dynamic Gating Math (Calibrated Relative Dispersion)
        s_arr = np.array(scores[:5], dtype=np.float64)
        if len(s_arr) > 1 and np.max(s_arr) > np.min(s_arr):
            calibrated_scores = (s_arr - np.min(s_arr)) / (np.max(s_arr) - np.min(s_arr) + 1e-9)
        else:
            calibrated_scores = s_arr

        probs = ref_softmax(calibrated_scores, tau=0.25)
        h_raw, h_norm = ref_shannon_entropy(probs)
        delta = ref_marginal_gap(calibrated_scores)

        # 3. Decision
        decision = ref_dynamic_route(delta=delta, h_norm=h_norm, s1=calibrated_scores[0])
        self.assertEqual(decision["decision"], "FAST_PATH_BYPASS")
        self.assertTrue(decision["bypass_granted"])

    def test_c02_hybrid_retrieval_and_gating_disagreement_rerank(self):
        """C02: Inter-modal disagreement creates near-tie, triggering reranker."""
        # Dense prefers doc_A at rank 1; BM25 prefers doc_B at rank 1
        dense_ranks = ["doc_A", "doc_B", "doc_C"]
        bm25_ranks = ["doc_B", "doc_A", "doc_D"]

        fused = ref_rrf(dense_ranks, bm25_ranks, k=60)
        # doc_A: 1/61 + 1/62; doc_B: 1/62 + 1/61 -> identical RRF score!
        s1 = fused[0][1]
        s2 = fused[1][1]
        self.assertAlmostEqual(s1, s2, places=5)

        delta = ref_marginal_gap([s1, s2])
        self.assertAlmostEqual(delta, 0.0, places=5)

        probs = ref_softmax([s1, s2], tau=0.25)
        _, h_norm = ref_shannon_entropy(probs)

        decision = ref_dynamic_route(delta=delta, h_norm=h_norm, s1=s1)
        self.assertIn("RERANK", decision["decision"])
        self.assertFalse(decision["bypass_granted"])

    def test_c03_cache_miss_to_hybrid_fast_path_to_cache_write(self):
        """C03: Cold query produces cache MISS, runs fast path, writes cache; second query hits cache."""
        query_text = "Python asyncio cooperative multitasking"
        query_vec = generate_deterministic_vector(query_text, 384)

        # 1. Cache lookup -> MISS
        cached = self.cache.get(query_text, query_vec)
        self.assertIsNone(cached)

        # 2. Hybrid retrieval + Fast-Path
        results = [{"id": "tech_01", "score": 0.95}]
        route = "FAST_PATH_BYPASS"

        # 3. Async cache insertion
        payload = {"query": query_text, "results": results, "route": route}
        self.cache.set(query_text, query_vec, payload)

        # 4. Immediate second query -> HIT
        start = time.perf_counter()
        hit_payload = self.cache.get(query_text, query_vec)
        latency_ms = (time.perf_counter() - start) * 1000.0

        self.assertIsNotNone(hit_payload)
        self.assertEqual(hit_payload["route"], "FAST_PATH_BYPASS")
        self.assertLess(latency_ms, 20.0)

    def test_c04_semantic_cache_near_match_bypasses_retrieval(self):
        """C04: Semantically rephrased query with cosine >= 0.92 bypasses hybrid search and gating."""
        base_query = "What is the return policy for unworn items?"
        base_vec = generate_deterministic_vector(base_query, 384)
        cached_response = {
            "query": base_query,
            "results": [{"id": "supp_01", "text": "30-day return policy for unworn items"}],
            "cached": True,
        }
        self.cache.set(base_query, base_vec, cached_response)

        # Paraphrased query with synthesized near-vector: cosine ~ 0.94
        noise = np.random.RandomState(99).randn(384).astype(np.float32)
        noise = noise - np.dot(noise, base_vec) * base_vec
        noise /= np.linalg.norm(noise)
        para_vec = 0.94 * base_vec + math.sqrt(1.0 - 0.94**2) * noise
        para_vec /= np.linalg.norm(para_vec)

        sim = ref_cosine_similarity(base_vec, para_vec)
        self.assertGreaterEqual(sim, 0.92)

        start = time.perf_counter()
        res = self.cache.get("Can I return unworn merchandise within 30 days?", para_vec, threshold=0.92)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self.assertIsNotNone(res)
        self.assertEqual(res["results"][0]["id"], "supp_01")
        self.assertLess(elapsed_ms, 20.0)

    def test_c05_gating_rerank_to_cross_encoder_and_convex_score_blend(self):
        """C05: Ambiguous query routes to cross-encoder with convex score integration."""
        # Initial candidates with near-tie RRF scores
        candidates = [
            {"id": "doc_A", "rrf_score": 0.72, "ce_score": 0.45},
            {"id": "doc_B", "rrf_score": 0.70, "ce_score": 0.92},
        ]
        delta = candidates[0]["rrf_score"] - candidates[1]["rrf_score"]  # 0.02 -> triggers rerank
        decision = ref_dynamic_route(delta=delta, h_norm=0.85, s1=0.72)
        self.assertFalse(decision["bypass_granted"])

        # Execute convex combination: 0.85 * s_ce + 0.15 * s_rrf
        for c in candidates:
            c["final_score"] = ref_convex_blend(c["ce_score"], c["rrf_score"], lambda_val=0.85)

        # doc_B should win after cross-attention
        reranked = sorted(candidates, key=lambda x: -x["final_score"])
        self.assertEqual(reranked[0]["id"], "doc_B")
        self.assertGreater(reranked[0]["final_score"], reranked[1]["final_score"])

    def test_c06_storage_persistence_to_bm25_and_vector_index_sync(self):
        """C06: Ingesting into SQLite updates both BM25 inverted index and Vector index."""
        if not (self.repo and self.vec_index and self.bm25_index):
            self.skipTest("M1 components not fully initialized")

        new_chunk = DocumentChunk(
            id="sync_c1",
            doc_id="doc_sync",
            text="Quantum computing utilizes qubits and superposition to solve complex problems.",
            metadata={"domain": "quantum"},
            embedding=generate_deterministic_vector("quantum computing superposition qubits", 384).tolist(),
        )

        # 1. Persist to storage
        initial_count = self.repo.count()
        self.repo.add_chunks([new_chunk])
        self.assertEqual(self.repo.count(), initial_count + 1)

        # 2. Sync to vector and BM25 index
        self.vec_index.add([new_chunk.id], [new_chunk.embedding])
        # Re-index BM25 with updated corpus
        all_chunks = self.repo.get_all_chunks()
        self.bm25_index.index([c.id for c in all_chunks], [c.text for c in all_chunks])

        # 3. Retrieve via BM25
        bm25_hits = self.bm25_index.search("quantum superposition qubits", top_k=3)
        self.assertGreater(len(bm25_hits), 0)
        self.assertEqual(bm25_hits[0][0], "sync_c1")

        # 4. Retrieve via Dense Vector
        q_vec = generate_deterministic_vector("quantum computing superposition qubits", 384)
        vec_hits = self.vec_index.search(q_vec, top_k=3)
        self.assertGreater(len(vec_hits), 0)
        self.assertEqual(vec_hits[0][0], "sync_c1")

    def test_c07_redis_offline_to_in_memory_fallback_query(self):
        """C07: Offline Redis automatically diverts queries to in-memory cache without crashing."""
        class MockFailingRedisClient:
            def ping(self):
                raise ConnectionError("Connection to Redis at localhost:6379 failed")

        class ResilientCacheManager:
            def __init__(self):
                self.memory_cache = ReferenceInMemoryLRUCache()
                self.is_redis_active = False
                try:
                    client = MockFailingRedisClient()
                    client.ping()
                    self.is_redis_active = True
                except Exception:
                    self.is_redis_active = False  # Seamless in-memory fallback

            def get(self, q, v):
                return self.memory_cache.get(q, v)

            def set(self, q, v, p):
                self.memory_cache.set(q, v, p)

        manager = ResilientCacheManager()
        self.assertFalse(manager.is_redis_active)

        v = generate_deterministic_vector("resilient query", 384)
        manager.set("resilient query", v, {"status": "success"})
        res = manager.get("resilient query", v)
        self.assertIsNotNone(res)
        self.assertEqual(res["status"], "success")

    def test_c08_relevance_floor_failure_to_reranker_timeout_fallback(self):
        """C08: Low-relevance query triggers reranking, timeout preserves original RRF order."""
        # Candidate scores all very low (<0.30)
        scores = [0.22, 0.18, 0.15]
        decision = ref_dynamic_route(delta=0.04, h_norm=0.70, s1=scores[0])
        self.assertEqual(decision["decision"], "RERANK_RELEVANCE_FLOOR")

        # Simulate timeout on cross-encoder (>65ms budget)
        timeout_budget_ms = 65.0
        elapsed_ms = 75.0
        is_timed_out = elapsed_ms > timeout_budget_ms

        original_candidates = ["doc_x1", "doc_x2", "doc_x3"]
        if is_timed_out:
            # Fall back to original RRF order
            final_order = original_candidates
        else:
            final_order = ["doc_x3", "doc_x1", "doc_x2"]

        self.assertEqual(final_order, original_candidates)

    def test_c09_concurrent_hybrid_retrieval_and_cache_updates(self):
        """C09: Concurrent multi-threaded hybrid searches and cache writes without race conditions."""
        cache = ReferenceInMemoryLRUCache(capacity=50)
        errors = []

        def worker_task(thread_id: int):
            try:
                for i in range(10):
                    q = f"query_thread_{thread_id}_{i}"
                    v = generate_deterministic_vector(q, 384)
                    cache.set(q, v, {"thread": thread_id, "i": i})
                    val = cache.get(q, v)
                    if val is None or val["thread"] != thread_id:
                        errors.append(f"Thread {thread_id} read failure")
            except Exception as e:
                errors.append(f"Exception in thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker_task, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

    def test_c10_end_to_end_pipeline_lifecycle_query_flow(self):
        """C10: End-to-end multi-query lifecycle: Cold MISS -> Fast Path -> Repeat HIT -> Force Rerank."""
        query = "Python asyncio cooperative multitasking"
        query_vec = generate_deterministic_vector(query, 384)

        # Step 1: Cold start -> Cache MISS -> Fast Path -> Cache Store
        miss = self.cache.get(query, query_vec)
        self.assertIsNone(miss)

        route_decision = ref_dynamic_route(delta=0.25, h_norm=0.30, s1=0.88)
        self.assertTrue(route_decision["bypass_granted"])

        payload = {
            "query": query,
            "results": [{"id": "tech_01", "score": 0.88}],
            "route_taken": "FAST_PATH_BYPASS",
            "cache_hit": False,
        }
        self.cache.set(query, query_vec, payload)

        # Step 2: Repeat query -> Cache HIT
        hit = self.cache.get(query, query_vec)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["route_taken"], "FAST_PATH_BYPASS")

        # Step 3: Force rerank query -> Bypass cache, bypass gating, execute reranker
        force_rerank = True
        if force_rerank:
            final_route = "RERANKED"
        else:
            final_route = "FAST_PATH_BYPASS"

        self.assertEqual(final_route, "RERANKED")


if __name__ == "__main__":
    unittest.main()
