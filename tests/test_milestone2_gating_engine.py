"""Unit and integration test suite for Milestone 2: Gating, Embedder, and Engine."""

import math
import pytest
import numpy as np

from app.api.schemas import DocumentChunk, QueryRequest, RetrievedCandidate
from app.core.embedder import DenseEmbedder
from app.core.engine import SpecRAGEngine
from app.core.gating import DynamicConfidenceGate, RouteDecision
from app.core.reranker import NativeRescorer
from app.retrieval.bm25 import OkapiBM25Index
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.vector_store import DenseVectorIndex


def test_embedder_dimension_and_normalization():
    """Verify DenseEmbedder produces normalized 384-dim vectors."""
    embedder = DenseEmbedder(dimension=384)
    v1 = embedder.embed_text("enterprise financial regulation")
    v2 = embedder.embed_text("enterprise financial regulation")

    assert len(v1) == 384
    norm = np.linalg.norm(v1)
    assert pytest.approx(norm, rel=1e-4) == 1.0
    # Determinism
    assert v1 == v2


def test_gating_high_confidence_bypass():
    """Dominant top-1 candidate should be granted FAST_PATH_BYPASS."""
    gate = DynamicConfidenceGate(tau=0.25, theta_gap=0.15, theta_entropy=0.45, theta_min=0.30)
    # Standard K=10 candidate pool with one dominant top-1 candidate
    scores = [0.95, 0.25, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
    telemetry = gate.evaluate(scores)

    assert telemetry.bypass_granted is True
    assert telemetry.decision == RouteDecision.FAST_PATH_BYPASS
    assert telemetry.marginal_gap == pytest.approx(0.70, rel=1e-3)
    assert telemetry.normalized_entropy < 0.45


def test_gating_low_gap_ambiguity():
    """Candidates with nearly identical top scores should trigger RERANK_LOW_GAP."""
    gate = DynamicConfidenceGate(tau=0.25, theta_gap=0.15, theta_entropy=0.55, theta_min=0.30)
    # Top 2 are virtually tied, with the rest decaying sharply
    scores = [0.88, 0.86, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
    telemetry = gate.evaluate(scores)

    assert telemetry.bypass_granted is False
    assert telemetry.decision == RouteDecision.RERANK_LOW_GAP
    assert telemetry.marginal_gap == pytest.approx(0.02, rel=1e-3)


def test_gating_high_entropy_flat_distribution():
    """Flat/uniform score distribution should trigger RERANK_HIGH_ENTROPY."""
    gate = DynamicConfidenceGate(tau=0.50, theta_gap=0.01, theta_entropy=0.50, theta_min=0.30)
    scores = [0.60, 0.59, 0.58, 0.57, 0.56]
    telemetry = gate.evaluate(scores)

    assert telemetry.bypass_granted is False
    assert telemetry.decision == RouteDecision.RERANK_HIGH_ENTROPY
    assert telemetry.normalized_entropy > 0.50


def test_gating_relevance_floor():
    """Sub-threshold scores should be routed to RERANK_RELEVANCE_FLOOR."""
    gate = DynamicConfidenceGate(theta_min=0.50)
    scores = [0.25, 0.20, 0.15]
    telemetry = gate.evaluate(scores)

    assert telemetry.bypass_granted is False
    assert telemetry.decision == RouteDecision.RERANK_RELEVANCE_FLOOR


def test_gating_edge_cases():
    """Verify single element, empty list, and infinite/NaN safety."""
    gate = DynamicConfidenceGate(theta_min=0.30)

    # Empty
    t_empty = gate.evaluate([])
    assert t_empty.bypass_granted is False
    assert t_empty.decision == RouteDecision.RERANK_RELEVANCE_FLOOR

    # Single
    t_single = gate.evaluate([0.85])
    assert t_single.bypass_granted is True
    assert t_single.decision == RouteDecision.FAST_PATH_BYPASS

    # Non-finite
    t_nan = gate.evaluate([float("nan"), 0.5, float("inf")])
    assert isinstance(t_nan.normalized_entropy, float)

    # Identical scores (uniform probability distribution)
    t_identical = gate.evaluate([0.75, 0.75, 0.75, 0.75])
    assert t_identical.marginal_gap == 0.0
    assert pytest.approx(t_identical.normalized_entropy, rel=1e-3) == 1.0
    assert t_identical.decision in [RouteDecision.RERANK_LOW_GAP, RouteDecision.RERANK_HIGH_ENTROPY]

    # Negative scores (sub-threshold, routed to floor)
    t_neg = gate.evaluate([-0.10, -0.50, -1.20])
    assert t_neg.bypass_granted is False
    assert t_neg.decision == RouteDecision.RERANK_RELEVANCE_FLOOR

    # Zero tau validation error
    with pytest.raises(ValueError, match="strictly positive"):
        DynamicConfidenceGate(tau=0.0)


def test_native_rescorer():
    """Verify NativeRescorer elevates exact matches and token overlap."""
    reranker = NativeRescorer()
    c1 = RetrievedCandidate(chunk_id="c1", doc_id="d1", text="irrelevant text about animals", score=0.80)
    c2 = RetrievedCandidate(chunk_id="c2", doc_id="d2", text="enterprise financial liquidity and capital reserves", score=0.60)

    query = "financial liquidity reserves"
    results = reranker.rerank(query, [c1, c2], top_k=2)

    # c2 should be boosted above c1 due to exact token overlap
    assert len(results) == 2
    assert results[0].chunk_id == "c2"
    assert results[0].score > results[1].score


def test_specrag_engine_end_to_end(tmp_path):
    """Verify complete SpecRAGEngine query execution and routing."""
    db_path = str(tmp_path / "test_engine.db")
    storage = SQLiteStorageRepository(db_path)
    vec_index = DenseVectorIndex(dimension=384)
    bm25 = OkapiBM25Index()
    embedder = DenseEmbedder(dimension=384)
    engine = SpecRAGEngine(
        storage_repo=storage,
        vector_index=vec_index,
        bm25_index=bm25,
        embedder=embedder,
    )

    # Ingest sample corpus
    chunks = [
        DocumentChunk(
            id="chk-1",
            doc_id="doc-fin",
            text="Basel III requires banking institutions to maintain a minimum liquidity coverage ratio of 100%.",
        ),
        DocumentChunk(
            id="chk-2",
            doc_id="doc-hr",
            text="Standard annual paid leave entitlement for full-time employees is 12 working days.",
        ),
        DocumentChunk(
            id="chk-3",
            doc_id="doc-tech",
            text="PostgreSQL HNSW index supports high-dimensional cosine similarity nearest neighbor search.",
        ),
    ]
    count = engine.ingest_chunks(chunks)
    assert count == 3

    # Query 1: High-confidence query
    req1 = QueryRequest(query="liquidity coverage ratio Basel III", top_k=1)
    res1 = engine.query_sync(req1)

    assert len(res1.results) == 1
    assert res1.results[0].chunk_id == "chk-1"
    assert res1.route_taken in ["FAST_PATH_BYPASS", "RERANKED"]
    assert res1.latency_ms > 0

    # Query 2: Force Rerank
    req2 = QueryRequest(query="annual paid leave days", top_k=1, force_rerank=True)
    res2 = engine.query_sync(req2)
    assert res2.route_taken == "RERANKED"
    assert res2.results[0].chunk_id == "chk-2"


def test_dynamic_confidence_gate_invalid_top_k():
    """Verify top_k < 1 raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        DynamicConfidenceGate(top_k=0)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        DynamicConfidenceGate(top_k=-5)

