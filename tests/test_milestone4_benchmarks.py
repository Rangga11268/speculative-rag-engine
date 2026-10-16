"""Tests for Milestone 4: Ablation Benchmarks & Evaluation Suite."""

import os
import pytest
from benchmarks.benchmark_ablation import (
    calculate_dcg,
    calculate_ndcg,
    calculate_recall,
    run_ablation_benchmarks,
)


def test_ndcg_calculation():
    """Verify standard NDCG@K calculation with perfect, partial, and zero relevance."""
    gt = {"doc_1": 3.0, "doc_2": 2.0, "doc_3": 1.0}

    # Perfect ranking
    perfect = ["doc_1", "doc_2", "doc_3"]
    assert calculate_ndcg(perfect, gt, k=3) == pytest.approx(1.0, rel=1e-3)

    # Suboptimal ranking
    suboptimal = ["doc_3", "doc_2", "doc_1"]
    score_sub = calculate_ndcg(suboptimal, gt, k=3)
    assert 0.0 < score_sub < 1.0

    # Irrelevant ranking
    irrelevant = ["doc_x", "doc_y"]
    assert calculate_ndcg(irrelevant, gt, k=3) == 0.0


def test_recall_calculation():
    """Verify Recall@K fraction."""
    gt_ids = {"d1", "d2", "d3"}
    retrieved = ["d1", "d2", "d4", "d5"]
    assert calculate_recall(retrieved, gt_ids, k=4) == pytest.approx(2.0 / 3.0)
    assert calculate_recall([], gt_ids, k=5) == 0.0


def test_run_ablation_benchmarks_smoke(tmp_path):
    """Smoke test running 1 cycle of ablation benchmarks and asserting metrics presence."""
    results = run_ablation_benchmarks(num_cycles=1)

    assert "architectures" in results
    archs = results["architectures"]
    assert "Vanilla Bi-Encoder" in archs
    assert "Full Cross-Encoder" in archs
    assert "SpecRAG (Ours)" in archs

    spec = archs["SpecRAG (Ours)"]
    full = archs["Full Cross-Encoder"]

    # Verify SpecRAG matches or closely approaches Full Cross-Encoder ranking quality
    assert spec["ndcg_at_10"] >= full["ndcg_at_10"] * 0.95

    # Verify SpecRAG bypassed reranker on clear queries
    assert spec["compute_cost_reduction_pct"] > 0.0
