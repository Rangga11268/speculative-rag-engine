"""Unit tests for Ragas-compatible evaluation metrics and MLflow integration."""

import os
import pytest
import numpy as np

from app.core.embedder import DenseEmbedder
from app.evaluation.metrics import RagasCompatibleEvaluator, EvaluationResult
from benchmarks.mlflow_experiment import run_single_experiment


def test_ragas_context_precision_and_recall():
    """Verify context precision and recall calculations."""
    evaluator = RagasCompatibleEvaluator()

    retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
    relevant = {"chunk_1", "chunk_3"}

    # Precision@2: ["chunk_1", "chunk_2"] -> 1/2 = 0.5
    assert evaluator.context_precision(retrieved, relevant, k=2) == 0.5

    # Precision@4: ["chunk_1", "chunk_2", "chunk_3", "chunk_4"] -> 2/4 = 0.5
    assert evaluator.context_precision(retrieved, relevant, k=4) == 0.5

    # Recall@2: ["chunk_1", "chunk_2"] -> 1/2 = 0.5
    assert evaluator.context_recall(retrieved, relevant, k=2) == 0.5

    # Recall@4: ["chunk_1", "chunk_2", "chunk_3", "chunk_4"] -> 2/2 = 1.0
    assert evaluator.context_recall(retrieved, relevant, k=4) == 1.0

    # Edge cases
    assert evaluator.context_precision([], relevant) == 0.0
    assert evaluator.context_recall(retrieved, set()) == 0.0


def test_ragas_answer_relevancy():
    """Verify embedding-based answer relevancy proxy."""
    embedder = DenseEmbedder(dimension=384)
    evaluator = RagasCompatibleEvaluator(embedder=embedder)

    query = "Distributed consensus algorithm Paxos"
    relevant_contexts = [
        "Paxos solves consensus in a network of unreliable processors.",
        "Raft is an alternative consensus algorithm to Paxos.",
    ]
    irrelevant_contexts = [
        "Chocolate chip cookies recipe with butter and sugar.",
        "Gardening tips for growing organic tomatoes in summer.",
    ]

    rel_score = evaluator.answer_relevancy(query, relevant_contexts)
    irrel_score = evaluator.answer_relevancy(query, irrelevant_contexts)

    # Relevancy proxy should score higher on semantically related contexts
    assert 0.0 <= rel_score <= 1.0
    assert 0.0 <= irrel_score <= 1.0
    assert rel_score > irrel_score


def test_ragas_batch_evaluation():
    """Verify batch evaluation aggregation and composite score calculation."""
    evaluator = RagasCompatibleEvaluator()

    queries = ["q1", "q2"]
    retrieved = [["c1", "c2"], ["c3", "c4"]]
    ground_truth = [{"c1"}, {"c3", "c4"}]

    res = evaluator.evaluate_batch(
        queries=queries,
        retrieved_ids_per_query=retrieved,
        ground_truth_ids_per_query=ground_truth,
        k=2,
    )

    assert isinstance(res, EvaluationResult)
    assert res.n_queries == 2
    # q1: P=1/2=0.5, R=1/1=1.0
    # q2: P=2/2=1.0, R=2/2=1.0
    # mean_P = 0.75, mean_R = 1.0
    assert res.mean_context_precision == pytest.approx(0.75)
    assert res.mean_context_recall == pytest.approx(1.0)
    assert res.to_dict()["n_queries_evaluated"] == 2


def test_run_single_experiment_smoke():
    """Smoke test running single experiment configuration."""
    metrics = run_single_experiment(
        theta_gap=0.15,
        theta_entropy=0.45,
        theta_min=0.30,
        tau=0.25,
        top_k=5,
        n_cycles=1,
    )

    assert "p50_latency_ms" in metrics
    assert "bypass_ratio" in metrics
    assert "ragas_context_precision" in metrics
    assert "ragas_context_recall" in metrics
    assert "ragas_composite_score" in metrics
