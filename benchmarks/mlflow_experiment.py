"""MLflow Experiment Tracking for SpecRAG gating threshold exploration.

This module wraps the ablation benchmark inside an MLflow experiment run,
logging gating hyperparameters and retrieval quality metrics so threshold
combinations can be compared in the MLflow UI (mlflow ui → localhost:5000).

Usage:
    python benchmarks/mlflow_experiment.py
    python benchmarks/mlflow_experiment.py --sweep
    mlflow ui  # view results at http://127.0.0.1:5000
"""

import argparse
import json
import math
import os
import sys
import time
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
import mlflow.tracking

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.api.schemas import DocumentChunk, QueryRequest
from app.config import Settings, get_settings
from app.core.embedder import DenseEmbedder
from app.core.engine import SpecRAGEngine
from app.core.gating import DynamicConfidenceGate
from app.core.reranker import NativeRescorer
from app.evaluation.metrics import RagasCompatibleEvaluator
from app.retrieval.bm25 import OkapiBM25Index
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.vector_store import DenseVectorIndex

EXPERIMENT_NAME = "SpecRAG-GatingThreshold-Sweep"

# --------------------------------------------------------------------------- #
# Inline corpus & ground-truth (same 20-doc corpus as benchmark_ablation.py)  #
# --------------------------------------------------------------------------- #

def _build_corpus():
    corpus_defs = [
        ("dist_01", "Raft is a consensus algorithm designed as an alternative to Paxos. It achieves consensus via leader election, log replication, and safety guarantees."),
        ("dist_02", "Paxos is a family of protocols for solving consensus in a network of unreliable processors, widely implemented in Chubby and Spanner."),
        ("dist_03", "Two-Phase Commit (2PC) is a distributed transaction protocol ensuring atomic commit across multiple autonomous database nodes."),
        ("dist_04", "Vector clocks provide partial ordering of events in distributed systems and detect causal concurrency violations."),
        ("dist_05", "CAP theorem states that any distributed data store can only provide at most two out of Consistency, Availability, and Partition tolerance."),
        ("rag_01", "Dense bi-encoders encode queries and documents independently into high-dimensional vector spaces using Siamese transformer networks."),
        ("rag_02", "Cross-encoders process query and candidate chunk simultaneously through full cross-attention layers, yielding superior scoring accuracy at high latency."),
        ("rag_03", "Reciprocal Rank Fusion (RRF) combines rankings from multiple information retrieval systems without requiring raw score normalization."),
        ("rag_04", "Okapi BM25 is a probabilistic term-frequency ranking function modeling term saturation and document length normalization."),
        ("rag_05", "Speculative RAG employs entropy gating over candidate confidence distributions to conditionally trigger computationally expensive rerankers."),
        ("db_01", "B-trees maintain sorted key-value pairs allowing logarithmic amortized search, insertion, and deletion times in relational storage engines."),
        ("db_02", "LSM trees optimize write throughput by sequentially writing to an immutable memtable and sstables with compaction."),
        ("db_03", "PostgreSQL GiST and SP-GiST indexes provide generalized search trees supporting multi-dimensional geometric and spatial indexing."),
        ("db_04", "SQLite stores data in single cross-platform disk files, utilizing B-tree page hierarchies with write-ahead logging (WAL) for concurrency."),
        ("db_05", "Redis is an in-memory key-value data structure store supporting strings, hashes, lists, sets, sorted sets, and bitmap operations."),
        ("ml_01", "Adam optimizer computes individual adaptive learning rates from estimates of first and second moments of the gradients."),
        ("ml_02", "Transformer self-attention maps query, key, and value vectors via scaled dot-product attention with softmax normalization."),
        ("ml_03", "ONNX Runtime executes optimized computational graphs using hardware-specific execution providers (DirectML, TensorRT, CPU OpenMP)."),
        ("ml_04", "Cosine similarity measures the cosine of the angle between two non-zero vectors in inner product space, invariant to scale."),
        ("ml_05", "Shannon entropy quantifies the expected amount of information or uncertainty contained in a probability distribution."),
    ]
    return [
        DocumentChunk(
            id=f"chunk_{cid}",
            doc_id=f"doc_{cid.split('_')[0]}",
            text=text,
            metadata={"domain": cid.split("_")[0]},
        )
        for cid, text in corpus_defs
    ]


EVAL_QUERIES = [
    {
        "query": "Raft consensus algorithm leader election and log replication",
        "gt": {"chunk_dist_01", "chunk_dist_02"},
    },
    {
        "query": "Log-Structured Merge-trees LSM write throughput memtable compaction",
        "gt": {"chunk_db_02", "chunk_db_01"},
    },
    {
        "query": "Shannon entropy uncertainty information probability distribution",
        "gt": {"chunk_ml_05"},
    },
    {
        "query": "Reciprocal Rank Fusion RRF ranking combination without normalization",
        "gt": {"chunk_rag_03", "chunk_rag_04"},
    },
    {
        "query": "Two-Phase Commit distributed transaction atomic commit protocol",
        "gt": {"chunk_dist_03"},
    },
    {
        "query": "distributed consensus protocol state machine fault tolerance",
        "gt": {"chunk_dist_01", "chunk_dist_02", "chunk_dist_05"},
    },
    {
        "query": "information retrieval ranking algorithms comparison",
        "gt": {"chunk_rag_04", "chunk_rag_03", "chunk_rag_01", "chunk_rag_02"},
    },
    {
        "query": "neural network mathematical optimization techniques",
        "gt": {"chunk_ml_01", "chunk_ml_02", "chunk_ml_05"},
    },
]


# --------------------------------------------------------------------------- #
# Core benchmark runner for a single hyperparameter configuration             #
# --------------------------------------------------------------------------- #

def run_single_experiment(
    theta_gap: float,
    theta_entropy: float,
    theta_min: float,
    tau: float,
    top_k: int,
    n_cycles: int = 3,
) -> Dict[str, Any]:
    """Execute SpecRAG benchmark for one gating hyperparameter set, return metrics dict."""
    settings = get_settings()
    db_path = f"mlflow_bench_{int(time.time()*1000)}.db"

    storage = SQLiteStorageRepository(db_path)
    vector_idx = DenseVectorIndex(dimension=settings.VECTOR_DIMENSION)
    bm25_idx = OkapiBM25Index()
    embedder = DenseEmbedder(dimension=settings.VECTOR_DIMENSION)

    custom_settings = Settings(
        GATING_TAU=tau,
        GATING_THETA_GAP=theta_gap,
        GATING_THETA_ENTROPY=theta_entropy,
        GATING_THETA_MIN=theta_min,
        GATING_TOP_K=top_k,
    )

    engine = SpecRAGEngine(
        storage_repo=storage,
        vector_index=vector_idx,
        bm25_index=bm25_idx,
        embedder=embedder,
        reranker=NativeRescorer(),
        settings=custom_settings,
    )

    corpus = _build_corpus()
    engine.ingest_chunks(corpus)

    evaluator = RagasCompatibleEvaluator(embedder=embedder)

    latencies, route_counts = [], {"CACHE_HIT": 0, "FAST_PATH_BYPASS": 0, "RERANKED": 0}
    all_queries, all_retrieved_ids, all_gt_ids, all_retrieved_texts = [], [], [], []

    workload = EVAL_QUERIES * n_cycles
    for item in workload:
        req = QueryRequest(query=item["query"], top_k=top_k, use_cache=True)
        resp = engine.query_sync(req)

        latencies.append(resp.latency_ms)
        route_counts[resp.route_taken] = route_counts.get(resp.route_taken, 0) + 1

        all_queries.append(item["query"])
        all_retrieved_ids.append([c.chunk_id for c in resp.results])
        all_gt_ids.append(item["gt"])
        all_retrieved_texts.append([c.text for c in resp.results])

    ragas_result = evaluator.evaluate_batch(
        queries=all_queries,
        retrieved_ids_per_query=all_retrieved_ids,
        ground_truth_ids_per_query=all_gt_ids,
        retrieved_texts_per_query=all_retrieved_texts,
        k=top_k,
    )

    total = len(workload)
    bypass_ratio = route_counts.get("FAST_PATH_BYPASS", 0) / total
    cache_hit_ratio = route_counts.get("CACHE_HIT", 0) / total
    rerank_ratio = route_counts.get("RERANKED", 0) / total

    metrics = {
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "mean_latency_ms": float(np.mean(latencies)),
        "bypass_ratio": round(bypass_ratio, 4),
        "cache_hit_ratio": round(cache_hit_ratio, 4),
        "rerank_ratio": round(rerank_ratio, 4),
        **ragas_result.to_dict(),
    }

    storage.close()
    try:
        os.remove(db_path)
    except Exception:
        pass

    return metrics


# --------------------------------------------------------------------------- #
# MLflow logging wrapper                                                       #
# --------------------------------------------------------------------------- #

def log_run(
    theta_gap: float,
    theta_entropy: float,
    theta_min: float,
    tau: float,
    top_k: int,
    n_cycles: int = 3,
    run_name: Optional[str] = None,
) -> None:
    """Execute one benchmark configuration and log everything to MLflow."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = {
        "tau": tau,
        "theta_gap": theta_gap,
        "theta_entropy": theta_entropy,
        "theta_min": theta_min,
        "top_k": top_k,
        "n_cycles": n_cycles,
    }

    if run_name is None:
        run_name = f"gap={theta_gap}_ent={theta_entropy}_min={theta_min}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        print(f"  [MLflow] Running: {run_name}")
        metrics = run_single_experiment(
            theta_gap=theta_gap,
            theta_entropy=theta_entropy,
            theta_min=theta_min,
            tau=tau,
            top_k=top_k,
            n_cycles=n_cycles,
        )
        mlflow.log_metrics(metrics)

        print(
            f"  >> bypass={metrics['bypass_ratio']:.1%} | "
            f"ragas_composite={metrics['ragas_composite_score']:.4f} | "
            f"p50={metrics['p50_latency_ms']:.2f}ms"
        )


def main():
    parser = argparse.ArgumentParser(description="SpecRAG MLflow experiment tracking")
    parser.add_argument("--sweep", action="store_true", help="Run full hyperparameter grid sweep")
    parser.add_argument("--cycles", type=int, default=3, help="Query cycles per run")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    if args.sweep:
        print(f"[MLflow] Starting hyperparameter sweep — Experiment: '{EXPERIMENT_NAME}'")
        # Focused Grid: theta_gap x theta_entropy
        theta_gaps = [0.10, 0.15, 0.20]
        theta_entropies = [0.40, 0.45, 0.50]
        theta_mins = [0.30]

        total = len(theta_gaps) * len(theta_entropies) * len(theta_mins)
        print(f"[MLflow] {total} configurations to evaluate...\n")

        for gap, entropy, tmin in product(theta_gaps, theta_entropies, theta_mins):
            log_run(
                theta_gap=gap,
                theta_entropy=entropy,
                theta_min=tmin,
                tau=0.25,
                top_k=10,
                n_cycles=args.cycles,
            )
    else:
        print(f"[MLflow] Logging default SpecRAG configuration — Experiment: '{EXPERIMENT_NAME}'")
        log_run(
            theta_gap=0.15,
            theta_entropy=0.45,
            theta_min=0.30,
            tau=0.25,
            top_k=10,
            n_cycles=args.cycles,
            run_name="SpecRAG-Default-Config",
        )

    print(f"\n[MLflow] Done. View results with:\n  mlflow ui\n  → http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
