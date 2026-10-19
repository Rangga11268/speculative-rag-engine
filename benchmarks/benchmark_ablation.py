"""Ablation Benchmarking Suite for SpecRAG.

Compares three retrieval architectures side-by-side:
1. Vanilla Bi-Encoder (Dense Vector Search only)
2. Full Cross-Encoder (Hybrid Retrieval + 100% Mandatory Reranker)
3. SpecRAG (Hybrid Retrieval + Dynamic Confidence Gating + Speculative Reranking + Semantic Cache)

Measures:
- Latency (Mean, P50, P95, P99 in ms)
- Ranking Accuracy (NDCG@10, Recall@10)
- Speculative Routing & Compute Reduction (% Bypass, % Cache Hit, % Reranked)
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.api.schemas import DocumentChunk, QueryRequest
from app.config import get_settings
from app.core.embedder import DenseEmbedder
from app.core.engine import SpecRAGEngine
from app.core.reranker import NativeRescorer
from app.retrieval.bm25 import OkapiBM25Index
from app.retrieval.hybrid import HybridRetriever
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.vector_store import DenseVectorIndex


# -----------------------------------------------------------------------------
# Metric Calculations
# -----------------------------------------------------------------------------

def calculate_dcg(relevance_scores: List[float], k: int = 10) -> float:
    """Calculate Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
    return dcg


def calculate_ndcg(retrieved_ids: List[str], ground_truth: Dict[str, float], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank k."""
    if not ground_truth:
        return 0.0

    retrieved_rel = [ground_truth.get(cid, 0.0) for cid in retrieved_ids[:k]]
    actual_dcg = calculate_dcg(retrieved_rel, k)

    ideal_rel = sorted(ground_truth.values(), reverse=True)[:k]
    ideal_dcg = calculate_dcg(ideal_rel, k)

    return (actual_dcg / ideal_dcg) if ideal_dcg > 0.0 else 0.0


def calculate_recall(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
    """Calculate Recall at rank k."""
    if not ground_truth_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    hits = len(top_k_set.intersection(ground_truth_ids))
    return hits / float(len(ground_truth_ids))


def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 10) -> float:
    """Calculate Mean Reciprocal Rank at rank k."""
    if not ground_truth_ids:
        return 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        if cid in ground_truth_ids:
            return 1.0 / (i + 1)
    return 0.0


# -----------------------------------------------------------------------------
# Synthetic Benchmark Corpus Generation
# -----------------------------------------------------------------------------

def build_benchmark_corpus() -> Tuple[List[DocumentChunk], List[Dict[str, Any]]]:
    """Generate a realistic technical corpus with domain-specific queries and ground truth."""
    corpus_definitions = [
        # Distributed Systems
        ("dist_01", "Raft is a consensus algorithm designed as an alternative to Paxos. It achieves consensus via leader election, log replication, and safety guarantees."),
        ("dist_02", "Paxos is a family of protocols for solving consensus in a network of unreliable processors, widely implemented in Chubby and Spanner."),
        ("dist_03", "Two-Phase Commit (2PC) is a distributed transaction protocol ensuring atomic commit across multiple autonomous database nodes."),
        ("dist_04", "Vector clocks provide partial ordering of events in distributed systems and detect causal concurrency violations."),
        ("dist_05", "CAP theorem states that any distributed data store can only simultaneously provide at most two out of Consistency, Availability, and Partition tolerance."),
        
        # Retrieval-Augmented Generation & Search
        ("rag_01", "Dense bi-encoders encode queries and documents independently into high-dimensional vector spaces using Siamese transformer networks."),
        ("rag_02", "Cross-encoders process query and candidate chunk simultaneously through full cross-attention layers, yielding superior scoring accuracy at high latency."),
        ("rag_03", "Reciprocal Rank Fusion (RRF) combines rankings from multiple information retrieval systems without requiring raw score normalization."),
        ("rag_04", "Okapi BM25 is a probabilistic term-frequency ranking function modeling term saturation and document length normalization."),
        ("rag_05", "Speculative RAG employs entropy gating over candidate confidence distributions to conditionally trigger computationally expensive rerankers."),

        # Database Systems & Indexing
        ("db_01", "B-trees maintain sorted key-value pairs allowing logarithmic amortized search, insertion, and deletion times in relational storage engines."),
        ("db_02", "LSM trees (Log-Structured Merge-trees) optimize write throughput by sequentially writing to an immutable memtable and sstables with compaction."),
        ("db_03", "PostgreSQL GiST and SP-GiST indexes provide generalized search trees supporting multi-dimensional geometric and spatial indexing."),
        ("db_04", "SQLite stores data in single cross-platform disk files, utilizing B-tree page hierarchies with write-ahead logging (WAL) for concurrency."),
        ("db_05", "Redis is an in-memory key-value data structure store supporting strings, hashes, lists, sets, sorted sets, and bitmap operations."),

        # Machine Learning & Optimization
        ("ml_01", "Adam optimizer computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients."),
        ("ml_02", "Transformer self-attention maps query, key, and value vectors via scaled dot-product attention with softmax normalization."),
        ("ml_03", "ONNX Runtime executes optimized computational graphs using hardware-specific execution providers (DirectML, TensorRT, CPU OpenMP)."),
        ("ml_04", "Cosine similarity measures the cosine of the angle between two non-zero vectors in inner product space, invariant to scale."),
        ("ml_05", "Shannon entropy quantifies the expected amount of information or uncertainty contained in a probability distribution."),
    ]

    chunks = [
        DocumentChunk(
            id=f"chunk_{cid}",
            doc_id=f"doc_{cid.split('_')[0]}",
            text=text,
            metadata={"domain": cid.split("_")[0]},
        )
        for cid, text in corpus_definitions
    ]

    # Benchmark queries: clear queries (high confidence), ambiguous queries (high entropy), and repeated queries (cache test)
    query_benchmarks = [
        # Clear Queries (Distinct Top-1, should trigger FAST_PATH_BYPASS)
        {
            "query": "Raft consensus algorithm leader election and log replication",
            "ground_truth": {"chunk_dist_01": 3.0, "chunk_dist_02": 1.0},
            "type": "clear",
        },
        {
            "query": "Log-Structured Merge-trees LSM write throughput memtable compaction",
            "ground_truth": {"chunk_db_02": 3.0, "chunk_db_01": 1.0},
            "type": "clear",
        },
        {
            "query": "Shannon entropy uncertainty information in probability distribution",
            "ground_truth": {"chunk_ml_05": 3.0},
            "type": "clear",
        },
        {
            "query": "Reciprocal Rank Fusion RRF ranking combination without normalization",
            "ground_truth": {"chunk_rag_03": 3.0, "chunk_rag_04": 1.0},
            "type": "clear",
        },
        {
            "query": "Two-Phase Commit distributed transaction atomic commit protocol",
            "ground_truth": {"chunk_dist_03": 3.0, "chunk_dist_01": 1.0},
            "type": "clear",
        },
        # Ambiguous / Overlapping Queries (Score gap small, should trigger RERANK)
        {
            "query": "distributed consensus protocol state machine",
            "ground_truth": {"chunk_dist_01": 3.0, "chunk_dist_02": 3.0, "chunk_dist_05": 2.0},
            "type": "ambiguous",
        },
        {
            "query": "information retrieval ranking algorithms comparison",
            "ground_truth": {"chunk_rag_04": 3.0, "chunk_rag_03": 3.0, "chunk_rag_01": 2.0, "chunk_rag_02": 2.0},
            "type": "ambiguous",
        },
        {
            "query": "database storage structures and indexing techniques",
            "ground_truth": {"chunk_db_01": 3.0, "chunk_db_02": 3.0, "chunk_db_04": 2.0},
            "type": "ambiguous",
        },
        {
            "query": "neural network mathematical optimization techniques",
            "ground_truth": {"chunk_ml_01": 3.0, "chunk_ml_02": 2.0, "chunk_ml_05": 1.0},
            "type": "ambiguous",
        },
        {
            "query": "system architectures for scalable data persistence",
            "ground_truth": {"chunk_db_04": 3.0, "chunk_db_05": 3.0, "chunk_dist_05": 2.0},
            "type": "ambiguous",
        },
        # Repeated queries (to measure Semantic Cache speedup)
        {
            "query": "Raft consensus algorithm leader election and log replication",
            "ground_truth": {"chunk_dist_01": 3.0, "chunk_dist_02": 1.0},
            "type": "repeated",
        },
        {
            "query": "Log-Structured Merge-trees LSM write throughput memtable compaction",
            "ground_truth": {"chunk_db_02": 3.0, "chunk_db_01": 1.0},
            "type": "repeated",
        },
    ]

    return chunks, query_benchmarks


# -----------------------------------------------------------------------------
# Benchmark Runner
# -----------------------------------------------------------------------------

def run_ablation_benchmarks(num_cycles: int = 5) -> Dict[str, Any]:
    """Execute ablation comparison across Vanilla Bi-Encoder, Full Cross-Encoder, and SpecRAG."""
    settings = get_settings()
    chunks, queries = build_benchmark_corpus()

    # Ephemeral storage & engine
    db_path = "specrag_benchmark_temp.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    storage = SQLiteStorageRepository(db_path)
    vector_index = DenseVectorIndex(dimension=settings.VECTOR_DIMENSION)
    bm25_index = OkapiBM25Index()
    embedder = DenseEmbedder(dimension=settings.VECTOR_DIMENSION)
    reranker = NativeRescorer()

    engine = SpecRAGEngine(
        storage_repo=storage,
        vector_index=vector_index,
        bm25_index=bm25_index,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
    )

    # Ingest corpus
    engine.ingest_chunks(chunks)

    # Hybrid retriever for baseline evaluations
    hybrid_retriever = HybridRetriever(
        vector_index=vector_index,
        bm25_index=bm25_index,
        storage_repo=storage,
        default_rrf_k=settings.RRF_K,
    )

    # Workload expansion based on num_cycles
    workload = queries * num_cycles

    results: Dict[str, Any] = {
        "workload_size": len(workload),
        "corpus_size": len(chunks),
        "architectures": {},
    }

    # =========================================================================
    # Baseline 1: BM25-Only (Keyword Search Baseline)
    # =========================================================================
    latencies_bm25: List[float] = []
    ndcgs_bm25: List[float] = []
    recalls_bm25: List[float] = []
    mrrs_bm25: List[float] = []

    for item in workload:
        q_text = item["query"]
        gt = item["ground_truth"]
        gt_keys = set(gt.keys())

        t0 = time.perf_counter()
        bm25_hits = bm25_index.search(q_text, top_k=10)
        retrieved_ids = [cid for cid, _ in bm25_hits]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        latencies_bm25.append(latency_ms)
        ndcgs_bm25.append(calculate_ndcg(retrieved_ids, gt, k=10))
        recalls_bm25.append(calculate_recall(retrieved_ids, gt_keys, k=10))
        mrrs_bm25.append(calculate_mrr(retrieved_ids, gt_keys, k=10))

    results["architectures"]["BM25-Only"] = {
        "mean_latency_ms": round(float(np.mean(latencies_bm25)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies_bm25, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_bm25, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies_bm25, 99)), 2),
        "ndcg_at_10": round(float(np.mean(ndcgs_bm25)), 4),
        "recall_at_10": round(float(np.mean(recalls_bm25)), 4),
        "mrr_at_10": round(float(np.mean(mrrs_bm25)), 4),
        "rerank_calls": 0,
        "rerank_ratio": 0.0,
        "cache_hits": 0,
    }

    # =========================================================================
    # Baseline 2: Vanilla Bi-Encoder (Dense Search Only, No Reranker)
    # =========================================================================
    latencies_bi: List[float] = []
    ndcgs_bi: List[float] = []
    recalls_bi: List[float] = []
    mrrs_bi: List[float] = []

    for item in workload:
        q_text = item["query"]
        gt = item["ground_truth"]
        gt_keys = set(gt.keys())

        t0 = time.perf_counter()
        q_vec = embedder.embed_text(q_text)
        hits = vector_index.search(q_vec, top_k=10)
        retrieved_ids = [cid for cid, _ in hits]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        latencies_bi.append(latency_ms)
        ndcgs_bi.append(calculate_ndcg(retrieved_ids, gt, k=10))
        recalls_bi.append(calculate_recall(retrieved_ids, gt_keys, k=10))
        mrrs_bi.append(calculate_mrr(retrieved_ids, gt_keys, k=10))

    results["architectures"]["Vanilla Bi-Encoder"] = {
        "mean_latency_ms": round(float(np.mean(latencies_bi)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies_bi, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_bi, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies_bi, 99)), 2),
        "ndcg_at_10": round(float(np.mean(ndcgs_bi)), 4),
        "recall_at_10": round(float(np.mean(recalls_bi)), 4),
        "mrr_at_10": round(float(np.mean(mrrs_bi)), 4),
        "rerank_calls": 0,
        "rerank_ratio": 0.0,
        "cache_hits": 0,
    }

    # =========================================================================
    # Baseline 3: Hybrid Retrieval (Dense + BM25 via RRF, No Reranker)
    # =========================================================================
    latencies_hyb: List[float] = []
    ndcgs_hyb: List[float] = []
    recalls_hyb: List[float] = []
    mrrs_hyb: List[float] = []

    for item in workload:
        q_text = item["query"]
        gt = item["ground_truth"]
        gt_keys = set(gt.keys())

        t0 = time.perf_counter()
        q_vec = embedder.embed_text(q_text)
        candidates = hybrid_retriever.retrieve_sync(
            query_text=q_text,
            query_vector=q_vec,
            top_k=10,
        )
        retrieved_ids = [c.chunk_id for c in candidates]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        latencies_hyb.append(latency_ms)
        ndcgs_hyb.append(calculate_ndcg(retrieved_ids, gt, k=10))
        recalls_hyb.append(calculate_recall(retrieved_ids, gt_keys, k=10))
        mrrs_hyb.append(calculate_mrr(retrieved_ids, gt_keys, k=10))

    results["architectures"]["Hybrid (Dense+BM25)"] = {
        "mean_latency_ms": round(float(np.mean(latencies_hyb)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies_hyb, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_hyb, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies_hyb, 99)), 2),
        "ndcg_at_10": round(float(np.mean(ndcgs_hyb)), 4),
        "recall_at_10": round(float(np.mean(recalls_hyb)), 4),
        "mrr_at_10": round(float(np.mean(mrrs_hyb)), 4),
        "rerank_calls": 0,
        "rerank_ratio": 0.0,
        "cache_hits": 0,
    }

    # =========================================================================
    # Baseline 2: Full Cross-Encoder (Hybrid Retrieval + 100% Mandatory Reranker)
    # =========================================================================
    latencies_full: List[float] = []
    ndcgs_full: List[float] = []
    recalls_full: List[float] = []
    mrrs_full: List[float] = []

    for item in workload:
        q_text = item["query"]
        gt = item["ground_truth"]
        gt_keys = set(gt.keys())

        t0 = time.perf_counter()
        q_vec = embedder.embed_text(q_text)
        # Hybrid retrieval
        candidates = hybrid_retriever.retrieve_sync(
            query_text=q_text,
            query_vector=q_vec,
            top_k=20,
        )
        # 100% Cross-encoder reranking
        final_candidates = reranker.rerank(
            query=q_text,
            candidates=candidates,
            top_k=10,
        )
        retrieved_ids = [c.chunk_id for c in final_candidates]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        latencies_full.append(latency_ms)
        ndcgs_full.append(calculate_ndcg(retrieved_ids, gt, k=10))
        recalls_full.append(calculate_recall(retrieved_ids, gt_keys, k=10))
        mrrs_full.append(calculate_mrr(retrieved_ids, gt_keys, k=10))

    results["architectures"]["Full Cross-Encoder"] = {
        "mean_latency_ms": round(float(np.mean(latencies_full)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies_full, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_full, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies_full, 99)), 2),
        "ndcg_at_10": round(float(np.mean(ndcgs_full)), 4),
        "recall_at_10": round(float(np.mean(recalls_full)), 4),
        "mrr_at_10": round(float(np.mean(mrrs_full)), 4),
        "rerank_calls": len(workload),
        "rerank_ratio": 1.0,
        "cache_hits": 0,
    }

    # =========================================================================
    # System: SpecRAG (Hybrid + Entropy Gating + Speculative Reranking + Cache)
    # =========================================================================
    latencies_spec: List[float] = []
    ndcgs_spec: List[float] = []
    recalls_spec: List[float] = []
    mrrs_spec: List[float] = []
    routes_count = {"CACHE_HIT": 0, "FAST_PATH_BYPASS": 0, "RERANKED": 0}

    for item in workload:
        q_text = item["query"]
        gt = item["ground_truth"]
        gt_keys = set(gt.keys())

        req = QueryRequest(query=q_text, top_k=10, use_cache=True)
        resp = engine.query_sync(req)

        latencies_spec.append(resp.latency_ms)
        retrieved_ids = [c.chunk_id for c in resp.results]
        ndcgs_spec.append(calculate_ndcg(retrieved_ids, gt, k=10))
        recalls_spec.append(calculate_recall(retrieved_ids, gt_keys, k=10))
        mrrs_spec.append(calculate_mrr(retrieved_ids, gt_keys, k=10))

        route = resp.route_taken
        routes_count[route] = routes_count.get(route, 0) + 1

    total_q = len(workload)
    rerank_calls = routes_count.get("RERANKED", 0)
    rerank_ratio = round(rerank_calls / total_q, 4)

    results["architectures"]["SpecRAG (Ours)"] = {
        "mean_latency_ms": round(float(np.mean(latencies_spec)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies_spec, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_spec, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies_spec, 99)), 2),
        "ndcg_at_10": round(float(np.mean(ndcgs_spec)), 4),
        "recall_at_10": round(float(np.mean(recalls_spec)), 4),
        "mrr_at_10": round(float(np.mean(mrrs_spec)), 4),
        "rerank_calls": rerank_calls,
        "rerank_ratio": rerank_ratio,
        "cache_hits": routes_count.get("CACHE_HIT", 0),
        "fast_path_bypasses": routes_count.get("FAST_PATH_BYPASS", 0),
        "compute_cost_reduction_pct": round((1.0 - rerank_ratio) * 100.0, 1),
    }

    # Clean up temp db
    storage.close()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    return results


def print_comparison_table(results: Dict[str, Any]) -> None:
    """Print clean formatted markdown and terminal comparison table."""
    archs = results["architectures"]
    print("\n" + "=" * 102)
    print("                              SPECRAG ABLATION BENCHMARK RESULTS")
    print("=" * 102)
    header = f"{'Architecture':<22} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'NDCG@10':<9} | {'Recall@10':<9} | {'MRR@10':<8} | {'Rerank %':<8}"
    print(header)
    print("-" * 102)

    for name, data in archs.items():
        mrr_val = data.get('mrr_at_10', 0.0)
        row = (
            f"{name:<22} | "
            f"{data['p50_latency_ms']:<9.2f} | "
            f"{data['p95_latency_ms']:<9.2f} | "
            f"{data['ndcg_at_10']:<9.4f} | "
            f"{data['recall_at_10']:<9.4f} | "
            f"{mrr_val:<8.4f} | "
            f"{data['rerank_ratio'] * 100:<7.1f}%"
        )
        print(row)
    print("=" * 102)

    spec = archs["SpecRAG (Ours)"]
    full = archs["Full Cross-Encoder"]
    latency_reduction = round((1.0 - (spec["p50_latency_ms"] / full["p50_latency_ms"])) * 100.0, 1)
    print(f"\n[SUMMARY FINDINGS]")
    print(f"* Latency P50 Speedup           : {latency_reduction}% reduction vs Full Cross-Encoder")
    print(f"* Heavy Reranker Compute Saved  : {spec['compute_cost_reduction_pct']}% of queries bypassed reranker")
    print(f"* Ranking Quality Retention     : NDCG@10 of {spec['ndcg_at_10']} vs {full['ndcg_at_10']} ({round(spec['ndcg_at_10']/full['ndcg_at_10']*100, 1)}% parity)")
    print(f"* Semantic Cache Hits Recorded  : {spec['cache_hits']} query hits")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run SpecRAG ablation benchmarks.")
    parser.add_argument("--cycles", type=int, default=5, help="Number of repetitions of test queries (default: 5)")
    parser.add_argument("--output", type=str, default="benchmarks/ablation_report.json", help="Path to save JSON report")
    args = parser.parse_args()

    print(f"[BENCHMARK] Executing ablation benchmark with {args.cycles} cycles...")
    results = run_ablation_benchmarks(num_cycles=args.cycles)
    print_comparison_table(results)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[BENCHMARK] Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
