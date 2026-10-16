# Original User Request

## Initial Request — 2026-10-16T08:09:12Z

Build **SpecRAG**, a lightweight, production-grade, low-latency, entropy-gated speculative retrieval-augmented generation engine with hybrid search, dynamic confidence gating, cross-encoder reranking, and semantic caching that runs natively on Windows without requiring Docker.

Working directory: `D:\Ngoding\speculative-rag-engine`
Integrity mode: development

## Requirements

### R1. Native Lightweight Storage & Hybrid Retrieval Layer
Support both local PostgreSQL (`psql`) and embedded storage (SQLite / Chroma / in-memory index) without requiring Docker. Implement Reciprocal Rank Fusion (RRF) between dense semantic vectors and sparse BM25 keyword search.

### R2. Entropy & Dynamic Confidence Gating
Statistical scoring mechanism computing marginal score gap and Shannon entropy over top-k candidate retrieval to dynamically route queries:
- High-confidence queries bypass the cross-encoder reranker directly to the generator (<40ms).
- Ambiguous/high-entropy queries trigger cross-encoder reranking.

### R3. In-Memory & Redis Semantic Cache Layer
FastAPI async API layer with semantic similarity caching to serve repeated/similar queries in sub-25ms. Must operate in native Windows environment (supporting Laragon's local Redis or lightweight in-process fallback) without Docker.

### R4. Automated Evaluation & Ablation Benchmarks
Automated test suite (`pytest`) and benchmarking script (`benchmark_ablation.py`) demonstrating Pareto frontier comparison (latency, accuracy NDCG@10, and compute cost reduction).

## Acceptance Criteria

### Execution & Compatibility
- [ ] Runs 100% natively on Windows using Python 3.12 without requiring Docker or virtual machines.
- [ ] Single entrypoint startup via `python run.py` or `uvicorn app.main:app`.

### Performance & Latency
- [ ] P99 retrieval latency for high-confidence queries under 45ms.
- [ ] Semantic cache hit latency under 20ms.

### Quality & Accuracy
- [ ] Entropy gate accurately distinguishes confident vs ambiguous queries on test sets.
- [ ] Automated test suite passes (`pytest tests/ -v`) with full code coverage across core algorithms.


## Follow-up — 2026-10-16T09:41:17Z

USER INSTRUCTION: Hold execution of Milestone 2 immediately. Please pause all implementation subagents for Milestone 2. Stand by while we generate the comprehensive documentation (PRD.md, ARCHITECTURE.md, CONTEXT.md) as requested by the user.
