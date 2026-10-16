# SpecRAG: Developer & AI Agent Guide (AGENTS.md)

## Project Overview
**SpecRAG** is an entropy-gated speculative Retrieval-Augmented Generation (RAG) engine designed for low-latency, low-cost enterprise retrieval. High-confidence queries bypass heavy Cross-Encoder rerankers via Shannon Entropy and Marginal Gap gating (<40ms), while ambiguous queries route to deliberate reranking.

---

## Tech Stack & Architecture
- **Language & Runtime:** Python 3.12+ (100% Native Windows, Zero-Docker).
- **Web API:** FastAPI, Uvicorn, Pydantic v2.
- **Storage:** Dual-engine (Embedded SQLite with `float32` BLOB serialization + local PostgreSQL fallback).
- **Retrieval:** Hybrid (Dense Vector NumPy SIMD dot-product + Pure-Python Okapi BM25).
- **Fusion:** Deterministic Reciprocal Rank Fusion ($k=60$) with 5-stage tie-breaker.
- **Gating Policy:** Shannon Entropy ($\mathcal{H}_{\text{norm}}$) + Marginal Score Gap ($\Delta S$).
- **Cache:** In-process LRU Semantic Cache with cosine similarity threshold (>0.92) + Redis fallback.

---

## Essential Commands

```bash
# Run complete test suite (284 tests across all tiers & milestones)
python -m pytest tests/
# or
python run_tests.py

# Run ablation benchmark (Vanilla Bi-Encoder vs Full Cross-Encoder vs SpecRAG)
python benchmarks/benchmark_ablation.py

# Launch FastAPI server
python run.py
# API docs available at http://127.0.0.1:8000/docs
```

---

## Architectural Boundaries & Non-Negotiables
1. **Zero-Docker Requirement:** Never introduce Docker containers, WSL-only scripts, or heavy virtualization. Everything must run natively on Windows.
2. **No Heavy GPU/Torch Dependencies:** Keep compute footprint light (<300 MB RAM). Reranking and vector operations rely on CPU-optimized vector math / ONNX.
3. **Data Integrity:** All embeddings must be strictly L2-normalized `float32` arrays. Sanitize against `NaN`, `Inf`, and zero-vector divisions.
4. **Resilience & Fallback:** Storage must gracefully fall back to SQLite if PostgreSQL is unreachable; cache must fall back to in-memory LRU if Redis is offline.
5. **Testing Contract:** Any code changes must maintain 100% pass rate on `tests/`. Always run `python -m pytest tests/` before considering any task complete.
