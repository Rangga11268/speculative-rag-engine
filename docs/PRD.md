# PRODUCT REQUIREMENTS DOCUMENT (PRD)
## SpecRAG: Low-Latency, Entropy-Gated Speculative RAG Engine

| Metadata | Details |
| :--- | :--- |
| **Project Name** | SpecRAG (Speculative Retrieval-Augmented Generation) |
| **Target Role** | AI Engineer / Applied ML Systems Engineer / Data Scientist |
| **Status** | All Milestones (1 to 4) Completed & Verified (284/284 Tests Passing) |
| **Version** | 1.0.0-release |
| **Author** | Darell Rangga Putra Rachman |
| **Primary Repository** | `D:\Ngoding\speculative-rag-engine` |

---

## 1. Executive Summary & Problem Statement

### 1.1 The Production RAG Dilemma
In enterprise Retrieval-Augmented Generation (RAG) systems (such as querying 10,000+ pages of financial SEC filings, internal banking SOPs, or legal contracts), standard architectures enforce an exhaustive **Cross-Encoder Reranker** (e.g., `BGE-Reranker-Large`, `Cohere Rerank`) over all incoming candidate document chunks.

While Cross-Encoder reranking yields superior semantic ranking precision ($NDCG@10 > 0.78$), it introduces a catastrophic **P99 latency penalty of 300ms–500ms** per query and inflates compute/GPU costs by $4\times$. Conversely, omitting the reranker and relying solely on bi-encoder dense vector retrieval degrades generation accuracy by 30%–40% due to "Lost-in-the-Middle" distractor noise and lexical keyword blindness.

### 1.2 The Broken Assumption
Current industry pipelines treat all user queries identically: **easy, unambiguous queries and complex, multi-hop queries are forced through the exact same heavy, slow reranking pipeline.**

### 1.3 The Solution: SpecRAG
**SpecRAG** introduces an **entropy-gated speculative execution policy**:
1. **Fast Hybrid Retrieval**: Combines Dense Vector embeddings with Okapi BM25 lexical search via Reciprocal Rank Fusion ($k=60$).
2. **Dynamic Confidence & Entropy Gating**: Evaluates the candidate retrieval distribution using Shannon Entropy $\mathcal{H}$ and Marginal Score Gap $\Delta S$.
3. **Speculative Fast-Path Bypass**: For high-confidence queries (~70% of production traffic), the system **bypasses the heavy Cross-Encoder reranker entirely**, delivering retrieved context to generation in **sub-40ms**.
4. **Deliberate Deep-Path**: Only ambiguous or high-entropy queries trigger the heavy Cross-Encoder reranker.
5. **In-Memory & Redis Semantic Caching**: Serves repeated/similar queries in **sub-20ms**.

---

## 2. Target Users & Personas

1. **Enterprise Knowledge Worker**: Needs instant, verifiable answers from complex corporate documentation without waiting for slow multi-second retrieval pipelines.
2. **FinTech & Banking Analyst**: Requires strictly grounded facts, zero hallucination, and verifiable audit citations for compliance.
3. **AI Systems / Infrastructure Engineer**: Needs high-throughput (>100 RPS), predictable P99 latency SLA (<50ms for cached/fast-path queries), zero heavy virtualization dependencies, and clean native execution.

---

## 3. Product Goals & Key Metrics (SLAs)

| Metric | Baseline (Standard Hybrid RAG) | SpecRAG Target (Engineered) | Status / Target |
| :--- | :--- | :--- | :--- |
| **P99 Latency (Fast-Path Bypass)** | 380 ms – 450 ms | **< 45 ms** | Verified in test benchmarks |
| **P99 Latency (Cache Hit)** | 150 ms – 250 ms | **< 20 ms** | Verified in test benchmarks |
| **NDCG@10 Recovery Rate** | Baseline (100%) | **≥ 98.0%** of full Cross-Encoder | Mathematical target |
| **Inference Compute Cost Reduction** | $0 baseline | **≥ 65% reduction** | Via speculative gating & semantic cache |
| **Environment Footprint** | Docker required (2–4 GB RAM) | **Zero-Docker native Python** (<300 MB RAM) | Achieved (Native Windows Python 3.12) |
| **Test Suite Coverage** | Minimal | **180 automated tests (100% pass)** | Achieved (Tiers 1–4) |

---

## 4. Functional Requirements (FR)

### FR-1: Storage Layer Abstraction
- Must support dual persistence: embedded SQLite with `float32` BLOB vector serialization and native local PostgreSQL (via `psql` / port 5432).
- Automatic directory provisioning and connection fallback if PostgreSQL is unavailable.
- Strict vector dimension validation (reject dimension mismatch and non-finite `NaN`/`Inf` vectors).

### FR-2: Lexical Retrieval (Okapi BM25)
- Pure-Python zero-dependency implementation of Okapi BM25.
- Uses Robertson-Spärck Jones IDF with smoothing parameter $k_1 = 1.5$ and $b = 0.75$.
- Thread-safe tokenization, term-frequency inverted index, and document length normalization.

### FR-3: Dense Vector Similarity Search
- In-memory NumPy float32 matrix operations for ultra-low latency cosine similarity.
- Sanitized dot product with zero-vector division safeguards.
- In-batch document ID deduplication.

### FR-4: Reciprocal Rank Fusion (RRF)
- Merges ranked lists from lexical (BM25) and dense vector retrievers.
- Standard RRF parameter: $k = 60$. Formula: $RRF(d) = \sum_{m} \frac{1}{k + r_m(d)}$.
- 5-stage deterministic tie-breaker (combined score &rarr; consensus count &rarr; best rank &rarr; lexical score &rarr; chunk ID string order).

### FR-5: Dynamic Entropy & Confidence Gating (Milestone 2)
- Calculates temperature-scaled softmax distribution over top-$k$ retrieved scores.
- Computes Marginal Score Gap: $\Delta S = s_1 - s_k$.
- Computes Shannon Entropy: $\mathcal{H} = -\sum p_i \log_2 p_i$.
- Determines routing policy: `FAST_PATH` vs `CROSS_ENCODER_PATH`.

### FR-6: Cross-Encoder Reranking (Milestone 2)
- Local lightweight Cross-Encoder execution (ONNX Runtime / FP8/INT8 quantized).
- Evaluates candidate pairs `(query, document_chunk)` to reorder ambiguous candidate sets.

### FR-7: In-Memory & Redis Semantic Cache (Milestone 3)
- Fast vector similarity lookup for past queries.
- Cosine similarity threshold $> 0.95$ triggers instant cache hit (<20ms).
- Graceful in-process LRU cache fallback if Redis daemon is offline.

### FR-8: Serving API Layer (Milestone 3)
- Asynchronous FastAPI application exposing RESTful endpoints:
  - `POST /v1/ingest`: Batch document chunk ingestion and vector indexing.
  - `POST /v1/query`: Query retrieval with routing provenance metadata (`path: "fast_path" | "cross_encoder"`).
  - `GET /v1/health`: Subsystem health status and memory footprint telemetry.

---

## 5. Non-Functional Requirements (NFR)

- **NFR-1 (Portability)**: 100% native execution on Windows 11 with Python 3.12 without requiring Docker or WSL.
- **NFR-2 (Determinism)**: All ranking algorithms, tie-breakers, and routing gates must produce 100% reproducible results for identical inputs.
- **NFR-3 (Robustness)**: Graceful handling of corrupted vectors, empty queries, out-of-vocabulary terms, and zero document stores without crashing.
- **NFR-4 (Memory Footprint)**: Total resident memory for 10,000 document chunks must stay below 350 MB.
- **NFR-5 (Testability)**: Multi-tiered opaque-box and white-box test suites executable via single command `python run_tests.py`.

---

## 6. Project Milestones & Current Progress

```
[Milestone 1: Storage & Hybrid Retrieval]  ✅ COMPLETED & VERIFIED (180 Tests Passed)
   ├── SQLite & PostgreSQL Storage Engines
   ├── Pure-Python Okapi BM25 Lexical Engine
   ├── Dense Vector In-Memory Index (NumPy float32)
   └── Deterministic Reciprocal Rank Fusion (k=60)
[Milestone 2: Entropy Gating & Cross-Encoder] ✅ COMPLETED & VERIFIED (272 Tests Total)
   ├── Shannon Entropy & Marginal Gap Formula (app/core/gating.py)
   ├── Dense Embedder Float32 Normalization (app/core/embedder.py)
   ├── Speculative Dynamic Routing Gate (Fast-Path Bypass)
   └── Native Cross-Encoder Rescorer (app/core/reranker.py)
         │
[Milestone 3: Semantic Caching & FastAPI Serving] ✅ COMPLETED & VERIFIED (9 Tests Passed)
   ├── In-Process / Redis Semantic Cache (MemorySemanticCache & RedisSemanticCache)
   ├── FastAPI Async Endpoints (/api/v1/query, /api/v1/ingest, /api/v1/cache/stats, /health)
   └── Single Launcher Script (run.py)
         │
[Milestone 4: Ablation Benchmarks & Evaluation] ✅ COMPLETED & VERIFIED (3 Tests Passed)
   ├── Automated Pareto Frontier Benchmark (Latency vs NDCG: benchmark_ablation.py)
   ├── Full GitHub README Publication & Architecture Specs
   └── Complete Test Suite: 284/284 Tests Passing (100%)
```
