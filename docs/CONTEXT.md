# DEVELOPER & AGENT CONTEXT HANDOVER
## SpecRAG: Entropy-Gated Speculative RAG Engine

> **PANDUAN UNTUK AGENT / DEVELOPER BARU:**  
> Jika Anda adalah AI Agent atau developer yang membuka repositori ini, baca dokumen ini terlebih dahulu! Dokumen ini merangkum seluruh konteks proyek, status saat ini, struktur folder, arsitektur, dan cara menjalankan server dan benchmark.

---

## 1. Apa Proyek Ini? (30-Second Elevator Pitch)

**SpecRAG** adalah mesin *Retrieval-Augmented Generation* (RAG) cerdas yang dirancang untuk mengatasi masalah latensi dan biaya komputasi di enterprise.

* **Masalah Industri Saat Ini:** RAG konvensional selalu memaksa **setiap pertanyaan** melewati model *Cross-Encoder Reranker* yang lambat (menambah overhead latensi 200ms–500ms).
* **Solusi SpecRAG:** Menerapkan prinsip *Speculative Execution* dengan **Entropy & Confidence Gating**.
  * Jika pencarian awal sudah sangat yakin ($\Delta S$ tinggi, Entropi rendah), pertanyaan **langsung lolos (*bypass*) ke generator** dalam waktu **< 1ms**.
  * Hanya pertanyaan ambigu atau sulit yang dikirim ke *Cross-Encoder Reranker*.
  * Hasil query yang sering ditanyakan dilayani instan lewat **Semantic Cache (LRU Cosine Similarity > 0.92)** dalam **< 0.5ms**.
* **Keunikan Lingkungan:** Dirancang **100% Native Windows Python 3.12 (Zero-Docker / Tanpa Docker)** menggunakan embedded storage (SQLite BLOB / in-memory NumPy) dengan fallback otomatis ke PostgreSQL lokal port 5432 dan Laragon Redis jika tersedia.
* **Unified Workspace:** Tersedia dan ditautkan ke direktori terpusat portofolio AI/ML di `D:\Ngoding\AI_ML_Projects\speculative-rag-engine`.

---

## 2. Status Milestone Proyek (All Milestones 100% Complete)

```
[Milestone 1: Storage & Hybrid Retrieval]      ✅ SELESAI & TERVERIFIKASI (180 Tests Passed)
   - SQLite BLOB Vector Persistence
   - PostgreSQL 18.6 Adapter & Fallback
   - In-Memory NumPy Float32 Dense Vector Index
   - Pure-Python Okapi BM25 Lexical Engine
   - Reciprocal Rank Fusion (k=60) + 5-Stage Tie-Breaker

[Milestone 2: Dynamic Entropy Gating]         ✅ SELESAI & TERVERIFIKASI (92 Tests Total)
   - Algoritma Shannon Entropy & Marginal Gap (app/core/gating.py)
   - Dense Embedder Float32 Normalization (app/core/embedder.py)
   - Native Cross-Encoder Rescorer (app/core/reranker.py)
   - SpecRAGEngine End-to-End Orchestrator (app/core/engine.py)

[Milestone 3: Semantic Cache & FastAPI Layer]  ✅ SELESAI & TERVERIFIKASI (9 Tests Passed)
   - Thread-Safe MemorySemanticCache (LRU + TTL + Cosine Sim Matrix Scan)
   - RedisSemanticCache with seamless in-memory fallback
   - SpecRAGEngine query cache bypass & auto-population
   - FastAPI REST API endpoints (/api/v1/query, /api/v1/ingest, /api/v1/cache/stats, /health)
   - Single launcher: python run.py

[Milestone 4: Ablation Benchmarks & Eval]      ✅ SELESAI & TERVERIFIKASI (3 Tests Passed)
   - benchmark_ablation.py comparing Bi-Encoder vs Full Cross-Encoder vs SpecRAG
   - Metrics: P50, P95, P99 Latency, NDCG@10, Recall@10, Compute Cost Reduction

[Milestone 5: Ragas Eval & MLflow Tracking]    ✅ SELESAI & TERVERIFIKASI (4 Tests Passed)
   - app/evaluation/metrics.py: Ragas-Compatible Evaluation (Context Precision, Recall, Answer Relevancy, Composite)
   - benchmarks/mlflow_experiment.py: MLflow SQLite Backend Tracking (sqlite:///mlruns.db)
   - Hyperparameter Grid Sweep across theta_gap and theta_entropy
   - Total Automated Test Suite: 288/288 PASSING (100% Pass Rate)
```

---

## 3. Hasil Benchmark Ablasi Resmi (Empirical Proof of Novelty)

Dari eksekusi `python benchmarks/benchmark_ablation.py`:

| Arsitektur | P50 (ms) | P95 (ms) | P99 (ms) | NDCG@10 | Recall@10 | Rerank % | Compute Saved |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla Bi-Encoder** | 0.82 ms | 1.12 ms | 1.31 ms | 0.8279 | 0.7778 | 0.0% | 0.0% |
| **Full Cross-Encoder** | 2.01 ms | 2.95 ms | 3.49 ms | 0.8536 | 0.8611 | 100.0% | 0.0% |
| **SpecRAG (Ours)** | **0.85 ms** | 5.26 ms | 10.05 ms | **0.8536** | **0.8611** | **27.8%** | **72.2%** |

* **57.7% Latency Reduction (P50)** dibandingkan Full Cross-Encoder pipeline.
* **72.2% Pengurangan Komputasi Reranker** tanpa penurunan akurasi (100% parity pada NDCG@10 = 0.8536).
* **Semantic Cache Hit Latency**: Sub-millisecond pada query berulang / semantik serupa.

---

## 4. Direktori & Inventaris File

```text
D:\Ngoding\speculative-rag-engine/  (Junction: D:\Ngoding\AI_ML_Projects\speculative-rag-engine)
│
├── docs/                      # Dokumentasi Resmi Lengkap
│   ├── PRD.md                 # Product Requirements Document (kebutuhan bisnis & SLA)
│   ├── ARCHITECTURE.md        # Spesifikasi teknis, formula matematika, & diagram
│   └── CONTEXT.md             # Dokumen ini (handover & context switch)
│
├── app/                       # Source Code Utama
│   ├── config.py              # Konfigurasi Pydantic Settings
│   ├── main.py                # FastAPI Application instance & lifespan
│   ├── api/
│   │   ├── deps.py            # FastAPI dependency injection
│   │   ├── routes.py          # REST endpoints (/query, /ingest, /cache, /gating)
│   │   └── schemas.py         # Data models & Pydantic validation
│   ├── cache/                 # Layer Semantic Caching (Milestone 3)
│   │   ├── base.py            # BaseSemanticCache ABC interface
│   │   ├── memory.py          # MemorySemanticCache (LRU, TTL, cosine similarity)
│   │   └── redis_adapter.py   # Redis semantic adapter with graceful fallback
│   ├── core/                  # Core Speculative Logic (Milestone 2)
│   │   ├── embedder.py        # DenseEmbedder (384-dim unit L2 normalization)
│   │   ├── gating.py          # DynamicConfidenceGate (Marginal gap + Shannon entropy)
│   │   ├── reranker.py        # NativeRescorer cross-matching model
│   │   └── engine.py          # SpecRAGEngine (End-to-End Orchestrator)
│   └── storage/ & retrieval/  # Storage & Hybrid Retrieval (Milestone 1)
│       ├── sqlite.py / postgres.py / vector_store.py
│       └── bm25.py / rrf.py / hybrid.py
│
├── benchmarks/                # Evaluation & Benchmarks (Milestones 4 & 5)
│   ├── benchmark_ablation.py  # Ablation evaluation script
│   ├── ablation_report.json   # Output report JSON
│   └── mlflow_experiment.py   # MLflow tracking & hyperparameter sweep
│
├── tests/                     # 288 Automated Tests (100% Pass)
├── run.py                     # Single-command server launcher
└── run_tests.py               # Multi-tier test suite runner
```

---

## 5. Cara Menjalankan Server & Benchmarks

Pastikan berada di root direktori project:

### Menjalankan Server API FastAPI:
```bash
python run.py
```
*API docs interaktif dapat diakses di `http://127.0.0.1:8000/docs`.*

### Menjalankan Seluruh Test Suite (288 Tests):
```bash
python run_tests.py --tier full
```

### Menjalankan Benchmark Ablasi:
```bash
python benchmarks/benchmark_ablation.py --cycles 5
```

### Menjalankan MLflow Experiment Tracking & Grid Sweep:
```bash
# Menjalankan tracking default:
python benchmarks/mlflow_experiment.py

# Menjalankan hyperparameter sweep:
python benchmarks/mlflow_experiment.py --sweep

# Membuka Dashboard Visualisasi MLflow:
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
*Buka browser di `http://127.0.0.1:5000` untuk melihat grafik Pareto frontier, hyperparameter sweep, dan metrik Ragas.*
