# SpecRAG: Comprehensive Test Infrastructure & Quality Assurance Specification

**Document Version**: 1.0.0  
**Author**: `test_writer_track_b` (Track B: E2E Testing Track)  
**Target Engine**: SpecRAG (Speculative Retrieval-Augmented Generation Engine)  
**Project Root**: `D:\Ngoding\speculative-rag-engine`  
**Execution Environment**: Windows Native (Python 3.12, no Docker)

---

## 1. Executive Summary & Quality Strategy

SpecRAG is a high-performance speculative RAG engine designed to run 100% natively on Windows under Python 3.12 without containerization dependencies (Docker/WSL2). It achieves near-cross-encoder accuracy while cutting mean query latency by 50–70% through dynamic confidence gating (Shannon entropy + marginal score gap), Reciprocal Rank Fusion (RRF), and sub-20ms semantic caching.

Track B (E2E Testing Track) delivers an **authoritative, requirement-driven, opaque-box test suite** derived directly from `ORIGINAL_REQUEST.md` and `PROJECT.md § Feature Inventory`. The suite is architected around the **Four-Tier Testing Pyramid**, guaranteeing comprehensive validation across all 16 inventoried features, rigorous boundary and stress limits, multi-system pairwise interactions, and realistic domain-specific workloads.

```
                     ▲
                    / \
                   /   \
                  / T4  \     Tier 4: Real-World Workload Scenarios (>=8 scenarios)
                 /-------\    - Technical Q&A, Scientific, Customer Support, Ambiguity
                /   T3    \   Tier 3: Cross-Feature Combinations (Pairwise interactions)
               /-----------\  - Hybrid+Gating, Cache Miss+Fast Path, Rerank+Convex Blend
              /     T2      \ Tier 2: Boundary & Corner Cases (>=80 tests across 16 features)
             /---------------\- Out-of-vocabulary, zero-vectors, ties, overflow, large sets
            /       T1        \Tier 1: Feature Coverage (>=80 tests across 16 features)
           /-------------------\- Isolated functional verification of each feature contract
```

---

## 2. Test Architecture & Multi-Runner Design

### 2.1 Dual-Runner Compatibility
The target host environment runs Python 3.12 on Windows. Standard library `unittest` is guaranteed to be available with zero extra packages, while `pytest` may or may not be installed. 

To ensure continuous, friction-free verification across development, CI, and local development:
1. **Primary CLI Multi-Runner**: `python run_tests.py`
   - Discovers and executes tests with rich terminal summaries, timing statistics, and structured JSON telemetry export.
   - Automatically detects whether `pytest` is installed. If present and selected, it executes through pytest; otherwise, it falls back to Python's built-in `unittest.TestLoader` without requiring any external dependencies.
2. **Standard Test Frameworks**:
   - `python run_tests.py --tier all`
   - `python -m unittest discover tests -v`
   - `pytest tests/ -v`

### 2.2 CLI Runner Interface (`run_tests.py`)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--tier` | String | `all` | Select test tier to run: `1`, `2`, `3`, `4`, or `all`. |
| `--runner` | String | `auto` | Execution engine: `auto` (auto-detect pytest vs unittest), `pytest`, or `unittest`. |
| `-v`, `--verbose` | Flag | `False` | Verbose output displaying each test name and execution duration. |
| `--pattern` | String | `test_*.py` | Filename pattern filter for selective test execution. |
| `--json-output` | Path | `None` | Path to export test execution summary and metrics in JSON format. |
| `--failfast` | Flag | `False` | Stop execution on the first failure or error encountered. |

### 2.3 Exit Codes
- `0`: All executed test cases passed successfully.
- `1`: One or more test cases failed or encountered an unhandled exception.
- `2`: Invalid CLI arguments or runner configuration error.

---

## 3. The Four-Tier Test Suite Specification

### 3.1 Tier 1: Feature Coverage (`tests/test_tier1_features.py`)
Tier 1 establishes isolated verification for each of the 16 features defined in `PROJECT.md § Feature Inventory`. Each feature possesses **at least 5 distinct test cases** (minimum 80 test cases total):

| # | Feature Under Test | Test Prefix | Core Assertions & Verification Target |
|---|--------------------|-------------|---------------------------------------|
| 1 | Dual-Storage Abstraction | `test_f01_` | SQLite CRUD, metadata JSON serialization, float32 BLOB vector encoding, batch upsert, Postgres fallback handling. |
| 2 | Sparse BM25 Keyword Search | `test_f02_` | Term frequency, Robertson-Spärck Jones IDF, tokenization, multi-term query ranking, zero-match candidate behavior. |
| 3 | Dense Semantic Vector Search | `test_f03_` | Dot-product cosine similarity, matrix shape validation, top-k ranking order, custom top-k slicing, metadata persistence. |
| 4 | Reciprocal Rank Fusion (RRF) | `test_f04_` | $k=60$ scoring, theoretical max normalization ($[0, 1]$), multi-modal consensus dominance, 5-stage tie-breaker, disjoint candidate sets. |
| 5 | Temperature-Scaled Softmax | `test_f05_` | Probability sum $= 1.0$, monotonicity preservation ($s_i > s_j \implies p_i > p_j$), temperature sensitivity ($\tau$), max-subtraction stability, single candidate $[1.0]$. |
| 6 | Shannon Entropy Computation | `test_f06_` | Uniform distribution $\tilde{H} = 1.0$, one-hot certainty $\tilde{H} = 0.0$, scaling across $K=5, 10, 20$, safe handling of $0 \log 0$, normalization bounds $[0, 1]$. |
| 7 | Marginal Score Gap ($\Delta$) | `test_f07_` | Score difference $\Delta = s_1 - s_2$, single candidate default $\Delta = 1.0$, exact tie $\Delta = 0.0$, distinct margin verification ($\Delta \ge 0.15$), joint confidence index $C$. |
| 8 | Dynamic Speculative Routing | `test_f08_` | Fast-path bypass conjunction ($\Delta \ge 0.15 \land \tilde{H} \le 0.45 \land s_1 \ge 0.30$), relevance floor failure, high entropy failure, narrow gap failure, telemetry emission. |
| 9 | Cross-Encoder Reranker & Fallback | `test_f09_` | Pairwise scoring, pure Python fallback rescorer, convex score blending ($0.85 \cdot s_{\text{CE}} + 0.15 \cdot s_{\text{RRF}}$), candidate re-ordering, circuit-breaker timeout fallback. |
| 10 | In-Memory Semantic Cache | `test_f10_` | Cache put/get exact match, sub-20ms latency SLA, LRU capacity eviction, TTL expiration, cache flush. |
| 11 | Redis Cache Adapter | `test_f11_` | Adapter initialization, set/get operations, automatic fallback to in-memory on connection failure, clear operation, payload serialization. |
| 12 | Semantic Similarity Matcher | `test_f12_` | Exact vector match (similarity 1.0), near-match cosine $\ge 0.92$ hit, dissimilar query miss ($< 0.92$), SHA-256 hash shortcut, empty cache fast miss. |
| 13 | FastAPI Asynchronous Service | `test_f13_` | `/health` 200 OK, `/api/v1/query` response schema, `/api/v1/ingest` batch chunk persistence, `/api/v1/cache` flush, latency header verification. |
| 14 | Dual Startup Entrypoints | `test_f14_` | CLI `run.py --help`, CLI `run.py --test-startup` non-blocking verification, `app.main:app` FastAPI instance, port configuration, clean shutdown code. |
| 15 | Automated Evaluation Suite | `test_f15_` | Runner tier filtering, test auto-discovery, JSON metrics export, dual runner compatibility, non-zero exit code on failure. |
| 16 | Ablation & Benchmark Harness | `test_f16_` | 3-mode evaluation (Fast, Rerank, SpecRAG), latency percentiles (P50, P90, P99), NDCG@10 relevance metric, compute savings $\% \ge 40\%$, Pareto trade-off curve. |

---

### 3.2 Tier 2: Boundary & Corner Cases (`tests/test_tier2_boundaries.py`)
Tier 2 stress-tests algorithmic stability, numeric limits, degenerate inputs, and error recovery across all 16 features (minimum 80 test cases total):

1. **Storage Limits**: Empty chunk ingestion, 0-length text strings, 100,000-character massive text, Unicode/emojis/math symbols, auto-creation of missing parent directories.
2. **Lexical Limits**: Out-of-vocabulary terms, punctuation/stop-word only queries (`"the is a and"`), single-character queries (`"a"`, `"?"`), whitespace queries, corpus with zero-IDF repeated terms.
3. **Vector Limits**: All-zero query vectors (zero-norm division guard), anti-parallel vectors (similarity $-1.0$), dimension mismatch exceptions, extreme floating point numbers ($\pm 1e10$, $1e-30$), empty index search.
4. **Fusion Limits**: Disjoint candidate lists from both retrievers, multi-way identical score ties resolved deterministically by 5-stage tie-breaker, extreme smoothing constants ($k=1$ vs $k=10000$), massive pools ($N=1000$), zero weight handling.
5. **Softmax Limits**: Numerical overflow prevention via max-subtraction with $s = [1e6, 1e6-1]$, extreme negative values, near-zero temperature ($\tau = 0.001$), infinite temperature ($\tau = 1000$), empty candidate vectors.
6. **Entropy Limits**: Single candidate $K=1$ yielding 0.0 bits, exact two-way tie ($p=[0.5, 0.5]$) yielding maximum normalized entropy 1.0, extreme probability skew ($p=[1.0 - 1e-12, 1e-12]$), invalid negative probabilities rejection, large $K$ ($K=1000$) boundary invariance.
7. **Marginal Gap Limits**: Identical top-2 scores ($\Delta = 0.0$), single candidate boundary ($\Delta = 1.0$), machine epsilon differences ($s_1 = 0.5 + 1e-15, s_2 = 0.5$), unsorted input auto-sorting.
8. **Routing Limits**: Exact boundary values ($\Delta = 0.15, \tilde{H} = 0.45, s_1 = 0.30$) inclusive checks, sub-floor rejection ($s_1 = 0.299$), wide gap but high entropy rejection, low entropy but narrow gap rejection, unconditional `force_rerank` override.
9. **Reranker Limits**: Empty candidate lists, empty text candidates, timeout circuit breaker triggering graceful pass-through, extreme sequence lengths, convex blend boundary conditions ($\lambda = 0.0$ and $\lambda = 1.0$).
10. **Cache Limits**: Single-element capacity ($N=1$) instant LRU eviction, non-positive TTL handling, concurrent read/write thread-safety, 10MB large payload storage, all-zero query vector handling.
11. **Redis Limits**: Unreachable Redis host (`localhost:9999`) graceful fallback, mid-session connection drop recovery, authentication failure handling, corrupted cache payload recovery.
12. **Matcher Limits**: Boundary cosine similarities ($\text{sim} = 0.920000$, $0.919999$, $0.920001$), unnormalized vector inputs, case-insensitive whitespace-tolerant exact string hashing.
13. **API Limits**: Malformed JSON bodies, negative/zero `top_k`, excessive `top_k` ($>100$), empty ingestion lists, ultra-long queries (10,000 characters).
14. **Entrypoint Limits**: Unknown CLI flags, invalid port numbers (`-1`, `999999`), port collision detection, graceful SIGINT exit, read-only storage fallback.
15. **Runner Limits**: Invalid `--tier` argument, zero matched tests pattern, unwritable JSON path handling, concurrent test executions.
16. **Benchmark Limits**: Empty benchmark dataset handling, zero relevant document calculation ($IDCG=0 \implies NDCG=0.0$), zero latency division guard, 100% fast-path compute savings calculation.

---

### 3.3 Tier 3: Cross-Feature Combinations (`tests/test_tier3_combinations.py`)
Tier 3 exercises pairwise interactions and state transitions across subsystem boundaries:
1. **Hybrid Retrieval + Dynamic Confidence Gating**: Verifies that multi-modal consensus (both dense and BM25 ranking doc #1) drives down entropy and expands margin gap, successfully triggering `FAST_PATH_BYPASS`. Conversely, inter-modal disagreement triggers `RERANK_HIGH_ENTROPY`.
2. **Semantic Cache Miss + Fast-Path Retrieval + Cache Write**: Traces query execution on cold start (cache miss), parallel hybrid retrieval, fast-path bypass, asynchronous cache insertion, and immediate subsequent identical query returning cache HIT in $< 1.5$ms.
3. **Semantic Cache Paraphrase Match**: Verifies that a semantically rephrased query with cosine similarity $0.94 \ge 0.92$ short-circuits the entire retrieval and gating pipeline.
4. **Gating Rerank Triggered + Cross-Encoder Execution + Convex Score Blending**: Validates ambiguous candidate routing to the cross-encoder, deep pair rescoring, and convex combination $0.85 \cdot s_{\text{CE}} + 0.15 \cdot s_{\text{RRF}}$ updating final rank order.
5. **Storage Ingestion + Lexical Indexing + Vector Matrix Sync**: Documents chunk ingestion into SQLite, verifying that both BM25 inverted index and in-memory NumPy matrix immediately reflect the newly added documents.
6. **Redis Down + In-Memory Fallback + Fast-Path Query**: Simulates offline Redis, verifying transparent transition to in-memory LRU without dropping queries or raising 500 errors.
7. **Relevance Floor Failure + Reranker Timeout Circuit Breaker**: Tests out-of-domain query ($s_1 < 0.30$) triggering reranking, combined with cross-encoder timeout triggering deterministic fallback to RRF order.
8. **End-to-End API Multi-Route Flow**: Executes sequence of Cache Miss -> Fast Path -> Cache Hit -> Force Rerank over HTTP ASGI schemas.

---

### 3.4 Tier 4: Real-World Application Workloads (`tests/test_tier4_workloads.py`)
Tier 4 tests realistic end-to-end RAG application workloads with domain-specific vocabularies, query distributions, and concurrency profiles:
1. **Workload 1: Technical Documentation Q&A (Python AsyncIO)**: Ingests technical documentation on Python coroutines, event loops, and tasks. Tests keyword-precise and conceptual queries.
2. **Workload 2: Scientific & Biomedical Queries (CRISPR-Cas9)**: Ingests genetic engineering and biomedical literature. Tests domain-specific terminology, acronyms, and semantic density.
3. **Workload 3: Customer Support & E-commerce Troubleshooting (Return Policy)**: Tests colloquial, conversational queries ("Where is my refund?", "How do I return worn shoes?"), verifying semantic cache hits on varied paraphrases.
4. **Workload 4: Ambiguous Multi-Intent Disambiguation ("Apple performance")**: Evaluates polysemous queries matching both consumer technology hardware and agricultural fruit nutrition, verifying high entropy detection and cross-encoder disambiguation.
5. **Workload 5: Legal & Compliance Contract Analysis**: Ingests confidentiality, indemnity, and termination clauses; tests strict lexical precision combined with semantic retrieval.
6. **Workload 6: Developer API Reference (REST Authentication)**: Tests queries with code syntax, headers (`Authorization: Bearer <token>`), and status codes (`401 Unauthorized`).
7. **Workload 7: Financial Market & Banking Queries**: Disambiguates financial institutions from river banks and geographical terms using dense vector embeddings.
8. **Workload 8: High-Concurrency Burst Simulation**: Simulates 50 concurrent asynchronous queries executing across cache, hybrid search, and gating to verify thread-safety, memory stability, and zero race conditions.

---

## 4. Mathematical Invariants & Reference Oracles

Every test assertion in the SpecRAG test suite is derived from strict mathematical properties and specifications documented in `survey_report.md` and `PROJECT.md`.

### 4.1 Reciprocal Rank Fusion (RRF)
$$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + r_m(d)}$$
- Constant: $k = 60$, weights $w_{\text{dense}} = w_{\text{bm25}} = 1.0$.
- Theoretical upper bound: $RRF_{\max} = \frac{1}{61} + \frac{1}{61} = \frac{2}{61} \approx 0.032787$.
- Normalized score: $s_i = \frac{\text{RRF}(d_i)}{RRF_{\max}} \in [0.0, 1.0]$.
- **5-Stage Deterministic Tie-Breaker**:
  $$\text{Key}(d) = \Big( -\text{RRF}(d), \; -|M(d)|, \; \min_{m} r_m(d), \; -s_{\text{dense}}(d), \; \text{doc\_id}(d) \Big)$$

### 4.2 Numerically Stable Temperature-Scaled Softmax
$$z_i = \frac{s_i - \max_j s_j}{\tau}, \quad p_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}$$
- Default temperature: $\tau = 0.25$.
- Invariant 1: $\sum_{i=1}^K p_i = 1.0 \pm 10^{-6}$.
- Invariant 2: $p_i \ge 0 \quad \forall i$.
- Invariant 3: $s_i > s_j \implies p_i > p_j$.

### 4.3 Normalized Shannon Entropy
$$H(P) = -\sum_{i=1}^K p_i \log_2(p_i + 10^{-12}), \quad \tilde{H}(P) = \frac{H(P)}{\log_2(K)}$$
- Invariant 1 (One-hot distribution): $\tilde{H}([1.0, 0, \dots, 0]) = 0.0$.
- Invariant 2 (Uniform distribution): $\tilde{H}([1/K, \dots, 1/K]) = 1.0 \pm 10^{-6}$.
- Invariant 3 (Boundedness): $\tilde{H}(P) \in [0.0, 1.0]$ for all valid probability vectors.

### 4.4 Marginal Score Gap & Continuous Confidence
$$\Delta = s_1 - s_2 \quad (\text{with } \Delta = 1.0 \text{ if } K=1)$$
$$C = \Delta \cdot (1 - \tilde{H}(P)) \cdot s_1 \in [0.0, 1.0]$$

### 4.5 Speculative Fast-Path Decision Conjunction
$$\text{FastPath}(q) = (\Delta \ge 0.15) \;\land\; (\tilde{H}(P) \le 0.45) \;\land\; (s_1 \ge 0.30)$$

### 4.6 Convex Score Integration
$$s_{\text{final}}(d) = 0.85 \cdot s_{\text{CE}}(d) + 0.15 \cdot s_{\text{RRF\_norm}}(d)$$

---

## 5. Test Directory & Artifact Layout

```
D:\Ngoding\speculative-rag-engine\
├── TEST_INFRA.md                   # This authoritative test infrastructure specification
├── TEST_READY.md                   # Final readiness report, coverage matrix, and test counts
├── run_tests.py                    # Multi-runner CLI entrypoint (unittest + pytest)
└── tests/
    ├── __init__.py                 # Test package initialization
    ├── conftest.py                 # Pytest & Unittest fixtures, mock corpus, mathematical oracles
    ├── test_tier1_features.py      # Tier 1: 16 Features x >=5 tests (>=80 isolated tests)
    ├── test_tier2_boundaries.py    # Tier 2: 16 Features x >=5 tests (>=80 boundary/corner tests)
    ├── test_tier3_combinations.py  # Tier 3: Cross-feature pairwise interaction tests
    └── test_tier4_workloads.py     # Tier 4: >=8 Domain-specific realistic RAG workloads
```

---

## 6. How to Run the Tests

```powershell
# 1. Run all test tiers using the multi-runner
python run_tests.py

# 2. Run a specific tier (e.g. Tier 1 features only)
python run_tests.py --tier 1 -v

# 3. Run with JSON summary output
python run_tests.py --tier all --json-output test_results.json

# 4. Run via standard library unittest discover
python -m unittest discover tests -v

# 5. Run via pytest (if installed)
pytest tests/ -v
```
