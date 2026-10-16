"""Pytest fixtures, reference implementations, and mock corpus for SpecRAG test suites."""

import builtins
import hashlib
import json
import math
import os
import tempfile
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

try:
    from app.api.schemas import DocumentChunk
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class DocumentChunk:
        id: str
        doc_id: str
        text: str
        metadata: Dict[str, Any] = field(default_factory=dict)
        embedding: Optional[List[float]] = None

# Ensure json is accessible to all test modules
builtins.json = json


# -----------------------------------------------------------------------------
# Reference Mathematical and Algorithmic Functions
# -----------------------------------------------------------------------------

def generate_deterministic_vector(text: str, dim: int = 384) -> np.ndarray:
    """Generate a deterministic unit-normalized float32 vector for a string."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    raw = rng.randn(dim).astype(np.float32)
    norm = float(np.linalg.norm(raw))
    return raw / norm if norm > 0.0 else raw


def ref_softmax(scores: Sequence[float], tau: float = 0.25) -> np.ndarray:
    """Compute temperature-scaled Softmax with numerical max-subtraction."""
    s = np.asarray(scores, dtype=np.float64)
    if len(s) == 0:
        return np.array([], dtype=np.float64)
    if len(s) == 1:
        return np.array([1.0], dtype=np.float64)
    shifted = (s - np.max(s)) / tau
    exp_s = np.exp(shifted)
    return exp_s / np.sum(exp_s)


def ref_shannon_entropy(probs: Sequence[float]) -> Tuple[float, float]:
    """Compute raw Shannon entropy H(P) and normalized entropy H_norm in [0, 1]."""
    p = np.asarray(probs, dtype=np.float64)
    k = len(p)
    if k <= 1:
        return 0.0, 0.0

    # Filter out zero probabilities safely
    mask = p > 0.0
    if not np.any(mask):
        return 0.0, 0.0

    p_valid = p[mask]
    h_raw = float(-np.sum(p_valid * np.log2(p_valid)))
    max_h = math.log2(k)
    h_norm = float(h_raw / max_h) if max_h > 0.0 else 0.0
    return h_raw, max(0.0, min(1.0, h_norm))


def ref_marginal_gap(scores: Sequence[float]) -> float:
    """Compute marginal score gap Delta = s1 - s2."""
    arr = sorted(list(scores), reverse=True)
    if len(arr) <= 1:
        return 1.0
    return float(arr[0] - arr[1])


def ref_dynamic_route(
    delta: float,
    h_norm: float,
    s1: float,
    theta_gap: float = 0.15,
    theta_entropy: float = 0.45,
    theta_min: float = 0.30,
) -> Dict[str, Any]:
    """Evaluate speculative confidence conjunction."""
    if s1 < theta_min:
        decision = "RERANK_RELEVANCE_FLOOR"
        bypass = False
    elif h_norm > theta_entropy:
        decision = "RERANK_HIGH_ENTROPY"
        bypass = False
    elif delta < theta_gap:
        decision = "RERANK_LOW_GAP"
        bypass = False
    else:
        decision = "FAST_PATH_BYPASS"
        bypass = True

    return {
        "decision": decision,
        "bypass_granted": bypass,
        "marginal_gap": delta,
        "normalized_entropy": h_norm,
        "top1_score": s1,
    }


def ref_cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if (na > 0.0 and nb > 0.0) else 0.0


def ref_convex_blend(s_ce: float, s_rrf: float, lambda_val: float = 0.85) -> float:
    """Convex combination of cross-encoder and RRF scores."""
    return float((lambda_val * s_ce) + ((1.0 - lambda_val) * s_rrf))


def ref_rrf(
    dense: Sequence[Any],
    bm25: Sequence[Any],
    k: int = 60,
    w_dense: float = 1.0,
    w_bm25: float = 1.0,
    dense_scores: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """Reference Reciprocal Rank Fusion implementation with 5-stage tie-breaker."""
    raw_scores: Dict[str, float] = {}
    consensus: Dict[str, int] = {}
    min_ranks: Dict[str, int] = {}
    d_scores: Dict[str, float] = dict(dense_scores or {})

    for r, item in enumerate(dense, start=1):
        did = item[0] if isinstance(item, (tuple, list)) else str(item)
        if isinstance(item, (tuple, list)) and did not in d_scores:
            d_scores[did] = float(item[1])
        raw_scores[did] = raw_scores.get(did, 0.0) + (w_dense / (k + r))
        consensus[did] = consensus.get(did, 0) + 1
        min_ranks[did] = r

    for r, item in enumerate(bm25, start=1):
        did = item[0] if isinstance(item, (tuple, list)) else str(item)
        raw_scores[did] = raw_scores.get(did, 0.0) + (w_bm25 / (k + r))
        consensus[did] = consensus.get(did, 0) + 1
        min_ranks[did] = min(min_ranks.get(did, 999999), r)

    rrf_max = (w_dense / (k + 1.0)) + (w_bm25 / (k + 1.0))
    if rrf_max <= 0.0:
        rrf_max = 1.0

    def sort_key(did: str):
        return (
            -raw_scores[did],
            -consensus[did],
            min_ranks[did],
            -d_scores.get(did, -1.0),
            did,
        )

    sorted_ids = sorted(raw_scores.keys(), key=sort_key)
    return [(did, float(raw_scores[did] / rrf_max)) for did in sorted_ids]


# -----------------------------------------------------------------------------
# Reference In-Memory LRU Semantic Cache
# -----------------------------------------------------------------------------

class ReferenceInMemoryLRUCache:
    """Reference LRU cache supporting exact hash matching and cosine similarity thresholding."""

    def __init__(self, capacity: int = 100, default_ttl: int = 3600):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def get(
        self,
        query_text: str,
        query_vec: np.ndarray,
        threshold: float = 0.92,
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        # 1. Exact string hash check (<0.1ms)
        h = self._hash(query_text)
        if h in self.cache:
            entry = self.cache[h]
            if entry["expire_at"] > now:
                self.cache.move_to_end(h)
                return entry["payload"]
            else:
                del self.cache[h]

        # 2. Vector cosine similarity check
        q = np.asarray(query_vec, dtype=np.float32).flatten()
        norm_q = np.linalg.norm(q)
        if norm_q > 0.0:
            q = q / norm_q

        for k, entry in list(self.cache.items()):
            if entry["expire_at"] <= now:
                del self.cache[k]
                continue

            cached_vec = entry["vec"]
            norm_c = np.linalg.norm(cached_vec)
            if norm_c > 0.0:
                sim = float(np.dot(q, cached_vec / norm_c))
                if sim >= threshold:
                    self.cache.move_to_end(k)
                    return entry["payload"]

        return None

    def set(
        self,
        query_text: str,
        query_vec: np.ndarray,
        payload: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        h = self._hash(query_text)
        if h in self.cache:
            del self.cache[h]
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)

        q = np.asarray(query_vec, dtype=np.float32).flatten()
        self.cache[h] = {
            "query": query_text,
            "vec": q,
            "payload": payload,
            "expire_at": time.time() + ttl,
        }

    def clear(self) -> None:
        self.cache.clear()


# -----------------------------------------------------------------------------
# MOCK CORPUS (Diverse documents for BM25, dense search, and hybrid retrieval)
# -----------------------------------------------------------------------------

MOCK_RAW_DOCS = [
    ("tech_01", "Python asyncio coroutines and event loops provide scalable concurrency."),
    ("tech_02", "Asyncio tasks run cooperatively on a single-threaded asynchronous loop."),
    ("tech_03", "Asynchronous network sockets in Python asyncio yield high throughput."),
    ("tech_04", "FastAPI leverages Starlette and Pydantic for high-performance REST APIs."),
    ("tech_05", "Uvicorn is a lightning-fast ASGI web server implementation using uvloop."),
    ("tech_06", "NumPy provides multidimensional array operations implemented in C."),
    ("tech_07", "Vector similarity search uses dot-product cosine similarity over normalized floats."),
    ("tech_08", "PostgreSQL is an advanced open-source object-relational database system."),
    ("tech_09", "SQLite is a C-language library that implements a small, fast SQL database engine."),
    ("tech_10", "Redis is an in-memory data structure store used as a database, cache, and broker."),
    ("sci_01", "CRISPR Cas9 genome editing guide RNA directs precise DNA modifications."),
    ("sci_02", "Target DNA cleavage by restriction endonuclease enzyme systems."),
    ("sci_03", "Polymerase chain reaction amplifies specific DNA fragments exponentially."),
    ("sci_04", "Protein folding dynamics are modeled using deep learning neural networks."),
    ("sci_05", "Ribosomal translation synthesizes polypeptide chains from mRNA templates."),
    ("sci_06", "Mitochondrial oxidative phosphorylation generates cellular adenosine triphosphate."),
    ("sci_07", "Photosynthetic light reactions convert solar photons into chemical energy."),
    ("sci_08", "Synaptic neurotransmission mediates rapid electrical signaling across neurons."),
    ("sci_09", "Monoclonal antibodies bind specific cell-surface receptor epitopes."),
    ("sci_10", "Thermodynamic free energy landscapes dictate macromolecular conformations."),
    ("fin_01", "Quantitative trading strategies exploit statistical arbitrage and liquidity imbalances."),
    ("fin_02", "Black-Scholes option pricing model calculates theoretical European derivatives."),
    ("fin_03", "Portfolio diversification minimizes idiosyncratic unsystematic financial risk."),
    ("fin_04", "Central bank interest rate decisions influence macroeconomic yield curves."),
    ("fin_05", "High-frequency algorithmic trading requires sub-millisecond execution latencies."),
    ("supp_01", "Our 30-day return policy permits customers to return unworn items in original packaging with receipt for a full monetary refund."),
    ("supp_02", "To initiate a product return, navigate to Order History, click Return Items, and download the prepaid UPS shipping label."),
    ("supp_03", "Refund processing takes 3 to 5 business days after our fulfillment warehouse inspects and approves the returned merchandise."),
    ("supp_04", "Standard shipping delivers within 3 to 5 business days; expedited overnight courier service is available at checkout for additional cost."),
    ("supp_05", "Password reset emails contain a cryptographically signed one-time link valid for exactly 15 minutes from generation."),
    ("ambi_tech_01", "Apple Silicon M3 Max demonstrates unprecedented single-core and multi-core benchmark performance with unified memory architecture."),
    ("ambi_fruit_01", "The Honeycrisp apple is prized for its crisp texture, balanced sweetness, and high dietary fiber content harvested in early autumn orchards."),
    ("ambi_fin_01", "Central banks adjust prime interest rate margins to regulate commercial bank credit liquidity and counter inflationary pressures."),
    ("ambi_geo_01", "Riparian vegetation along the river bank prevents soil erosion and stabilizes sediment during seasonal flooding events."),
    ("api_01", "REST API authentication requires an HTTP Authorization header containing a Bearer token: Authorization: Bearer <JWT_SECRET>."),
    ("api_02", "An HTTP 401 Unauthorized status indicates missing or expired authentication credentials, while 403 Forbidden indicates insufficient privileges."),
    ("api_03", "API rate limiting enforces 100 requests per minute per IP address, returning HTTP 429 Too Many Requests when exceeded."),
    ("legal_01", "Confidential Information encompasses all non-public proprietary data, software source code, trade secrets, and financial forecasts disclosed."),
    ("legal_02", "Each party shall indemnify, defend, and hold harmless the other party against third-party claims arising from gross negligence or willful misconduct."),
    ("legal_03", "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware without regard to conflict of law principles."),
]

MOCK_CORPUS = [
    {
        "id": doc_id,
        "doc_id": f"parent_{doc_id.split('_')[0]}",
        "text": text,
        "metadata": {"category": doc_id.split("_")[0], "char_len": len(text)},
        "embedding": generate_deterministic_vector(text, 384).tolist(),
    }
    for doc_id, text in MOCK_RAW_DOCS
]


# -----------------------------------------------------------------------------
# Pytest Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def tmp_sqlite_path():
    """Create a temporary SQLite database path and clean it up after the test."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="specrag_test_")
    os.close(fd)
    try:
        yield path
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Generate 5 deterministic 384-dimensional unit-normalized test vectors."""
    rng = np.random.RandomState(42)
    raw = rng.randn(5, 384).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


@pytest.fixture
def sample_chunks(sample_embeddings: np.ndarray) -> List[DocumentChunk]:
    """Generate a diverse set of test document chunks with metadata and embeddings."""
    texts = [
        "SpecRAG is a low-latency entropy-gated speculative RAG engine.",
        "PostgreSQL and SQLite provide robust dual-storage backends on native Windows.",
        "Dense semantic vector search leverages normalized dot-product cosine similarity.",
        "Okapi BM25 keyword search is implemented in pure Python with Robertson-Spärck Jones IDF.",
        "Reciprocal Rank Fusion fuses dense and sparse candidate rankings with deterministic tie-breaking.",
    ]
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(
            DocumentChunk(
                id=f"chunk_{i+1}",
                doc_id=f"doc_{i//2 + 1}",
                text=text,
                metadata={"category": "tech", "index": i, "tag": f"t_{i}"},
                embedding=sample_embeddings[i].tolist(),
            )
        )
    return chunks
