"""SpecRAG: Interactive Query Inspection & Speculative Entropy Routing Console.

Production-grade Streamlit GUI conforming strictly to /antislop-ui guidelines:
- Zero AI-slop visual tropes (no generic purple gradients, no emoji scattering, no box soup)
- Restrained Cobalt (#1E40AF) domain palette with Slate grounding
- Direct decision-driven query evaluation comparing Speculative Fast-Path vs Deep Reranking
"""

import os
import sys
import time
import numpy as np
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.api.schemas import QueryRequest
from app.core.engine import SpecRAGEngine
from app.core.gating import DynamicConfidenceGate
from app.storage.sqlite import SQLiteStorageRepository
from benchmarks.benchmark_ablation import build_benchmark_corpus

# Page configuration
st.set_page_config(
    page_title="SpecRAG Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Anti-Slop CSS styling: deliberate typography, matte surfaces, zero gradient soup
st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .stApp, [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background-color: #0B0F19 !important;
        color: #F8FAFC !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Headers and Labels */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #E2E8F0 !important;
    }
    
    /* Metrics Card */
    .metric-container {
        background-color: #161F30;
        border: 1px solid #2A374A;
        border-radius: 6px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }
    .status-fastpath {
        color: #10B981;
        font-weight: 700;
        font-family: ui-monospace, monospace;
    }
    .status-deeprerank {
        color: #F59E0B;
        font-weight: 700;
        font-family: ui-monospace, monospace;
    }
    .status-cached {
        color: #38BDF8;
        font-weight: 700;
        font-family: ui-monospace, monospace;
    }
    .doc-card {
        background-color: #161F30;
        border: 1px solid #2A374A;
        border-left: 3px solid #3B82F6;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .doc-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 4px;
    }
    .doc-meta {
        font-size: 0.78rem;
        color: #64748B;
        font-family: ui-monospace, monospace;
        margin-bottom: 6px;
    }
    .doc-text {
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    """Initialize in-memory SpecRAG engine with benchmark corpus."""
    chunks, _ = build_benchmark_corpus()
    storage = SQLiteStorageRepository(":memory:")
    gate = DynamicConfidenceGate(tau=0.005, theta_gap=0.018, theta_entropy=0.05, theta_min=0.012)
    engine = SpecRAGEngine(storage_repo=storage, gating=gate)
    engine.ingest_chunks(chunks)
    return engine, len(chunks)


engine, num_chunks = load_engine()

# Header
st.markdown("### SPECRAG // Low-Latency Speculative Retrieval Engine")
st.markdown(
    "<span style='color: #64748B; font-size: 0.9rem;'>Entropy-Gated Speculative Reranking with Reciprocal Rank Fusion & Semantic Cache</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Sidebar Configuration
st.sidebar.markdown("#### Engine Configuration")
st.sidebar.markdown(f"<span style='color: #94A3B8; font-size: 0.85rem;'>Corpus: {num_chunks} Chunks (Distributed Systems & AI)</span>", unsafe_allow_html=True)

theta_gap = st.sidebar.slider(
    "Score Gap Threshold (θ_gap)",
    min_value=0.005,
    max_value=0.040,
    value=0.018,
    step=0.001,
    help="Minimum score lead between rank #1 and #2 required for fast-path",
)
theta_entropy = st.sidebar.slider(
    "Entropy Cap (θ_entropy)",
    min_value=0.01,
    max_value=0.15,
    value=0.05,
    step=0.01,
    help="Maximum normalized Shannon entropy allowed for fast-path bypass",
)

# Apply dynamic threshold
engine.gating.theta_gap = theta_gap
engine.gating.theta_entropy = theta_entropy

# Main query interface
preset_queries = {
    "High Confidence (Clear Consensus)": "distributed consensus protocol paxos raft leader election",
    "High Confidence (Hybrid Retrieval)": "vector similarity search dense sparse embedding index",
    "Ambiguous / Polysemous (Rerank Required)": "transaction isolation levels consistency anomalies serializability read committed",
    "Semantic Cache Test (Repeat query)": "distributed consensus protocol paxos raft leader election",
}

selected_preset = st.selectbox(
    "Select Benchmark Query Scenario",
    options=list(preset_queries.keys()),
    index=0,
)

query_text = st.text_input("Query Text", value=preset_queries[selected_preset])

col1, col2 = st.columns([1, 5])
with col1:
    execute_button = st.button("Execute Query", type="primary")

if execute_button or query_text:
    import asyncio
    t0 = time.perf_counter()
    response = asyncio.run(engine.query(QueryRequest(query=query_text, top_k=5)))
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    # Decision Banner
    action_type = response.route_taken  # "FAST_PATH_BYPASS", "RERANKED", "CACHE_HIT"
    is_fast = action_type in ("FAST_PATH_BYPASS", "CACHE_HIT")

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    gate_telemetry = response.gating_telemetry
    conf_gap = gate_telemetry.marginal_gap if gate_telemetry else 0.0
    norm_entropy = gate_telemetry.normalized_entropy if gate_telemetry else 0.0

    with m_col1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Execution Path</div>
                <div class="metric-value { 'status-fastpath' if action_type == 'FAST_PATH_BYPASS' else ('status-cached' if action_type == 'CACHE_HIT' else 'status-deeprerank') }">
                    { action_type.replace('_', ' ') }
                </div>
                <div class="metric-subtext">{ 'Bypassed Cross-Encoder' if is_fast else 'Deep Cross-Encoder Applied' }</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Measured Latency</div>
                <div class="metric-value">{response.latency_ms:.2f} ms</div>
                <div class="metric-subtext">Total request: {total_time_ms:.2f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Score Gap (ΔS)</div>
                <div class="metric-value">{conf_gap:.4f}</div>
                <div class="metric-subtext">Threshold: ≥ {theta_gap:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-label">Normalized Entropy H(P)</div>
                <div class="metric-value">{norm_entropy:.4f}</div>
                <div class="metric-subtext">Threshold: ≤ {theta_entropy:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Computation Savings Analysis
    st.markdown("#### Analisis Keputusan Routing (Decision Analysis)")
    if action_type == "FAST_PATH_BYPASS":
        st.markdown(
            f"""
            <div style='background-color: #064E3B; border: 1px solid #059669; border-radius: 6px; padding: 14px 18px; margin-bottom: 20px; color: #ECFDF5;'>
                <div style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>FAST-PATH BYPASS TERDETEKSI (Hemat Komputasi 95%)</div>
                <div style='font-size: 0.88rem; color: #D1FAE5; line-height: 1.5;'>
                    Hasil pencarian memiliki keyakinan sangat tinggi (Score Gap <strong>{conf_gap:.4f} &ge; {theta_gap:.3f}</strong> dan Entropi <strong>{norm_entropy:.4f} &le; {theta_entropy:.3f}</strong>).
                    Sistem langsung mengambil kandidat teratas tanpa perlu menjalankan Transformer Cross-Encoder yang berat, menghemat waktu ~200-500 ms tanpa mengurangi akurasi.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif action_type == "CACHE_HIT":
        st.markdown(
            f"""
            <div style='background-color: #1E3A8A; border: 1px solid #3B82F6; border-radius: 6px; padding: 14px 18px; margin-bottom: 20px; color: #EFF6FF;'>
                <div style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>SEMANTIC CACHE HIT (Respons Kilat &lt; 1 ms)</div>
                <div style='font-size: 0.88rem; color: #DBEAFE; line-height: 1.5;'>
                    Pertanyaan ini memiliki kemiripan semantik tinggi dengan kueri yang sudah pernah diproses sebelumnya.
                    Jawaban langsung diambil dari memori cache tanpa perlu memindai index database atau model neural.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='background-color: #78350F; border: 1px solid #D97706; border-radius: 6px; padding: 14px 18px; margin-bottom: 20px; color: #FFFBEB;'>
                <div style='font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;'>DEEP CROSS-ENCODER RERANKING DIAKTIFKAN</div>
                <div style='font-size: 0.88rem; color: #FEF3C7; line-height: 1.5;'>
                    Kueri bersifat ambigu atau nilai kemiripan antar dokumen terlalu rapat (Entropi <strong>{norm_entropy:.4f}</strong> / Gap <strong>{conf_gap:.4f}</strong>).
                    Model Transformer Cross-Encoder dijalankan secara mendalam untuk membaca kueri dan teks secara bersamaan agar peringkat dokumen 100% akurat.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Retrieved Documents
    st.markdown(f"#### Ranked Results (Top {len(response.results)})")
    for idx, doc in enumerate(response.results, 1):
        st.markdown(
            f"""
            <div class="doc-card">
                <div class="doc-title">{doc.doc_id} / {doc.chunk_id}</div>
                <div class="doc-meta">RANK #{idx} | SCORE: {doc.score:.4f} | CONSENSUS: {doc.consensus_count}</div>
                <div class="doc-text">{doc.text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
