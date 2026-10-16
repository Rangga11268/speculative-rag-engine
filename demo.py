"""Interactive 60-second Demonstration for SpecRAG Engine.

Showcases:
1. Fast-Path Bypass for high-confidence queries (< 1ms)
2. Deliberate Deep-Path Cross-Encoder Reranking for ambiguous queries (~3-5ms)
3. Sub-millisecond Semantic Cache hits for repeated/paraphrased queries (< 0.5ms)
"""

import asyncio
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.api.schemas import QueryRequest
from app.core.engine import SpecRAGEngine
from app.storage.sqlite import SQLiteStorageRepository
from benchmarks.benchmark_ablation import build_benchmark_corpus

# ANSI styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def run_demo():
    print(f"\n{BOLD}{CYAN}{'=' * 78}{RESET}")
    print(f"{BOLD}{CYAN}  SpecRAG: Entropy-Gated Speculative RAG Engine — Live Showcase{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 78}{RESET}\n")

    # Ingest benchmark domain corpus
    print(f"{YELLOW}[*] Initializing SpecRAG in-memory engine with realistic domain corpus...{RESET}")
    chunks, _ = build_benchmark_corpus()
    storage = SQLiteStorageRepository(":memory:")
    # Calibrate gate for RRF reciprocal score distribution (where scores ~ 1/(60+rank))
    from app.core.gating import DynamicConfidenceGate
    gate = DynamicConfidenceGate(tau=0.005, theta_gap=0.018, theta_entropy=0.05, theta_min=0.012)
    engine = SpecRAGEngine(storage_repo=storage, gating=gate)
    engine.ingest_chunks(chunks)
    print(f"{GREEN}[OK] Ingested {len(chunks)} document chunks across Distributed Systems, DB, RAG & ML.{RESET}\n")

    queries = [
        (
            "Scenario 1: High-Confidence Query (Dominant Candidate)",
            "distributed consensus protocol paxos raft leader election",
            "Clear match: High Score Gap ΔS & Low Entropy H -> Fast-Path Bypass (< 1ms)",
        ),
        (
            "Scenario 2: Ambiguous Query (High Entropy / Competing Chunks)",
            "Shannon entropy uncertainty information in probability distribution",
            "Competing candidates: High Entropy H -> Trigger Heavy Cross-Encoder",
        ),
        (
            "Scenario 3: Repeated Query (Semantic Cache)",
            "distributed consensus protocol paxos raft leader election",
            "Cache Hit: Zero-compute cosine lookup -> Instant Return (< 0.5ms)",
        ),
    ]

    for title, q_text, expectation in queries:
        print(f"{BOLD}{MAGENTA}--- {title} ---{RESET}")
        print(f"  {BOLD}Query:{RESET} \"{q_text}\"")
        print(f"  {BOLD}Behavior:{RESET} {expectation}")

        resp = await engine.query(QueryRequest(query=q_text, top_k=3))

        print(f"  {BOLD}Route Taken:{RESET} ", end="")
        if resp.route_taken == "CACHE_HIT":
            print(f"{GREEN}{BOLD}[⚡ CACHE HIT] (Reranker & Retriever Bypassed){RESET}")
        elif "FAST" in resp.route_taken or (resp.gating_telemetry and resp.gating_telemetry.bypass_granted):
            print(f"{CYAN}{BOLD}[🚀 FAST-PATH BYPASS] (Reranker Skipped via Entropy Gate){RESET}")
        else:
            print(f"{YELLOW}{BOLD}[🔬 CROSS-ENCODER PATH] (Heavy Reranker Triggered){RESET}")

        if resp.gating_telemetry:
            tele = resp.gating_telemetry
            print(
                f"  {BOLD}Gating Telemetry:{RESET} Marginal Gap ΔS={tele.marginal_gap:.4f} | "
                f"Entropy H={tele.entropy:.4f} (Norm: {tele.normalized_entropy:.2f}) | "
                f"Bypass Granted: {tele.bypass_granted}"
            )

        print(f"  {BOLD}Query Latency:{RESET} {GREEN}{resp.latency_ms:.2f} ms{RESET}")
        print(f"  {BOLD}Top Result:{RESET} [{resp.results[0].chunk_id}] {resp.results[0].text[:85]}...\n")

    print(f"{BOLD}{CYAN}{'=' * 78}{RESET}")
    print(f"{GREEN}{BOLD}[OK] Showcase Complete! Empirical demonstration of dual-speed speculative execution.{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 78}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
