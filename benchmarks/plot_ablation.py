"""Generate publication-grade benchmark visualization chart for SpecRAG.

Saves figure to docs/assets/ablation_benchmark.png.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_PATH = os.path.join(PROJECT_ROOT, "benchmarks", "ablation_report.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "docs", "assets", "ablation_benchmark.png")


def generate_benchmark_chart():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    archs = ["Vanilla Bi-Encoder", "Full Cross-Encoder", "SpecRAG (Ours)"]
    p50_latencies = [data["architectures"][a]["p50_latency_ms"] for a in archs]
    ndcgs = [data["architectures"][a]["ndcg_at_10"] for a in archs]
    rerank_pcts = [data["architectures"][a]["rerank_ratio"] * 100 for a in archs]
    compute_saved = [100.0 - r for r in rerank_pcts]

    # Plot styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")

    colors = ["#64748b", "#f97316", "#10b981"]

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        ax.grid(color="#334155", linestyle="--", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # 1. Latency P50 (ms)
    bars1 = axes[0].bar(archs, p50_latencies, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.5)
    axes[0].set_title("P50 Retrieval Latency (Lower is Better)", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_ylabel("Latency (ms)", color="#94a3b8", fontsize=10)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} ms", ha="center", va="bottom", color="#f8fafc", fontweight="bold", fontsize=10)

    # 2. Ranking Quality NDCG@10
    bars2 = axes[1].bar(archs, ndcgs, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.5)
    axes[1].set_title("Ranking Quality NDCG@10 (Higher is Better)", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_ylabel("NDCG@10 Score", color="#94a3b8", fontsize=10)
    axes[1].set_ylim(0.75, 0.90)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha="center", va="bottom", color="#f8fafc", fontweight="bold", fontsize=10)

    # 3. Compute Savings (% Bypass/Saved)
    bars3 = axes[2].bar(archs, compute_saved, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.5)
    axes[2].set_title("Heavy Reranker Compute Saved (%)", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    axes[2].set_ylabel("Compute Cost Reduction (%)", color="#94a3b8", fontsize=10)
    axes[2].set_ylim(0, 100)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{yval:.1f}%", ha="center", va="bottom", color="#f8fafc", fontweight="bold", fontsize=10)

    # Adjust labels
    for ax in axes:
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels(["Vanilla\nBi-Encoder", "Full\nCross-Encoder", "SpecRAG\n(Ours)"], color="#f8fafc", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[OK] Publication chart generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_benchmark_chart()
