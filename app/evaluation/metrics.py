"""Ragas-compatible retrieval evaluation metrics implemented from scratch.

Implements the same conceptual metrics as the Ragas evaluation framework
(https://docs.ragas.io) without requiring the ragas library:

- context_precision   : Fraction of retrieved chunks that are relevant (Precision@K)
- context_recall      : Fraction of relevant ground-truth items retrieved (Recall@K)
- answer_relevancy    : Cosine similarity between query embedding and the
                        mean embedding of retrieved contexts (embedding-based proxy)

Using embeddings as the answer relevancy proxy is equivalent to Ragas' own
``answer_relevancy`` metric when no LLM is available for question-generation.
This approach is grounded in the literature on embedding-based evaluation
(e.g., BERTScore, SentEval).

Reference formulas:
    context_precision   = |relevant ∩ retrieved| / |retrieved|
    context_recall      = |relevant ∩ retrieved| / |relevant|
    answer_relevancy    = cos(q_vec, mean(ctx_vecs))   ∈ [−1, 1]
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Sequence
import numpy as np


@dataclass
class EvaluationResult:
    """Holds per-query and aggregate Ragas-compatible metrics."""

    # Per-query lists
    context_precision_scores: List[float] = field(default_factory=list)
    context_recall_scores: List[float] = field(default_factory=list)
    answer_relevancy_scores: List[float] = field(default_factory=list)

    # Aggregate stats (populated after calling .aggregate())
    mean_context_precision: float = 0.0
    mean_context_recall: float = 0.0
    mean_answer_relevancy: float = 0.0
    composite_score: float = 0.0   # Harmonic mean of the three metrics
    n_queries: int = 0

    def aggregate(self) -> "EvaluationResult":
        """Compute aggregate statistics from per-query score lists."""
        self.n_queries = len(self.context_precision_scores)
        if self.n_queries == 0:
            return self

        self.mean_context_precision = float(np.mean(self.context_precision_scores))
        self.mean_context_recall = float(np.mean(self.context_recall_scores))
        self.mean_answer_relevancy = float(np.mean(self.answer_relevancy_scores))

        # Composite = harmonic mean of three metrics (avoids reward hacking any single metric)
        metrics = [self.mean_context_precision, self.mean_context_recall, self.mean_answer_relevancy]
        if all(m > 0 for m in metrics):
            self.composite_score = len(metrics) / sum(1.0 / m for m in metrics)
        else:
            self.composite_score = 0.0

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to flat dictionary for MLflow logging."""
        return {
            "ragas_context_precision": round(self.mean_context_precision, 4),
            "ragas_context_recall": round(self.mean_context_recall, 4),
            "ragas_answer_relevancy": round(self.mean_answer_relevancy, 4),
            "ragas_composite_score": round(self.composite_score, 4),
            "n_queries_evaluated": self.n_queries,
        }


class RagasCompatibleEvaluator:
    """Evaluates RAG retrieval quality using Ragas-compatible metrics without LLM dependency.

    Implements Context Precision, Context Recall, and Embedding-based Answer Relevancy.
    Drop-in conceptual equivalent of the Ragas evaluation framework for environments
    where an LLM judge is unavailable or undesirable.
    """

    def __init__(self, embedder=None):
        """
        Args:
            embedder: Optional DenseEmbedder instance for answer_relevancy computation.
                      If None, answer_relevancy will be skipped (returns 0.0).
        """
        self.embedder = embedder

    def _normalize(self, vec: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr

    def context_precision(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: Optional[int] = None,
    ) -> float:
        """Precision@K: fraction of top-K retrieved chunks that are relevant.

        context_precision = |relevant ∩ retrieved[:k]| / min(k, |retrieved|)
        """
        top_k = retrieved_ids[:k] if k else retrieved_ids
        if not top_k:
            return 0.0
        hits = sum(1 for cid in top_k if cid in relevant_ids)
        return hits / len(top_k)

    def context_recall(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: Optional[int] = None,
    ) -> float:
        """Recall@K: fraction of relevant ground-truth chunks that were retrieved.

        context_recall = |relevant ∩ retrieved[:k]| / |relevant|
        """
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k]) if k else set(retrieved_ids)
        hits = len(top_k.intersection(relevant_ids))
        return hits / len(relevant_ids)

    def answer_relevancy(
        self,
        query: str,
        retrieved_texts: List[str],
    ) -> float:
        """Embedding-based answer relevancy: cosine similarity between query and mean context.

        answer_relevancy ≈ cos(q_vec, mean(context_vecs))

        This is the embedding-based proxy for Ragas' answer_relevancy metric,
        which normally requires an LLM to generate back-questions from the answer.
        """
        if self.embedder is None or not retrieved_texts:
            return 0.0

        q_vec = self._normalize(self.embedder.embed_text(query))
        ctx_vecs = np.vstack([
            self._normalize(self.embedder.embed_text(t))
            for t in retrieved_texts[:5]  # Cap at 5 contexts to keep it fast
        ])
        mean_ctx = ctx_vecs.mean(axis=0)
        norm = float(np.linalg.norm(mean_ctx))
        if norm > 1e-12:
            mean_ctx /= norm

        sim = float(np.dot(q_vec, mean_ctx))
        # Remap from [-1, 1] → [0, 1] for interpretability
        return (sim + 1.0) / 2.0

    def evaluate_batch(
        self,
        queries: List[str],
        retrieved_ids_per_query: List[List[str]],
        ground_truth_ids_per_query: List[Set[str]],
        retrieved_texts_per_query: Optional[List[List[str]]] = None,
        k: int = 10,
    ) -> EvaluationResult:
        """Evaluate a batch of query results and return aggregated metrics.

        Args:
            queries: List of query strings.
            retrieved_ids_per_query: List of retrieved chunk_id lists (ordered by rank).
            ground_truth_ids_per_query: List of sets of ground-truth relevant chunk_ids.
            retrieved_texts_per_query: Optional texts for answer_relevancy computation.
            k: Cutoff rank for precision and recall.

        Returns:
            EvaluationResult with per-query scores and aggregated statistics.
        """
        result = EvaluationResult()

        for i, (query, r_ids, gt_ids) in enumerate(
            zip(queries, retrieved_ids_per_query, ground_truth_ids_per_query)
        ):
            result.context_precision_scores.append(
                self.context_precision(r_ids, gt_ids, k=k)
            )
            result.context_recall_scores.append(
                self.context_recall(r_ids, gt_ids, k=k)
            )

            if retrieved_texts_per_query and i < len(retrieved_texts_per_query):
                ar = self.answer_relevancy(query, retrieved_texts_per_query[i])
            else:
                ar = 0.0
            result.answer_relevancy_scores.append(ar)

        return result.aggregate()
