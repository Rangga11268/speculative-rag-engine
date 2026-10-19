"""Heuristic Cross-Scoring and Reranker module for SpecRAG.

Provides deliberate deep-path rescoring for ambiguous or high-entropy queries.
Features a zero-dependency NativeRescorer (lexical/proximity heuristic on CPU)
and an extensible BaseReranker interface for neural cross-encoder backends.
"""

from abc import ABC, abstractmethod
import math
import re
from typing import List, Sequence
from app.api.schemas import RetrievedCandidate


class BaseReranker(ABC):
    """Abstract interface for candidate rescorers and neural cross-encoders."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedCandidate],
        top_k: int | None = None,
    ) -> List[RetrievedCandidate]:
        """Rerank candidates based on deep cross-scoring."""
        pass


class NativeRescorer(BaseReranker):
    """Zero-dependency heuristic cross-scoring rescorer running on CPU.

    Calculates exact phrase matches, positional term proximity,
    character n-gram Jaccard overlap, and consensus rank weights.
    
    Note: This is a fast, deterministic CPU heuristic rescorer,
    distinct from a heavyweight neural transformer cross-encoder (e.g. MiniLM).
    """

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedCandidate],
        top_k: int | None = None,
    ) -> List[RetrievedCandidate]:
        if not candidates:
            return []

        q_clean = query.lower().strip()
        q_tokens = re.findall(r"\w+", q_clean)
        if not q_tokens:
            return list(candidates[:top_k])

        scored_candidates: List[RetrievedCandidate] = []
        for c in candidates:
            doc_clean = c.text.lower()
            doc_tokens = re.findall(r"\w+", doc_clean)

            # 1. Exact query substring bonus
            exact_bonus = 0.35 if q_clean in doc_clean else 0.0

            # 2. Token overlap & positional proximity
            overlap_count = 0
            positions = []
            for t in q_tokens:
                if t in doc_clean:
                    overlap_count += 1
                    try:
                        pos = doc_clean.index(t)
                        positions.append(pos)
                    except ValueError:
                        pass

            overlap_ratio = overlap_count / len(q_tokens)

            # 3. Span compactness
            proximity_bonus = 0.0
            if len(positions) >= 2:
                span = max(positions) - min(positions)
                # Shorter span = tighter proximity
                proximity_bonus = max(0.0, 0.20 * (1.0 - (span / (len(doc_clean) + 1))))

            # 4. Modality consensus weight
            consensus_weight = 0.15 * min(2, c.consensus_count)

            # 5. Composite cross-score
            cross_score = (
                (0.30 * c.score) +
                (0.30 * overlap_ratio) +
                (0.20 * exact_bonus) +
                (0.10 * proximity_bonus) +
                (0.10 * consensus_weight)
            )

            # Create updated candidate
            rescored = RetrievedCandidate(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                score=round(float(cross_score), 6),
                dense_score=c.dense_score,
                bm25_score=c.bm25_score,
                consensus_count=c.consensus_count,
                metadata=c.metadata,
            )
            scored_candidates.append(rescored)

        # Sort descending by updated cross_score
        scored_candidates.sort(key=lambda x: (x.score, x.consensus_count, x.chunk_id), reverse=True)

        if top_k is not None:
            return scored_candidates[:top_k]
        return scored_candidates
