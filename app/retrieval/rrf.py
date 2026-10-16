"""Reciprocal Rank Fusion (RRF) with theoretical max normalization and 5-stage deterministic tie-breaker."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from app.api.schemas import RetrievedCandidate


def compute_rrf_scores(
    dense_results: Sequence[Union[Tuple[str, float], RetrievedCandidate]],
    bm25_results: Sequence[Union[Tuple[str, float], RetrievedCandidate]],
    k: int = 60,
    w_dense: float = 1.0,
    w_bm25: float = 1.0,
) -> List[Dict[str, Any]]:
    """Compute raw RRF scores, consensus count, best rank, and sorted ordering.

    Returns a list of dicts with:
    - chunk_id
    - rrf_score
    - normalized_score
    - dense_score
    - bm25_score
    - consensus_count
    - min_rank
    - doc_id (if available)
    - text (if available)
    - metadata (if available)
    """
    if k <= 0:
        raise ValueError(f"RRF smoothing parameter k must be positive, got {k}")

    rrf_raw: Dict[str, float] = {}
    dense_scores: Dict[str, float] = {}
    bm25_scores: Dict[str, float] = {}
    best_ranks: Dict[str, int] = {}
    consensus: Dict[str, int] = {}
    doc_ids: Dict[str, str] = {}
    texts: Dict[str, str] = {}
    metas: Dict[str, Dict[str, Any]] = {}

    dense_seen = set()
    # 1. Process Dense Ranks (1-based rank position)
    for rank_idx, item in enumerate(dense_results, start=1):
        if isinstance(item, RetrievedCandidate):
            cid = item.chunk_id
            d_score = item.dense_score if item.dense_score is not None else item.score
            doc_ids[cid] = item.doc_id
            texts[cid] = item.text
            metas[cid] = item.metadata
        elif isinstance(item, (tuple, list)):
            cid = str(item[0])
            d_score = float(item[1])
        else:
            cid = str(item)
            d_score = 0.0

        dense_scores[cid] = d_score
        rrf_raw[cid] = rrf_raw.get(cid, 0.0) + (w_dense / (k + rank_idx))
        best_ranks[cid] = min(best_ranks.get(cid, 999999), rank_idx)
        if cid not in dense_seen:
            dense_seen.add(cid)
            consensus[cid] = consensus.get(cid, 0) + 1

    bm25_seen = set()
    # 2. Process BM25 Ranks (1-based rank position)
    for rank_idx, item in enumerate(bm25_results, start=1):
        if isinstance(item, RetrievedCandidate):
            cid = item.chunk_id
            b_score = item.bm25_score if item.bm25_score is not None else item.score
            if cid not in doc_ids:
                doc_ids[cid] = item.doc_id
                texts[cid] = item.text
                metas[cid] = item.metadata
        elif isinstance(item, (tuple, list)):
            cid = str(item[0])
            b_score = float(item[1])
        else:
            cid = str(item)
            b_score = 0.0

        bm25_scores[cid] = b_score
        rrf_raw[cid] = rrf_raw.get(cid, 0.0) + (w_bm25 / (k + rank_idx))
        best_ranks[cid] = min(best_ranks.get(cid, 999999), rank_idx)
        if cid not in bm25_seen:
            bm25_seen.add(cid)
            consensus[cid] = consensus.get(cid, 0) + 1

    # Theoretical maximum score for global normalization into [0, 1]
    rrf_max = (w_dense / (k + 1.0)) + (w_bm25 / (k + 1.0))
    if rrf_max <= 0.0:
        rrf_max = 1.0

    # 5-Stage Deterministic Tie-Breaking Key:
    # 1. -rrf_raw[cid] (higher RRF score ranks first)
    # 2. -consensus[cid] (retrieved by both modalities beats single modality)
    # 3. best_ranks[cid] (best individual rank position, lower integer is better)
    # 4. -dense_score (fine-grained continuous cosine signal; defaults to -1.0)
    # 5. cid (lexicographical string comparison for 100% deterministic reproducibility)
    def tie_break_key(cid: str) -> Tuple[float, int, int, float, str]:
        d_score = dense_scores.get(cid, -1.0)
        return (
            -rrf_raw[cid],
            -consensus[cid],
            best_ranks[cid],
            -d_score,
            cid,
        )

    sorted_cids = sorted(rrf_raw.keys(), key=tie_break_key)

    output = []
    for cid in sorted_cids:
        raw_s = rrf_raw[cid]
        norm_s = float(raw_s / rrf_max)
        output.append(
            {
                "chunk_id": cid,
                "doc_id": doc_ids.get(cid, cid),
                "text": texts.get(cid, ""),
                "score": norm_s,
                "raw_rrf_score": raw_s,
                "dense_score": dense_scores.get(cid),
                "bm25_score": bm25_scores.get(cid),
                "consensus_count": consensus[cid],
                "min_rank": best_ranks[cid],
                "metadata": metas.get(cid, {}),
            }
        )

    return output


def reciprocal_rank_fusion(
    dense_results: Sequence[Union[Tuple[str, float], RetrievedCandidate]],
    bm25_results: Sequence[Union[Tuple[str, float], RetrievedCandidate]],
    k: int = 60,
    w_dense: float = 1.0,
    w_bm25: float = 1.0,
    top_k: Optional[int] = None,
) -> List[RetrievedCandidate]:
    """Fuse dense and BM25 rank lists using RRF with deterministic 5-stage tie-breaking.

    Returns a list of RetrievedCandidate instances sorted by score descending.
    """
    fused_dicts = compute_rrf_scores(
        dense_results=dense_results,
        bm25_results=bm25_results,
        k=k,
        w_dense=w_dense,
        w_bm25=w_bm25,
    )

    if top_k is not None:
        if top_k <= 0:
            return []
        fused_dicts = fused_dicts[:top_k]

    candidates = []
    for d in fused_dicts:
        candidate = RetrievedCandidate(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            text=d["text"],
            score=d["score"],
            dense_score=d["dense_score"],
            bm25_score=d["bm25_score"],
            consensus_count=d["consensus_count"],
            metadata=d["metadata"],
        )
        candidates.append(candidate)

    return candidates
