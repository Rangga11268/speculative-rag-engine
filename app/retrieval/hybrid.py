"""Hybrid retriever orchestrating parallel dense vector search and sparse BM25 search."""

import asyncio
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from app.api.schemas import DocumentChunk, RetrievedCandidate
from app.retrieval.bm25 import OkapiBM25Index
from app.retrieval.rrf import reciprocal_rank_fusion
from app.storage.base import StorageRepository
from app.storage.vector_store import DenseVectorIndex


class HybridRetriever:
    """Orchestrates parallel dense semantic and sparse lexical search fused via RRF."""

    def __init__(
        self,
        vector_index: DenseVectorIndex,
        bm25_index: OkapiBM25Index,
        storage_repo: Optional[StorageRepository] = None,
        default_rrf_k: int = 60,
    ):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.storage_repo = storage_repo
        self.default_rrf_k = default_rrf_k

    def retrieve_dense(
        self,
        query_vector: Union[np.ndarray, Sequence[float]],
        top_k: int = 10,
        namespace: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Synchronously execute dense cosine vector search."""
        return self.vector_index.search(query_vector, top_k=top_k, namespace=namespace)

    def retrieve_sparse(
        self,
        query_text: str,
        top_k: int = 10,
        namespace: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Synchronously execute Okapi BM25 keyword search."""
        return self.bm25_index.search(query_text, top_k=top_k, namespace=namespace)

    def _resolve_candidate_details(
        self, candidates: List[RetrievedCandidate]
    ) -> List[RetrievedCandidate]:
        """Populate text and metadata for candidates from the storage repository."""
        if not self.storage_repo or not candidates:
            return candidates

        chunk_ids = [c.chunk_id for c in candidates if not c.text]
        if not chunk_ids:
            return candidates

        stored_chunks: Dict[str, DocumentChunk] = self.storage_repo.get_chunks_by_ids(chunk_ids)
        for c in candidates:
            if c.chunk_id in stored_chunks:
                chunk = stored_chunks[c.chunk_id]
                c.text = chunk.text
                c.doc_id = chunk.doc_id
                if not c.metadata:
                    c.metadata = chunk.metadata

        return candidates

    async def retrieve(
        self,
        query_text: str,
        query_vector: Union[np.ndarray, Sequence[float]],
        top_k: int = 10,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> List[RetrievedCandidate]:
        """Asynchronously execute dense and sparse retrieval in parallel via asyncio.gather."""
        k_smooth = rrf_k if rrf_k is not None else self.default_rrf_k
        fetch_k = max(top_k * 2, 20)

        # Run both searches in parallel thread pool tasks to prevent blocking event loop
        dense_task = asyncio.to_thread(self.retrieve_dense, query_vector, fetch_k, namespace)
        bm25_task = asyncio.to_thread(self.retrieve_sparse, query_text, fetch_k, namespace)

        dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)

        # Fuse rankings using Reciprocal Rank Fusion + 5-stage tie-breaker
        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=k_smooth,
            w_dense=dense_weight,
            w_bm25=bm25_weight,
            top_k=top_k,
        )

        return await asyncio.to_thread(self._resolve_candidate_details, fused_candidates)

    def retrieve_sync(
        self,
        query_text: str,
        query_vector: Union[np.ndarray, Sequence[float]],
        top_k: int = 10,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> List[RetrievedCandidate]:
        """Synchronously execute retrieval and fusion (convenience wrapper)."""
        k_smooth = rrf_k if rrf_k is not None else self.default_rrf_k
        fetch_k = max(top_k * 2, 20)

        dense_results = self.retrieve_dense(query_vector, fetch_k, namespace=namespace)
        bm25_results = self.retrieve_sparse(query_text, fetch_k, namespace=namespace)

        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=k_smooth,
            w_dense=dense_weight,
            w_bm25=bm25_weight,
            top_k=top_k,
        )

        return self._resolve_candidate_details(fused_candidates)

    def index_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        persist: bool = True,
    ) -> int:
        """Add chunks to vector index, BM25 index, and optionally persist to storage."""
        if not chunks:
            return 0

        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        nss = [c.namespace for c in chunks]

        # 1. Update BM25 index
        self.bm25_index.add(ids, texts, namespaces=nss)

        # 2. Update Vector index (for chunks that have precomputed embeddings)
        embedded_ids = []
        vectors = []
        embedded_nss = []
        for c in chunks:
            if c.embedding is not None and len(c.embedding) > 0:
                embedded_ids.append(c.id)
                vectors.append(c.embedding)
                embedded_nss.append(c.namespace)

        if embedded_ids:
            self.vector_index.add(
                embedded_ids,
                np.array(vectors, dtype=np.float32),
                namespaces=embedded_nss,
            )

        # 3. Persist to storage repository if requested
        if persist and self.storage_repo is not None:
            self.storage_repo.add_chunks(list(chunks))

        return len(chunks)
