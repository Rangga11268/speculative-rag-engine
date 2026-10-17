"""Unit and integration tests for HybridRetriever."""

import asyncio
import numpy as np
import pytest

from app.api.schemas import DocumentChunk
from app.retrieval.bm25 import OkapiBM25Index
from app.retrieval.hybrid import HybridRetriever
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.vector_store import DenseVectorIndex


class TestHybridRetriever:
    """Test suite for HybridRetriever parallel retrieval and rank fusion."""

    @pytest.fixture
    def setup_retriever(self, tmp_sqlite_path, sample_chunks):
        dim = 384
        repo = SQLiteStorageRepository(db_path=tmp_sqlite_path, dimension=dim)
        v_idx = DenseVectorIndex(dimension=dim)
        bm_idx = OkapiBM25Index()

        retriever = HybridRetriever(
            vector_index=v_idx,
            bm25_index=bm_idx,
            storage_repo=repo,
            default_rrf_k=60,
        )

        # Index the sample corpus
        retriever.index_chunks(sample_chunks, persist=True)
        yield retriever
        repo.close()

    def test_sync_retrieval_and_metadata_resolution(self, setup_retriever, sample_chunks):
        retriever = setup_retriever
        query_vec = sample_chunks[0].embedding
        query_text = "speculative RAG engine low-latency"

        results = retriever.retrieve_sync(
            query_text=query_text,
            query_vector=query_vec,
            top_k=3,
        )

        assert len(results) == 3
        # Rank 1 should match chunk_1 due to high dense similarity and exact keywords
        assert results[0].chunk_id == "chunk_1"
        assert results[0].text == sample_chunks[0].text
        assert results[0].metadata == sample_chunks[0].metadata
        assert results[0].score > 0.0
        assert results[0].score <= 1.0

    def test_async_parallel_retrieval(self, setup_retriever, sample_chunks):
        retriever = setup_retriever
        query_vec = sample_chunks[1].embedding
        query_text = "PostgreSQL and SQLite dual-storage"

        results = asyncio.run(
            retriever.retrieve(
                query_text=query_text,
                query_vector=query_vec,
                top_k=2,
                dense_weight=1.0,
                bm25_weight=1.0,
            )
        )

        assert len(results) == 2
        # chunk_2 should be ranked top
        assert results[0].chunk_id == "chunk_2"
        assert "dual-storage" in results[0].text
        assert results[0].consensus_count >= 1

    def test_weight_ablation_bm25_only(self, setup_retriever, sample_chunks):
        """When dense_weight=0, ranking is driven entirely by BM25 keyword matching."""
        retriever = setup_retriever
        # Deliberately use query_vector for chunk 1, but text for chunk 4 ("Okapi BM25")
        query_vec = sample_chunks[0].embedding
        query_text = "Okapi BM25 Robertson-Spärck"

        results = retriever.retrieve_sync(
            query_text=query_text,
            query_vector=query_vec,
            top_k=3,
            dense_weight=0.0,
            bm25_weight=1.0,
        )

        assert len(results) >= 1
        assert results[0].chunk_id == "chunk_4"

    def test_weight_ablation_dense_only(self, setup_retriever, sample_chunks):
        """When bm25_weight=0, ranking is driven entirely by dense vector similarity."""
        retriever = setup_retriever
        # Deliberately use query_vector for chunk 3 ("Dense semantic vector search")
        # and unrelated text
        query_vec = sample_chunks[2].embedding
        query_text = "completely unrelated text about baking pastries"

        results = retriever.retrieve_sync(
            query_text=query_text,
            query_vector=query_vec,
            top_k=3,
            dense_weight=1.0,
            bm25_weight=0.0,
        )

        assert len(results) >= 1
        assert results[0].chunk_id == "chunk_3"
