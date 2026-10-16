"""Unit and property tests for DenseVectorIndex."""

import concurrent.futures
import numpy as np
import pytest

from app.storage.vector_store import DenseVectorIndex


class TestDenseVectorIndex:
    """Test suite for in-memory DenseVectorIndex."""

    def test_empty_index_search(self):
        index = DenseVectorIndex(dimension=4)
        assert index.size() == 0
        results = index.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_exact_cosine_similarity(self):
        index = DenseVectorIndex(dimension=4)
        ids = ["identical", "orthogonal", "opposite"]
        # Dimension = 4
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],   # identical to query
                [0.0, 1.0, 0.0, 0.0],   # orthogonal to query
                [-1.0, 0.0, 0.0, 0.0],  # opposite to query
            ],
            dtype=np.float32,
        )
        index.add(ids, vectors)
        assert index.size() == 3

        query = [1.0, 0.0, 0.0, 0.0]
        results = index.search(query, top_k=3)

        assert len(results) == 3
        # Rank 1: identical
        assert results[0][0] == "identical"
        assert pytest.approx(results[0][1], abs=1e-5) == 1.0

        # Rank 2: orthogonal
        assert results[1][0] == "orthogonal"
        assert pytest.approx(results[1][1], abs=1e-5) == 0.0

        # Rank 3: opposite
        assert results[2][0] == "opposite"
        assert pytest.approx(results[2][1], abs=1e-5) == -1.0

    def test_dimension_validation(self):
        index = DenseVectorIndex(dimension=384)
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            index.add(["bad_dim"], np.zeros((1, 128), dtype=np.float32))

        with pytest.raises(ValueError, match="Query vector dimension mismatch"):
            index.search(np.zeros(128, dtype=np.float32))

        with pytest.raises(ValueError, match="Count mismatch"):
            index.add(["id1", "id2"], np.zeros((1, 384), dtype=np.float32))

    def test_unnormalized_vectors_are_normalized(self):
        index = DenseVectorIndex(dimension=2)
        # Vector with norm = 5.0
        index.add(["v1"], [[3.0, 4.0]])
        # Query with norm = 10.0
        results = index.search([6.0, 8.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == "v1"
        # Cosine similarity between collinear vectors must be 1.0
        assert pytest.approx(results[0][1], abs=1e-5) == 1.0

    def test_update_existing_id(self):
        index = DenseVectorIndex(dimension=2)
        index.add(["id_a"], [[1.0, 0.0]])
        assert index.size() == 1

        # Query aligned with [0, 1]
        results = index.search([0.0, 1.0], top_k=1)
        assert pytest.approx(results[0][1], abs=1e-5) == 0.0

        # Update id_a to [0, 1]
        index.add(["id_a"], [[0.0, 1.0]])
        assert index.size() == 1

        results = index.search([0.0, 1.0], top_k=1)
        assert results[0][0] == "id_a"
        assert pytest.approx(results[0][1], abs=1e-5) == 1.0

    def test_remove_ids(self):
        index = DenseVectorIndex(dimension=3)
        index.add(["a", "b", "c"], np.eye(3, dtype=np.float32))
        assert index.size() == 3

        removed = index.remove(["b", "nonexistent"])
        assert removed == 1
        assert index.size() == 2

        results = index.search([0.0, 1.0, 0.0], top_k=5)
        res_ids = [r[0] for r in results]
        assert "b" not in res_ids
        assert set(res_ids) == {"a", "c"}

    def test_large_corpus_top_k_ordering(self):
        """Verify top_k ranking across a large synthetic dataset."""
        dim = 64
        n_docs = 200
        rng = np.random.RandomState(42)
        raw_vecs = rng.randn(n_docs, dim).astype(np.float32)
        doc_ids = [f"doc_{i}" for i in range(n_docs)]

        index = DenseVectorIndex(dimension=dim)
        index.add(doc_ids, raw_vecs)
        assert index.size() == n_docs

        query = rng.randn(dim).astype(np.float32)
        top_10 = index.search(query, top_k=10)

        assert len(top_10) == 10
        # Verify strictly monotonically decreasing scores
        scores = [score for _, score in top_10]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_concurrent_search_and_add(self):
        index = DenseVectorIndex(dimension=16)
        rng = np.random.RandomState(123)
        initial = rng.randn(20, 16).astype(np.float32)
        index.add([f"init_{i}" for i in range(20)], initial)

        def reader():
            q = rng.randn(16).astype(np.float32)
            for _ in range(50):
                res = index.search(q, top_k=5)
                assert len(res) <= 5

        def writer(offset: int):
            for i in range(10):
                vec = rng.randn(1, 16).astype(np.float32)
                index.add([f"writer_{offset}_{i}"], vec)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(reader),
                executor.submit(reader),
                executor.submit(writer, 1),
                executor.submit(writer, 2),
            ]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert index.size() == 40
