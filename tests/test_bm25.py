"""Unit and algorithmic tests for OkapiBM25Index."""

import pytest

from app.retrieval.bm25 import OkapiBM25Index


class TestOkapiBM25Index:
    """Test suite for pure-Python OkapiBM25Index."""

    def test_empty_search(self):
        index = OkapiBM25Index()
        assert index.size() == 0
        assert index.search("speculative rag") == []

    def test_exact_keyword_matching(self):
        index = OkapiBM25Index()
        doc_ids = ["doc_fastapi", "doc_django", "doc_flask"]
        texts = [
            "FastAPI is a modern, high-performance web framework for Python.",
            "Django is a batteries-included full-stack framework with ORM and admin.",
            "Flask is a micro web framework with simple modular routing.",
        ]
        index.index(doc_ids, texts)
        assert index.size() == 3

        results = index.search("FastAPI high-performance", top_k=3)
        assert len(results) >= 1
        assert results[0][0] == "doc_fastapi"
        assert results[0][1] > 0.0

    def test_term_frequency_impact(self):
        index = OkapiBM25Index()
        doc_ids = ["doc_single", "doc_multiple"]
        texts = [
            "Entropy is a measure of randomness.",
            "Entropy entropy entropy provides a rigorous measure of entropy in statistical systems.",
        ]
        index.index(doc_ids, texts)

        results = index.search("entropy", top_k=2)
        assert len(results) == 2
        # doc_multiple has higher term frequency and should score higher
        assert results[0][0] == "doc_multiple"
        assert results[0][1] > results[1][1]

    def test_length_normalization_penalty(self):
        """Under b=0.75, when two documents have equal TF for a term, the shorter document scores higher."""
        index = OkapiBM25Index(k1=1.5, b=0.75)
        doc_ids = ["doc_concise", "doc_verbose"]
        texts = [
            "Quantum entanglement.",  # 2 words, TF('entanglement')=1
            "Quantum entanglement is a very interesting physical phenomenon where particles remain connected across arbitrary cosmic distances regardless of time and space.",  # 20 words, TF('entanglement')=1
        ]
        index.index(doc_ids, texts)

        results = index.search("entanglement", top_k=2)
        assert len(results) == 2
        # doc_concise has much higher term density, so length normalization ranks it first
        assert results[0][0] == "doc_concise"
        assert results[0][1] > results[1][1]

    def test_idf_weights_rare_terms_higher(self):
        """Rare terms in the corpus should have strictly higher IDF than common terms."""
        index = OkapiBM25Index()
        doc_ids = ["d1", "d2", "d3", "d4"]
        texts = [
            "Python machine learning algorithms.",
            "Python web applications with asynchronous tasks.",
            "Python data analysis and statistics.",
            "Asymptotics of hyperparameter optimization in deep models.",
        ]
        index.index(doc_ids, texts)

        # 'python' appears in 3 out of 4 documents -> lower IDF
        # 'asymptotics' appears in 1 out of 4 documents -> higher IDF
        assert "python" in index.idf
        assert "asymptotics" in index.idf
        assert index.idf["asymptotics"] > index.idf["python"]

    def test_tokenizer_edge_cases(self):
        index = OkapiBM25Index()
        doc_ids = ["special_doc"]
        texts = ["Version 3.12-rc1, cost: $45.99, email: test@specrag.org!"]
        index.index(doc_ids, texts)

        # Search with punctuation and mixed casing
        results = index.search("VERsioN 3.12!", top_k=1)
        assert len(results) == 1
        assert results[0][0] == "special_doc"

    def test_dynamic_add_and_remove(self):
        index = OkapiBM25Index()
        index.index(["doc_1"], ["Initial document on speculative decoding."])
        assert index.size() == 1

        # Add second document
        index.add(["doc_2"], ["Second document on cross-encoder reranking."])
        assert index.size() == 2

        results = index.search("cross-encoder", top_k=2)
        assert len(results) == 1
        assert results[0][0] == "doc_2"

        # Remove doc_2
        removed = index.remove(["doc_2"])
        assert removed == 1
        assert index.size() == 1

        results_after = index.search("cross-encoder", top_k=2)
        assert results_after == []

    def test_clear(self):
        index = OkapiBM25Index()
        index.index(["d1", "d2"], ["Text one", "Text two"])
        assert index.size() == 2
        index.clear()
        assert index.size() == 0
        assert index.search("text") == []
