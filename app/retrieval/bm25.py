"""Pure-Python Okapi BM25 index with inverted index and Robertson-Spärck Jones IDF."""

import math
import re
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple


class OkapiBM25Index:
    """Zero-dependency, pure-Python Okapi BM25 lexical search index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_len: List[int] = []
        self.doc_ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self._token_pattern = re.compile(r"\b\w+\b", re.UNICODE)
        self._corpus: Dict[str, str] = {}
        self.lock = threading.RLock()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric words."""
        if not text:
            return []
        return self._token_pattern.findall(text.lower())

    def size(self) -> int:
        """Return number of documents in the index."""
        with self.lock:
            return self.corpus_size

    def clear(self) -> None:
        """Purge all indexed documents."""
        with self.lock:
            self._corpus.clear()
            self.corpus_size = 0
            self.avgdl = 0.0
            self.doc_len.clear()
            self.doc_ids.clear()
            self._id_to_idx.clear()
            self.inverted_index.clear()
            self.idf.clear()

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Build or replace the full index from scratch."""
        if len(doc_ids) != len(texts):
            raise ValueError(
                f"Count mismatch: {len(doc_ids)} doc_ids vs {len(texts)} texts"
            )

        with self.lock:
            self._corpus = dict(zip(doc_ids, texts))
            self.corpus_size = len(texts)
            self.doc_ids = list(doc_ids)
            self._id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
            self.doc_len.clear()
            self.inverted_index.clear()
            self.idf.clear()

            if self.corpus_size == 0:
                self.avgdl = 0.0
                return

            total_len = 0
            df: Dict[str, int] = defaultdict(int)

            for idx, text in enumerate(texts):
                tokens = self._tokenize(text)
                length = len(tokens)
                self.doc_len.append(length)
                total_len += length

                tf = Counter(tokens)
                for token, count in tf.items():
                    self.inverted_index[token].append((idx, count))
                    df[token] += 1

            self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

            # Precompute Robertson-Spärck Jones IDF (always positive with +1.0)
            for token, freq in df.items():
                self.idf[token] = math.log(
                    1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
                )

    def add(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Add new documents or update existing ones (rebuilds index)."""
        if len(doc_ids) != len(texts):
            raise ValueError(
                f"Count mismatch: {len(doc_ids)} doc_ids vs {len(texts)} texts"
            )
        if not doc_ids:
            return

        with self.lock:
            for did, txt in zip(doc_ids, texts):
                self._corpus[did] = txt
            all_ids = list(self._corpus.keys())
            all_texts = [self._corpus[did] for did in all_ids]
            self.index(all_ids, all_texts)

    def remove(self, doc_ids: Sequence[str]) -> int:
        """Remove specified documents from index. Returns count of removed items."""
        with self.lock:
            removed = 0
            for did in doc_ids:
                if did in self._corpus:
                    del self._corpus[did]
                    removed += 1
            if removed > 0:
                all_ids = list(self._corpus.keys())
                all_texts = [self._corpus[did] for did in all_ids]
                self.index(all_ids, all_texts)
            return removed

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search top-k documents matching the query using Okapi BM25."""
        with self.lock:
            if not query or self.corpus_size == 0 or top_k <= 0:
                return []

            tokens = self._tokenize(query)
            if not tokens:
                return []

            scores: Dict[int, float] = defaultdict(float)
            k1 = self.k1
            b = self.b
            avgdl = self.avgdl if self.avgdl > 0.0 else 1.0

            for token in tokens:
                if token not in self.inverted_index:
                    continue
                idf_val = self.idf[token]
                for doc_idx, count in self.inverted_index[token]:
                    dl = self.doc_len[doc_idx]
                    denom = count + k1 * (1.0 - b + b * (dl / avgdl))
                    scores[doc_idx] += idf_val * (count * (k1 + 1.0)) / denom

            if not scores:
                return []

            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
            return [(self.doc_ids[idx], float(score)) for idx, score in ranked]
