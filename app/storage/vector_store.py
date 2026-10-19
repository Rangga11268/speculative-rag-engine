"""Embedded in-memory dense vector index using NumPy normalized dot-product cosine similarity."""

import threading
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class DenseVectorIndex:
    """Thread-safe, in-memory dense vector index with L2-normalized cosine search."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.doc_ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._id_to_ns: Dict[str, str] = {}
        self.matrix: Optional[np.ndarray] = None  # Normalized float32 matrix (N, D)
        self.lock = threading.RLock()

    def size(self) -> int:
        """Return number of currently indexed vectors."""
        with self.lock:
            return len(self.doc_ids)

    def clear(self) -> None:
        """Clear all indexed vectors."""
        with self.lock:
            self.doc_ids.clear()
            self._id_to_idx.clear()
            self._id_to_ns.clear()
            self.matrix = None

    def add(
        self,
        ids: Sequence[str],
        vectors: Union[np.ndarray, Sequence[Sequence[float]]],
        namespaces: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        """Add or update vector embeddings for given chunk/document IDs."""
        if not ids:
            return

        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        if vecs.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vecs.shape[1]}"
            )

        if len(ids) != vecs.shape[0]:
            raise ValueError(
                f"Count mismatch: {len(ids)} ids provided for {vecs.shape[0]} vectors"
            )

        # Sanitize non-finite values (NaN/Inf) to prevent NaN leakage downstream
        vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)

        # L2-normalize vectors to allow cosine similarity via dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        norm_vecs = vecs / norms

        # Deduplicate IDs within single batch, preserving order and taking last vector
        if namespaces is None:
            ns_list = ["default"] * len(ids)
        elif isinstance(namespaces, str):
            ns_list = [namespaces] * len(ids)
        else:
            ns_list = list(namespaces)
            if len(ns_list) != len(ids):
                raise ValueError(f"Namespaces count mismatch: {len(ns_list)} vs {len(ids)} ids")

        seen: Dict[str, Tuple[np.ndarray, str]] = {}
        for i, doc_id in enumerate(ids):
            seen[doc_id] = (norm_vecs[i], ns_list[i])
        dedup_ids = list(seen.keys())
        dedup_vecs = [seen[did][0] for did in dedup_ids]
        dedup_nss = [seen[did][1] for did in dedup_ids]

        with self.lock:
            # Check for existing IDs to handle updates cleanly
            new_ids = []
            new_vecs = []

            for i, doc_id in enumerate(dedup_ids):
                self._id_to_ns[doc_id] = dedup_nss[i]
                if doc_id in self._id_to_idx:
                    idx = self._id_to_idx[doc_id]
                    self.matrix[idx] = dedup_vecs[i]
                else:
                    new_ids.append(doc_id)
                    new_vecs.append(dedup_vecs[i])

            if new_ids:
                new_matrix = np.vstack(new_vecs).astype(np.float32)
                if self.matrix is None or len(self.doc_ids) == 0:
                    self.doc_ids = list(new_ids)
                    self.matrix = new_matrix
                    self._id_to_idx = {did: idx for idx, did in enumerate(self.doc_ids)}
                else:
                    start_idx = len(self.doc_ids)
                    self.doc_ids.extend(new_ids)
                    self.matrix = np.vstack([self.matrix, new_matrix])
                    for offset, did in enumerate(new_ids):
                        self._id_to_idx[did] = start_idx + offset

    def remove(self, ids: Sequence[str]) -> int:
        """Remove vectors with the specified IDs. Returns count of removed items."""
        with self.lock:
            to_remove = set(ids).intersection(self._id_to_idx.keys())
            if not to_remove:
                return 0

            keep_mask = [did not in to_remove for did in self.doc_ids]
            self.doc_ids = [did for did in self.doc_ids if did not in to_remove]
            for did in to_remove:
                self._id_to_ns.pop(did, None)

            if self.doc_ids and self.matrix is not None:
                self.matrix = self.matrix[keep_mask]
                self._id_to_idx = {did: idx for idx, did in enumerate(self.doc_ids)}
            else:
                self.matrix = None
                self._id_to_idx.clear()
            return len(to_remove)

    def search(
        self,
        query_vector: Union[np.ndarray, Sequence[float]],
        top_k: int = 10,
        namespace: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Search top-k most similar chunks using cosine similarity, optionally filtered by namespace."""
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if q.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {q.shape[0]}"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("Query vector contains non-finite values (NaN or Inf)")

        with self.lock:
            if self.matrix is None or len(self.doc_ids) == 0 or top_k <= 0:
                return []

            # Determine candidate indices matching namespace if specified
            if namespace is not None:
                candidate_indices = [
                    idx for idx, did in enumerate(self.doc_ids)
                    if self._id_to_ns.get(did, "default") == namespace
                ]
                if not candidate_indices:
                    return []
                cand_idx_arr = np.array(candidate_indices, dtype=np.intp)
                sub_matrix = self.matrix[cand_idx_arr]
                sub_doc_ids = [self.doc_ids[idx] for idx in candidate_indices]
            else:
                sub_matrix = self.matrix
                sub_doc_ids = self.doc_ids

            # L2 normalize query
            norm = float(np.linalg.norm(q))
            if norm > 0.0:
                q = q / norm
            else:
                # If zero vector, cosine similarity is 0.0
                return [(did, 0.0) for did in sub_doc_ids[: min(top_k, len(sub_doc_ids))]]

            # Vectorized dot product for cosine similarity over unit vectors
            scores = np.dot(sub_matrix, q)
            k = min(top_k, len(scores))

            # Fast partial sort: O(N) partitioning followed by sorting only top-k
            if len(scores) > k:
                top_indices = np.argpartition(-scores, k)[:k]
                top_indices = top_indices[np.argsort(-scores[top_indices])]
            else:
                top_indices = np.argsort(-scores)

            return [(sub_doc_ids[idx], float(scores[idx])) for idx in top_indices]
