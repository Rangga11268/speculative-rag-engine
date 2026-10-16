"""Dense Text Embedder module for SpecRAG.

Provides deterministic dense embedding generation (float32 unit-normalized)
compatible with native Windows Python 3.12 without mandatory external model downloads.
"""

import hashlib
import math
from typing import List, Sequence
import numpy as np


class DenseEmbedder:
    """Generates normalized dense embeddings for document chunks and queries."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        """Generate a deterministic unit-normalized float32 embedding for a string."""
        if not text:
            # Zero vector normalized or small baseline
            return [0.0] * self.dimension

        # Feature hashing with character n-grams and token tokens
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().strip().split()

        for token in tokens:
            # Primary token hash
            h1 = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx1 = h1 % self.dimension
            sign1 = 1.0 if (h1 >> 8) & 1 else -1.0
            vec[idx1] += sign1 * 1.5

            # Secondary sub-token / character bigram hash for OOV robustness
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    trigram = token[i:i+3]
                    h2 = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
                    idx2 = h2 % self.dimension
                    sign2 = 1.0 if (h2 >> 4) & 1 else -1.0
                    vec[idx2] += sign2 * 0.5

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec = np.zeros(self.dimension, dtype=np.float32)

        return vec.tolist()

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        return [self.embed_text(t) for t in texts]
