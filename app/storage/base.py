"""Abstract base repository contract for SpecRAG document storage."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.api.schemas import DocumentChunk


class StorageRepository(ABC):
    """Abstract interface defining required document and vector persistence methods."""

    @abstractmethod
    def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        pass

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Persist a batch of document chunks with their embeddings.

        Returns the number of chunks successfully persisted.
        """
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Fetch a single chunk by its ID."""
        pass

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, DocumentChunk]:
        """Fetch multiple chunks by their IDs.

        Default implementation falls back to per-ID retrieval; concrete implementations
        should override this with a single batch query for maximum throughput.
        """
        results: Dict[str, DocumentChunk] = {}
        for cid in chunk_ids:
            chunk = self.get_chunk(cid)
            if chunk is not None:
                results[cid] = chunk
        return results

    @abstractmethod
    def get_all_chunks(self, namespace: Optional[str] = None) -> List[DocumentChunk]:
        """Fetch all chunks for index warming and corpus iteration, optionally filtered by namespace."""
        pass

    @abstractmethod
    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Fetch all chunk IDs and their stacked float32 embeddings (N, D)."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored chunks."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Purge all stored chunks."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connections and release held resources."""
        pass
