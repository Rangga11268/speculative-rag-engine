"""Data models and API schemas for SpecRAG."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -------------------------------------------------------------------------
# Dataclasses matching Core Storage & Retrieval Contracts (PROJECT.md)
# -------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Represents a discrete text chunk with vector embedding and metadata."""
    id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Instantiate DocumentChunk from dictionary."""
        return cls(
            id=str(data["id"]),
            doc_id=str(data["doc_id"]),
            text=str(data["text"]),
            metadata=dict(data.get("metadata") or {}),
            embedding=list(data["embedding"]) if data.get("embedding") is not None else None,
        )


@dataclass
class RetrievedCandidate:
    """Represents a scored candidate chunk retrieved from hybrid search."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    consensus_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to dictionary."""
        return asdict(self)


# -------------------------------------------------------------------------
# Pydantic Schemas for API Serialization & Validation
# -------------------------------------------------------------------------

class DocumentChunkSchema(BaseModel):
    """Pydantic schema for document chunk ingestion."""
    id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    text: str = Field(..., description="Raw text content of chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary JSON metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Precomputed dense embedding vector")

    def to_chunk(self) -> DocumentChunk:
        """Convert to internal DocumentChunk dataclass."""
        return DocumentChunk(
            id=self.id,
            doc_id=self.doc_id,
            text=self.text,
            metadata=self.metadata,
            embedding=self.embedding,
        )


class RetrievedCandidateSchema(BaseModel):
    """Pydantic schema for retrieved candidate in API response."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    consensus_count: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    """Request payload for hybrid retrieval."""
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of candidates to retrieve")
    dense_weight: float = Field(default=1.0, ge=0.0, description="RRF weight for dense vector search")
    bm25_weight: float = Field(default=1.0, ge=0.0, description="RRF weight for BM25 keyword search")


class IngestRequest(BaseModel):
    """Request payload for ingesting multiple chunks."""
    chunks: List[DocumentChunkSchema] = Field(..., min_length=1, description="List of chunks to ingest")


class IngestResponse(BaseModel):
    """Response payload for chunk ingestion."""
    count: int = Field(..., description="Number of chunks successfully persisted")
    message: str = Field(default="Ingestion successful", description="Status message")


class QueryRequest(BaseModel):
    """End-to-end query request."""
    query: str = Field(..., min_length=1, description="User search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Desired number of final ranked results")
    use_cache: bool = Field(default=True, description="Whether to check semantic cache")
    force_rerank: bool = Field(default=False, description="Bypass dynamic gating and force reranker")
    namespace: str = Field(default="default", description="Corpus or tenant namespace for cache isolation")
    index_version: Optional[str] = Field(default=None, description="Index version identifier for cache invalidation")



class GatingTelemetrySchema(BaseModel):
    """Telemetry captured by the dynamic confidence gate."""
    decision: str = Field(..., description="Routing decision code")
    entropy: float = Field(..., description="Shannon entropy H(P)")
    normalized_entropy: float = Field(..., description="Normalized entropy in [0, 1]")
    marginal_gap: float = Field(..., description="Score gap delta = s1 - s2")
    top1_score: float = Field(..., description="Normalized score of top-1 candidate")
    probabilities: List[float] = Field(default_factory=list, description="Softmax probabilities")
    bypass_granted: bool = Field(..., description="Whether fast path bypass was granted")


class QueryResponse(BaseModel):
    """End-to-end query response."""
    query: str
    results: List[RetrievedCandidateSchema]
    route_taken: str  # "CACHE_HIT", "FAST_PATH_BYPASS", "RERANKED"
    cache_hit: bool
    latency_ms: float
    gating_telemetry: Optional[GatingTelemetrySchema] = None
