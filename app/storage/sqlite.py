"""SQLite storage repository for SpecRAG with float32 BLOB vector persistence."""

import json
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.api.schemas import DocumentChunk
from app.storage.base import StorageRepository


class SQLiteStorageRepository(StorageRepository):
    """Embedded SQLite repository supporting float32 BLOB vector storage."""

    def __init__(self, db_path: str = "specrag.db", dimension: int = 384):
        self.db_path = db_path
        self.dimension = dimension
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.initialize()

    def _get_connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=30.0,
                )
                self._conn.execute("PRAGMA journal_mode = WAL;")
                self._conn.execute("PRAGMA synchronous = NORMAL;")
            return self._conn

    def initialize(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS specrag_chunks (
                        id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        embedding BLOB NOT NULL,
                        namespace TEXT NOT NULL DEFAULT 'default',
                        created_at REAL NOT NULL
                    );
                    """
                )
                # Ensure backward-compatible column addition if table existed without namespace
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(specrag_chunks);")
                columns = [col[1] for col in cursor.fetchall()]
                if "namespace" not in columns:
                    conn.execute("ALTER TABLE specrag_chunks ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default';")

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_specrag_doc_id
                    ON specrag_chunks(doc_id);
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_specrag_namespace
                    ON specrag_chunks(namespace);
                    """
                )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Persist a batch of document chunks with their embeddings."""
        if not chunks:
            return 0

        now = time.time()
        records = []
        for c in chunks:
            if c.embedding is not None and len(c.embedding) > 0:
                if len(c.embedding) != self.dimension:
                    raise ValueError(
                        f"Chunk {c.id} embedding dimension mismatch: expected {self.dimension}, got {len(c.embedding)}"
                    )
                emb_arr = np.asarray(c.embedding, dtype=np.float32)
                emb_bytes = emb_arr.tobytes()
            else:
                emb_bytes = b""

            meta_str = json.dumps(c.metadata, ensure_ascii=False)
            records.append((c.id, c.doc_id, c.text, meta_str, emb_bytes, c.namespace, now))

        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO specrag_chunks
                    (id, doc_id, text, metadata_json, embedding, namespace, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    records,
                )
        return len(records)

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Fetch a single chunk by ID."""
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, doc_id, text, metadata_json, embedding, namespace
                FROM specrag_chunks
                WHERE id = ?;
                """,
                (chunk_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_chunk(row)

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, DocumentChunk]:
        """Fetch multiple chunks by their IDs in batch queries."""
        if not chunk_ids:
            return {}

        results: Dict[str, DocumentChunk] = {}
        # Batch in chunks of 500 to stay well under SQLite parameter limit (999)
        batch_size = 500
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i : i + batch_size]
                placeholders = ",".join("?" for _ in batch)
                cur.execute(
                    f"""
                    SELECT id, doc_id, text, metadata_json, embedding, namespace
                    FROM specrag_chunks
                    WHERE id IN ({placeholders});
                    """,
                    batch,
                )
                for row in cur.fetchall():
                    chunk = self._row_to_chunk(row)
                    results[chunk.id] = chunk
        return results

    def get_all_chunks(self, namespace: Optional[str] = None) -> List[DocumentChunk]:
        """Fetch all chunks, optionally filtered by namespace, for index warming."""
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            if namespace is not None:
                cur.execute(
                    """
                    SELECT id, doc_id, text, metadata_json, embedding, namespace
                    FROM specrag_chunks
                    WHERE namespace = ?
                    ORDER BY created_at ASC;
                    """,
                    (namespace,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, doc_id, text, metadata_json, embedding, namespace
                    FROM specrag_chunks
                    ORDER BY created_at ASC;
                    """
                )
            rows = cur.fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Fetch all chunk IDs and their stacked float32 embeddings (N, D)."""
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, embedding
                FROM specrag_chunks
                WHERE length(embedding) > 0
                ORDER BY created_at ASC;
                """
            )
            rows = cur.fetchall()

        if not rows:
            return [], np.empty((0, self.dimension), dtype=np.float32)

        ids = [r[0] for r in rows]
        # High-performance zero-copy buffer concatenation
        raw_stream = b"".join(r[1] for r in rows)
        expected_bytes_per_vec = self.dimension * 4  # float32 = 4 bytes
        if len(raw_stream) % expected_bytes_per_vec != 0:
            # Fallback if variable dimensions or corruption
            vec_list = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
            matrix = np.vstack(vec_list).astype(np.float32)
        else:
            matrix = np.frombuffer(raw_stream, dtype=np.float32).reshape(len(rows), self.dimension)

        return ids, matrix

    def count(self) -> int:
        """Return total stored chunks count."""
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM specrag_chunks;")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def clear(self) -> None:
        """Purge all stored data."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("DELETE FROM specrag_chunks;")

    def close(self) -> None:
        """Close connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _row_to_chunk(self, row: Tuple) -> DocumentChunk:
        if len(row) == 6:
            cid, doc_id, text, meta_json, emb_blob, ns = row
        else:
            cid, doc_id, text, meta_json, emb_blob = row[:5]
            ns = "default"
        meta = json.loads(meta_json) if meta_json else {}
        emb = None
        if emb_blob and len(emb_blob) > 0:
            emb = np.frombuffer(emb_blob, dtype=np.float32).tolist()
        return DocumentChunk(
            id=cid,
            doc_id=doc_id,
            text=text,
            metadata=meta,
            embedding=emb,
            namespace=ns,
        )
