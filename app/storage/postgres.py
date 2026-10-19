"""PostgreSQL storage repository for SpecRAG with graceful SQLite fallback."""

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.api.schemas import DocumentChunk
from app.storage.base import StorageRepository
from app.storage.sqlite import SQLiteStorageRepository

logger = logging.getLogger("specrag.storage.postgres")


class PostgresStorageRepository(StorageRepository):
    """PostgreSQL storage repository using psycopg2, with automatic SQLite fallback."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5432,
        dbname: str = "postgres",
        user: str = "postgres",
        password: str = "postgres",
        fallback_to_sqlite: bool = True,
        sqlite_fallback_path: str = "specrag_fallback.db",
        dimension: int = 384,
    ):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.fallback_to_sqlite = fallback_to_sqlite
        self.sqlite_fallback_path = sqlite_fallback_path
        self.dimension = dimension

        self._lock = threading.RLock()
        self._conn = None
        self._fallback_repo: Optional[SQLiteStorageRepository] = None
        self._is_fallback = False

        self.initialize()

    @property
    def is_fallback(self) -> bool:
        """Return True if currently operating in SQLite fallback mode."""
        return self._is_fallback

    def _connect(self):
        import psycopg2

        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            connect_timeout=5,
        )

    def _activate_fallback(self, reason: Exception) -> None:
        if not self.fallback_to_sqlite:
            raise reason
        logger.warning(
            "PostgreSQL connection failed (%s). Activating SQLite fallback at '%s'.",
            reason,
            self.sqlite_fallback_path,
        )
        self._is_fallback = True
        self._fallback_repo = SQLiteStorageRepository(
            db_path=self.sqlite_fallback_path,
            dimension=self.dimension,
        )

    def initialize(self) -> None:
        """Create table and indexes in PostgreSQL, or activate fallback."""
        with self._lock:
            if self._is_fallback and self._fallback_repo:
                self._fallback_repo.initialize()
                return

            try:
                self._conn = self._connect()
                self._conn.autocommit = False
                with self._conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS specrag_chunks (
                            id VARCHAR(64) PRIMARY KEY,
                            doc_id VARCHAR(64) NOT NULL,
                            text TEXT NOT NULL,
                            metadata_json TEXT NOT NULL DEFAULT '{}',
                            embedding BYTEA NOT NULL,
                            namespace VARCHAR(64) NOT NULL DEFAULT 'default',
                            created_at DOUBLE PRECISION NOT NULL
                        );
                        """
                    )
                    cur.execute(
                        """
                        ALTER TABLE specrag_chunks
                        ADD COLUMN IF NOT EXISTS namespace VARCHAR(64) NOT NULL DEFAULT 'default';
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_specrag_doc_id
                        ON specrag_chunks(doc_id);
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_specrag_namespace
                        ON specrag_chunks(namespace);
                        """
                    )
                self._conn.commit()
                self._is_fallback = False
            except Exception as exc:
                self._activate_fallback(exc)

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Persist a batch of chunks to PostgreSQL, or fallback to SQLite."""
        if not chunks:
            return 0

        for c in chunks:
            if c.embedding is not None and len(c.embedding) > 0:
                if len(c.embedding) != self.dimension:
                    raise ValueError(
                        f"Chunk {c.id} embedding dimension mismatch: expected {self.dimension}, got {len(c.embedding)}"
                    )

        with self._lock:
            if self._is_fallback and self._fallback_repo:
                return self._fallback_repo.add_chunks(chunks)

            import psycopg2

            now = time.time()
            records = []
            for c in chunks:
                if c.embedding is not None and len(c.embedding) > 0:
                    emb_bytes = np.asarray(c.embedding, dtype=np.float32).tobytes()
                else:
                    emb_bytes = b""
                meta_str = json.dumps(c.metadata, ensure_ascii=False)
                records.append(
                    (
                        c.id,
                        c.doc_id,
                        c.text,
                        meta_str,
                        psycopg2.Binary(emb_bytes),
                        c.namespace,
                        now,
                    )
                )

            try:
                with self._conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO specrag_chunks
                        (id, doc_id, text, metadata_json, embedding, namespace, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            text = EXCLUDED.text,
                            metadata_json = EXCLUDED.metadata_json,
                            embedding = EXCLUDED.embedding,
                            namespace = EXCLUDED.namespace,
                            created_at = EXCLUDED.created_at;
                        """,
                        records,
                    )
                self._conn.commit()
                return len(records)
            except Exception as exc:
                if self.fallback_to_sqlite:
                    self._activate_fallback(exc)
                    return self._fallback_repo.add_chunks(chunks)
                raise

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Fetch a single chunk by ID."""
        with self._lock:
            if self._is_fallback and self._fallback_repo:
                return self._fallback_repo.get_chunk(chunk_id)

            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, doc_id, text, metadata_json, embedding, namespace
                        FROM specrag_chunks
                        WHERE id = %s;
                        """,
                        (chunk_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    return self._row_to_chunk(row)
            except Exception as exc:
                if self.fallback_to_sqlite:
                    self._activate_fallback(exc)
                    return self._fallback_repo.get_chunk(chunk_id)
                raise

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, DocumentChunk]:
        """Fetch multiple chunks by IDs in a single query."""
        if not chunk_ids:
            return {}

        with self._lock:
            if self._is_fallback and self._fallback_repo:
                return self._fallback_repo.get_chunks_by_ids(chunk_ids)

            try:
                results: Dict[str, DocumentChunk] = {}
                with self._conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, doc_id, text, metadata_json, embedding, namespace
                        FROM specrag_chunks
                        WHERE id = ANY(%s);
                        """,
                        (list(chunk_ids),),
                    )
                    for row in cur.fetchall():
                        chunk = self._row_to_chunk(row)
                        results[chunk.id] = chunk
                return results
            except Exception as exc:
                if self.fallback_to_sqlite:
                    self._activate_fallback(exc)
                    return self._fallback_repo.get_chunks_by_ids(chunk_ids)
                raise

    def get_all_chunks(self, namespace: Optional[str] = None) -> List[DocumentChunk]:
        """Fetch all chunks, optionally filtered by namespace."""
        with self._lock:
            if self._is_fallback and self._fallback_repo:
                return self._fallback_repo.get_all_chunks(namespace=namespace)

            try:
                with self._conn.cursor() as cur:
                    if namespace is not None:
                        cur.execute(
                            """
                            SELECT id, doc_id, text, metadata_json, embedding, namespace
                            FROM specrag_chunks
                            WHERE namespace = %s
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
                    return [self._row_to_chunk(r) for r in cur.fetchall()]
            except Exception as exc:
                if self.fallback_to_sqlite:
                    self._activate_fallback(exc)
                    return self._fallback_repo.get_all_chunks(namespace=namespace)
                raise

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Fetch all chunk IDs and their stacked float32 embeddings."""
        with self._lock:
            if self._is_fallback and self._fallback_repo:
                return self._fallback_repo.get_all_embeddings()

            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, embedding
                        FROM specrag_chunks
                        WHERE octet_length(embedding) > 0
                        ORDER BY created_at ASC;
                        """
                    )
                    rows = cur.fetchall()

                if not rows:
                    return [], np.empty((0, self.dimension), dtype=np.float32)

                ids = [r[0] for r in rows]
                blobs = [bytes(r[1]) for r in rows]
                raw_stream = b"".join(blobs)
                expected_bytes = self.dimension * 4
                if len(raw_stream) % expected_bytes != 0:
                    matrix = np.vstack([np.frombuffer(b, dtype=np.float32) for b in blobs]).astype(np.float32)
                else:
                    matrix = np.frombuffer(raw_stream, dtype=np.float32).reshape(len(rows), self.dimension)

                return ids, matrix
            except Exception as exc:
                if self.fallback_to_sqlite:
                    self._activate_fallback(exc)
                    return self._fallback_repo.get_all_embeddings()
                raise

    def count(self) -> int:
        """Return total chunks count."""
        with self._lock:
            if self._is_fallback and self._fallback_repo:
                return self._fallback_repo.count()

            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM specrag_chunks;")
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
            except Exception as exc:
                if self.fallback_to_sqlite:
                    self._activate_fallback(exc)
                    return self._fallback_repo.count()
                raise

    def clear(self) -> None:
        """Purge all stored chunks."""
        with self._lock:
            if self._is_fallback and self._fallback_repo:
                self._fallback_repo.clear()
                return

            try:
                with self._conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE specrag_chunks;")
                self._conn.commit()
            except Exception as exc:
                if self.fallback_to_sqlite:
                    self._activate_fallback(exc)
                    self._fallback_repo.clear()
                    return
                raise

    def close(self) -> None:
        """Close connection."""
        with self._lock:
            if self._fallback_repo is not None:
                self._fallback_repo.close()
                self._fallback_repo = None
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def _row_to_chunk(self, row: Tuple) -> DocumentChunk:
        if len(row) == 6:
            cid, doc_id, text, meta_json, emb_bytes, ns = row
        else:
            cid, doc_id, text, meta_json, emb_bytes = row[:5]
            ns = "default"
        meta = json.loads(meta_json) if meta_json else {}
        emb = None
        if emb_bytes is not None:
            raw = bytes(emb_bytes)
            if len(raw) > 0:
                emb = np.frombuffer(raw, dtype=np.float32).tolist()
        return DocumentChunk(
            id=cid,
            doc_id=doc_id,
            text=text,
            metadata=meta,
            embedding=emb,
            namespace=ns,
        )
