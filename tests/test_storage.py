"""Unit and integration tests for SQLite and PostgreSQL storage repositories."""

import concurrent.futures
import os
import tempfile
import numpy as np
import pytest

from app.api.schemas import DocumentChunk
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.postgres import PostgresStorageRepository


class TestSQLiteStorageRepository:
    """Test suite for SQLiteStorageRepository."""

    def test_initialize_and_schema(self, tmp_sqlite_path):
        repo = SQLiteStorageRepository(db_path=tmp_sqlite_path, dimension=384)
        assert repo.count() == 0
        repo.close()

    def test_add_and_get_chunk(self, tmp_sqlite_path, sample_chunks):
        repo = SQLiteStorageRepository(db_path=tmp_sqlite_path, dimension=384)
        added = repo.add_chunks(sample_chunks)
        assert added == len(sample_chunks)
        assert repo.count() == len(sample_chunks)

        chunk_1 = repo.get_chunk("chunk_1")
        assert chunk_1 is not None
        assert chunk_1.id == "chunk_1"
        assert chunk_1.doc_id == sample_chunks[0].doc_id
        assert chunk_1.text == sample_chunks[0].text
        assert chunk_1.metadata == sample_chunks[0].metadata
        assert chunk_1.embedding is not None

        # Verify float32 numerical fidelity
        original_emb = np.array(sample_chunks[0].embedding, dtype=np.float32)
        retrieved_emb = np.array(chunk_1.embedding, dtype=np.float32)
        np.testing.assert_allclose(retrieved_emb, original_emb, rtol=1e-6, atol=1e-6)
        repo.close()

    def test_get_chunks_by_ids(self, tmp_sqlite_path, sample_chunks):
        repo = SQLiteStorageRepository(db_path=tmp_sqlite_path, dimension=384)
        repo.add_chunks(sample_chunks)

        batch_result = repo.get_chunks_by_ids(["chunk_2", "chunk_4", "nonexistent"])
        assert len(batch_result) == 2
        assert "chunk_2" in batch_result
        assert "chunk_4" in batch_result
        assert "nonexistent" not in batch_result
        repo.close()

    def test_get_all_embeddings(self, tmp_sqlite_path, sample_chunks, sample_embeddings):
        repo = SQLiteStorageRepository(db_path=tmp_sqlite_path, dimension=384)
        repo.add_chunks(sample_chunks)

        ids, matrix = repo.get_all_embeddings()
        assert len(ids) == len(sample_chunks)
        assert matrix.shape == (len(sample_chunks), 384)
        assert matrix.dtype == np.float32

        # Check that stacked matrix rows match original embeddings
        for i, chunk_id in enumerate(ids):
            idx = int(chunk_id.split("_")[1]) - 1
            np.testing.assert_allclose(matrix[i], sample_embeddings[idx], rtol=1e-6, atol=1e-6)
        repo.close()

    def test_update_chunk(self, tmp_sqlite_path, sample_chunks):
        repo = SQLiteStorageRepository(db_path=tmp_sqlite_path, dimension=384)
        repo.add_chunks([sample_chunks[0]])
        assert repo.count() == 1

        updated_chunk = DocumentChunk(
            id=sample_chunks[0].id,
            doc_id="doc_updated",
            text="Updated text content",
            metadata={"new_field": True},
            embedding=sample_chunks[0].embedding,
        )
        repo.add_chunks([updated_chunk])
        assert repo.count() == 1

        retrieved = repo.get_chunk(sample_chunks[0].id)
        assert retrieved.doc_id == "doc_updated"
        assert retrieved.text == "Updated text content"
        assert retrieved.metadata == {"new_field": True}
        repo.close()

    def test_clear(self, tmp_sqlite_path, sample_chunks):
        repo = SQLiteStorageRepository(db_path=tmp_sqlite_path, dimension=384)
        repo.add_chunks(sample_chunks)
        assert repo.count() == 5
        repo.clear()
        assert repo.count() == 0
        repo.close()

    def test_concurrent_access(self, tmp_sqlite_path, sample_chunks):
        repo = SQLiteStorageRepository(db_path=tmp_sqlite_path, dimension=384)
        repo.add_chunks(sample_chunks)

        def worker(thread_idx: int):
            for i in range(10):
                cid = f"chunk_{(i % 5) + 1}"
                chunk = repo.get_chunk(cid)
                assert chunk is not None
                assert chunk.id == cid
            return thread_idx

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(8)]
            for f in concurrent.futures.as_completed(futures):
                assert f.result() >= 0
        repo.close()


class TestPostgresStorageRepository:
    """Test suite for PostgresStorageRepository with graceful fallback."""

    def test_graceful_fallback_on_bad_connection(self):
        fd, fallback_path = tempfile.mkstemp(suffix=".db", prefix="pg_fallback_test_")
        os.close(fd)
        try:
            # Connect to invalid port 59999 to force connection failure
            repo = PostgresStorageRepository(
                host="127.0.0.1",
                port=59999,
                dbname="nonexistent_db",
                user="bad_user",
                password="bad_password",
                fallback_to_sqlite=True,
                sqlite_fallback_path=fallback_path,
                dimension=384,
            )
            assert repo.is_fallback is True

            # Data operations should transparently work via SQLite fallback
            test_chunk = DocumentChunk(
                id="fb_1",
                doc_id="doc_fb",
                text="Fallback text",
                metadata={"fallback": True},
                embedding=[0.1] * 384,
            )
            added = repo.add_chunks([test_chunk])
            assert added == 1
            assert repo.count() == 1

            retrieved = repo.get_chunk("fb_1")
            assert retrieved is not None
            assert retrieved.id == "fb_1"
            assert retrieved.text == "Fallback text"

            repo.clear()
            assert repo.count() == 0
            repo.close()
        finally:
            if os.path.exists(fallback_path):
                os.remove(fallback_path)

    def test_strict_failure_when_fallback_disabled(self):
        import psycopg2

        with pytest.raises(Exception):
            PostgresStorageRepository(
                host="127.0.0.1",
                port=59999,
                dbname="nonexistent_db",
                user="bad_user",
                password="bad_password",
                fallback_to_sqlite=False,
                dimension=384,
            )

    def test_live_postgres_if_available(self, sample_chunks):
        """Test against live local PostgreSQL instance if reachable."""
        try:
            repo = PostgresStorageRepository(
                host="127.0.0.1",
                port=5432,
                dbname="postgres",
                user="postgres",
                password="postgres",
                fallback_to_sqlite=False,
                dimension=384,
            )
        except Exception:
            pytest.skip("Local PostgreSQL is not reachable with default credentials")

        try:
            repo.clear()
            added = repo.add_chunks(sample_chunks)
            assert added == len(sample_chunks)
            assert repo.count() == len(sample_chunks)

            # Retrieve chunk
            chunk_1 = repo.get_chunk("chunk_1")
            assert chunk_1 is not None
            assert chunk_1.id == "chunk_1"
            assert chunk_1.text == sample_chunks[0].text
            assert np.allclose(chunk_1.embedding, sample_chunks[0].embedding, atol=1e-6)

            # Batch retrieval
            batch = repo.get_chunks_by_ids(["chunk_1", "chunk_3"])
            assert len(batch) == 2

            # Embeddings matrix
            ids, matrix = repo.get_all_embeddings()
            assert len(ids) == len(sample_chunks)
            assert matrix.shape == (len(sample_chunks), 384)

            repo.clear()
            assert repo.count() == 0
        finally:
            repo.close()


class TestEngineStorageBackendSelection:
    """Tests verifying SpecRAGEngine correctly honors settings.STORAGE_BACKEND."""

    def test_engine_initializes_sqlite_by_default(self, tmp_path):
        from app.config import Settings
        from app.core.engine import SpecRAGEngine

        db_path = str(tmp_path / "sqlite_test.db")
        settings = Settings(STORAGE_BACKEND="sqlite", SQLITE_DB_PATH=db_path)
        engine = SpecRAGEngine(settings=settings)

        assert isinstance(engine.storage_repo, SQLiteStorageRepository)
        assert engine.storage_repo.db_path == db_path
        engine.storage_repo.close()

    def test_engine_initializes_postgres_with_fallback(self, tmp_path):
        from app.config import Settings
        from app.core.engine import SpecRAGEngine

        fallback_path = str(tmp_path / "fallback.db")
        settings = Settings(
            STORAGE_BACKEND="postgres",
            POSTGRES_HOST="127.0.0.1",
            POSTGRES_PORT=54329,  # Unreachable port
            POSTGRES_FALLBACK_TO_SQLITE=True,
            SQLITE_DB_PATH=fallback_path,
        )
        engine = SpecRAGEngine(settings=settings)
        assert isinstance(engine.storage_repo, PostgresStorageRepository)
        # Should have fallen back safely
        assert engine.storage_repo.is_fallback is True
        engine.storage_repo.close()

    def test_engine_postgres_fails_fast_when_fallback_disabled(self, tmp_path):
        from app.config import Settings
        from app.core.engine import SpecRAGEngine

        settings = Settings(
            STORAGE_BACKEND="postgres",
            POSTGRES_HOST="127.0.0.1",
            POSTGRES_PORT=54329,  # Unreachable port
            POSTGRES_FALLBACK_TO_SQLITE=False,
        )
        with pytest.raises(Exception):
            SpecRAGEngine(settings=settings)

    def test_engine_respects_explicit_storage_repo(self, tmp_path):
        from app.config import Settings
        from app.core.engine import SpecRAGEngine

        custom_path = str(tmp_path / "custom.db")
        custom_repo = SQLiteStorageRepository(db_path=custom_path, dimension=384)
        settings = Settings(STORAGE_BACKEND="postgres")  # Backend in settings is ignored if explicit repo passed

        engine = SpecRAGEngine(storage_repo=custom_repo, settings=settings)
        assert engine.storage_repo is custom_repo
        engine.storage_repo.close()

    def test_postgres_runtime_write_failure_fails_closed(self, sample_chunks, monkeypatch):
        """When PostgreSQL connection drops during add_chunks and fallback is False, fail closed."""
        from unittest.mock import MagicMock

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.executemany.side_effect = Exception("DB connection reset by peer")

        repo = PostgresStorageRepository(
            fallback_to_sqlite=False,
            dimension=384,
        )
        repo._conn = mock_conn

        with pytest.raises(Exception, match="DB connection reset"):
            repo.add_chunks(sample_chunks)

    def test_postgres_runtime_write_failure_activates_fallback_and_recovers(self, sample_chunks, tmp_path):
        """When PostgreSQL fails during add_chunks and fallback is True, switch to SQLite."""
        from unittest.mock import MagicMock

        fallback_file = str(tmp_path / "runtime_fb.db")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.executemany.side_effect = Exception("OperationalError: server closed the connection")

        repo = PostgresStorageRepository(
            fallback_to_sqlite=True,
            sqlite_fallback_path=fallback_file,
            dimension=384,
        )
        repo._conn = mock_conn

        added = repo.add_chunks(sample_chunks)
        assert added == len(sample_chunks)
        assert repo.is_fallback is True
        assert repo.count() == len(sample_chunks)
        repo.close()

    def test_health_endpoint_surfaces_degraded_fallback_status(self, tmp_path):
        """FastAPI /health reflects degraded status when storage operates on fallback."""
        from fastapi.testclient import TestClient
        from app.config import Settings
        from app.core.engine import SpecRAGEngine
        from app.api.deps import set_engine
        from app.main import app

        fallback_file = str(tmp_path / "health_fb.db")
        settings = Settings(
            STORAGE_BACKEND="postgres",
            POSTGRES_PORT=54329,
            POSTGRES_FALLBACK_TO_SQLITE=True,
            SQLITE_DB_PATH=fallback_file,
        )
        engine = SpecRAGEngine(settings=settings)
        set_engine(engine)

        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "degraded"
        assert data["is_fallback"] is True
        assert data["storage_backend"] == "postgres"

        set_engine(None)
        engine.storage_repo.close()


