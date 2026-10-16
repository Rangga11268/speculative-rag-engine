"""Empirical stress test harness for Challenger M1.2.
Probing:
1. Rapid concurrent reads and writes across threads.
2. Zero-vector queries, non-finite vectors (NaN, Inf).
3. Binary vector BLOB precision loss checks across serialization/deserialization.
4. SQLite busy/locked conditions and connection recovery.
"""

import concurrent.futures
import math
import os
import sqlite3
import tempfile
import threading
import time
from typing import List

import numpy as np
import pytest

from app.api.schemas import DocumentChunk
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.postgres import PostgresStorageRepository
from app.storage.vector_store import DenseVectorIndex


# ============================================================================
# 1. RAPID CONCURRENT READS AND WRITES ACROSS THREADS
# ============================================================================

def test_concurrent_threads_sqlite_single_repo():
    """Probe SQLiteStorageRepository under rapid simultaneous concurrent reads and writes."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_sqlite_single_")
    os.close(fd)
    try:
        repo = SQLiteStorageRepository(db_path=path, dimension=384)
        n_writers = 6
        n_readers = 6
        ops_per_thread = 30
        errors = []

        def writer_task(thread_id: int):
            try:
                for i in range(ops_per_thread):
                    cid = f"chunk_w{thread_id}_{i}"
                    emb = [float(thread_id * 100 + i)] * 384
                    chunk = DocumentChunk(
                        id=cid,
                        doc_id=f"doc_w{thread_id}",
                        text=f"Concurrent text from thread {thread_id} op {i}",
                        metadata={"thread": thread_id, "op": i},
                        embedding=emb,
                    )
                    repo.add_chunks([chunk])
            except Exception as e:
                errors.append(("writer", thread_id, e))

        def reader_task(thread_id: int):
            try:
                for i in range(ops_per_thread):
                    # Alternate between count, get_chunk, and get_all_embeddings
                    if i % 3 == 0:
                        repo.count()
                    elif i % 3 == 1:
                        repo.get_chunk(f"chunk_w{thread_id}_{max(0, i-1)}")
                    else:
                        repo.get_all_embeddings()
            except Exception as e:
                errors.append(("reader", thread_id, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_writers + n_readers) as executor:
            writer_futs = [executor.submit(writer_task, i) for i in range(n_writers)]
            reader_futs = [executor.submit(reader_task, i) for i in range(n_readers)]
            concurrent.futures.wait(writer_futs + reader_futs)

        assert not errors, f"Encountered concurrency errors: {errors}"
        expected_total = n_writers * ops_per_thread
        assert repo.count() == expected_total, f"Expected {expected_total} chunks, got {repo.count()}"
        ids, matrix = repo.get_all_embeddings()
        assert len(ids) == expected_total
        assert matrix.shape == (expected_total, 384)
        repo.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_concurrent_multi_instance_sqlite_wal():
    """Probe multi-instance SQLite connections concurrently reading & writing to same DB file."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_sqlite_multi_")
    os.close(fd)
    try:
        # Initial schema setup
        init_repo = SQLiteStorageRepository(db_path=path, dimension=64)
        init_repo.close()

        n_instances = 8
        ops_per_instance = 25
        errors = []

        def instance_task(instance_id: int):
            try:
                # Each thread gets its own SQLiteStorageRepository connection
                inst_repo = SQLiteStorageRepository(db_path=path, dimension=64)
                for i in range(ops_per_instance):
                    cid = f"inst_{instance_id}_{i}"
                    chunk = DocumentChunk(
                        id=cid,
                        doc_id=f"doc_{instance_id}",
                        text=f"Multi-instance text {instance_id}-{i}",
                        metadata={"inst": instance_id, "op": i},
                        embedding=[float(instance_id)] * 64,
                    )
                    inst_repo.add_chunks([chunk])
                    inst_repo.get_chunk(cid)
                    if i % 5 == 0:
                        inst_repo.count()
                inst_repo.close()
            except Exception as e:
                errors.append((instance_id, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_instances) as executor:
            futs = [executor.submit(instance_task, i) for i in range(n_instances)]
            concurrent.futures.wait(futs)

        assert not errors, f"Multi-instance SQLite errors: {errors}"
        verify_repo = SQLiteStorageRepository(db_path=path, dimension=64)
        assert verify_repo.count() == n_instances * ops_per_instance
        verify_repo.close()
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def test_dense_vector_index_concurrent_ops():
    """Stress test DenseVectorIndex under concurrent add, remove, and search."""
    dim = 32
    index = DenseVectorIndex(dimension=dim)
    n_ops = 50
    errors = []

    def writer_adder(tid: int):
        rng = np.random.RandomState(tid)
        try:
            for i in range(n_ops):
                v = rng.randn(2, dim).astype(np.float32)
                index.add([f"t{tid}_c{i}_a", f"t{tid}_c{i}_b"], v)
        except Exception as e:
            errors.append(("adder", tid, e))

    def reader_searcher(tid: int):
        rng = np.random.RandomState(tid + 100)
        try:
            for i in range(n_ops):
                q = rng.randn(dim).astype(np.float32)
                res = index.search(q, top_k=5)
                assert isinstance(res, list)
        except Exception as e:
            errors.append(("searcher", tid, e))

    def writer_remover(tid: int):
        try:
            for i in range(n_ops):
                if i % 3 == 0:
                    index.remove([f"t0_c{i}_a", f"t1_c{i}_a"])
        except Exception as e:
            errors.append(("remover", tid, e))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futs = [
            executor.submit(writer_adder, 0),
            executor.submit(writer_adder, 1),
            executor.submit(writer_adder, 2),
            executor.submit(reader_searcher, 3),
            executor.submit(reader_searcher, 4),
            executor.submit(writer_remover, 5),
        ]
        concurrent.futures.wait(futs)

    assert not errors, f"DenseVectorIndex concurrency errors: {errors}"
    with index.lock:
        matrix_len = len(index.matrix) if index.matrix is not None else 0
        assert index.size() == len(index.doc_ids) == len(index._id_to_idx) == matrix_len, (
            f"Index internal desync: size={index.size()}, "
            f"doc_ids={len(index.doc_ids)}, _id_to_idx={len(index._id_to_idx)}, matrix={matrix_len}"
        )


def test_dense_vector_index_duplicate_ids_in_single_batch():
    """Probe vulnerability: what happens when duplicate IDs are provided in the same add() batch?"""
    dim = 4
    index = DenseVectorIndex(dimension=dim)
    # Batch with duplicate ID 'dup_1'
    ids = ["dup_1", "dup_1", "unique_2"]
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    index.add(ids, vectors)

    with index.lock:
        doc_ids_len = len(index.doc_ids)
        dict_len = len(index._id_to_idx)
        matrix_len = len(index.matrix) if index.matrix is not None else 0

    # In a correct index, doc_ids, _id_to_idx, and matrix must have matching length!
    # If duplicate ID was appended twice, doc_ids_len is 3 while dict_len is 2!
    print(f"\n[DUPLICATE TEST] doc_ids={index.doc_ids}, _id_to_idx={index._id_to_idx}, matrix_rows={matrix_len}")
    assert doc_ids_len == dict_len, (
        f"BUG DISCOVERED: doc_ids length ({doc_ids_len}) != _id_to_idx length ({dict_len}). "
        f"Duplicate IDs in single add() batch corrupt index symmetry!"
    )


# ============================================================================
# 2. ZERO-VECTOR QUERIES & NON-FINITE VECTORS (NaN, Inf)
# ============================================================================

def test_zero_vector_query():
    """Probe DenseVectorIndex behavior on zero-vector queries."""
    dim = 4
    index = DenseVectorIndex(dimension=dim)
    index.add(["c1", "c2"], np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32))

    # All-zero query vector
    zero_q = [0.0, 0.0, 0.0, 0.0]
    results = index.search(zero_q, top_k=2)
    assert len(results) == 2
    for doc_id, score in results:
        assert score == 0.0, f"Expected 0.0 similarity for zero vector query, got {score}"


def test_nan_query_handling():
    """Probe DenseVectorIndex behavior on query vector containing NaN."""
    dim = 4
    index = DenseVectorIndex(dimension=dim)
    index.add(["c1", "c2"], np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32))

    nan_q = [float("nan"), 0.0, 0.0, 0.0]
    # Query vectors with NaN must either be rejected via ValueError or handled without leaking NaN or false zeros
    with pytest.raises(ValueError, match="non-finite|NaN|invalid"):
        index.search(nan_q, top_k=2)


def test_inf_query_handling():
    """Probe DenseVectorIndex behavior on query vector containing Inf."""
    dim = 4
    index = DenseVectorIndex(dimension=dim)
    index.add(["c1", "c2"], np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32))

    inf_q = [float("inf"), 1.0, 0.0, 0.0]
    # Query vectors with Inf must either be rejected via ValueError or handled without leaking NaN
    with pytest.raises(ValueError, match="non-finite|Inf|invalid"):
        index.search(inf_q, top_k=2)


def test_indexed_vector_with_nan():
    """Probe DenseVectorIndex when an indexed vector contains NaN."""
    dim = 4
    index = DenseVectorIndex(dimension=dim)
    # Adding a vector with NaN
    index.add(["good_1", "bad_nan", "good_2"], np.array([
        [1.0, 0.0, 0.0, 0.0],
        [np.nan, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32))

    # Now search with a completely valid query vector
    valid_q = [1.0, 0.0, 0.0, 0.0]
    results = index.search(valid_q, top_k=3)
    print(f"\n[INDEXED NAN TEST RESULTS]: {results}")
    # Check if bad_nan produces NaN score and what happens to good docs
    for doc_id, score in results:
        if doc_id == "bad_nan":
            print(f"bad_nan score: {score}")
        if doc_id == "good_1":
            assert pytest.approx(score, abs=1e-5) == 1.0


def test_indexed_vector_with_inf():
    """Probe DenseVectorIndex when an indexed vector contains Inf."""
    dim = 4
    index = DenseVectorIndex(dimension=dim)
    index.add(["good_1", "bad_inf"], np.array([
        [1.0, 0.0, 0.0, 0.0],
        [float("inf"), 1.0, 0.0, 0.0],
    ], dtype=np.float32))

    valid_q = [1.0, 0.0, 0.0, 0.0]
    results = index.search(valid_q, top_k=2)
    print(f"\n[INDEXED INF TEST RESULTS]: {results}")


def test_storage_nan_and_inf_roundtrip():
    """Probe SQLite and Postgres storage handling of NaN and Inf vectors."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_nan_inf_")
    os.close(fd)
    try:
        repo = SQLiteStorageRepository(db_path=path, dimension=4)
        nan_chunk = DocumentChunk(
            id="nan_chunk",
            doc_id="doc_nan",
            text="NaN text",
            metadata={},
            embedding=[float("nan"), 1.0, float("inf"), float("-inf")],
        )
        repo.add_chunks([nan_chunk])
        retrieved = repo.get_chunk("nan_chunk")
        assert retrieved is not None
        assert math.isnan(retrieved.embedding[0])
        assert retrieved.embedding[1] == 1.0
        assert math.isinf(retrieved.embedding[2]) and retrieved.embedding[2] > 0
        assert math.isinf(retrieved.embedding[3]) and retrieved.embedding[3] < 0

        # Now test get_all_embeddings
        ids, matrix = repo.get_all_embeddings()
        assert ids == ["nan_chunk"]
        assert matrix.shape == (1, 4)
        assert np.isnan(matrix[0, 0])
        assert np.isinf(matrix[0, 2])
        repo.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


# ============================================================================
# 3. BINARY VECTOR BLOB PRECISION LOSS & DIMENSION CORRUPTION
# ============================================================================

def test_blob_precision_fidelity_across_dtypes():
    """Verify exact float32 bitwise fidelity and quantify float64 truncation."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_precision_")
    os.close(fd)
    try:
        dim = 16
        repo = SQLiteStorageRepository(db_path=path, dimension=dim)

        # 1. Exact float32 bit patterns
        rng = np.random.RandomState(42)
        exact_float32 = rng.randn(dim).astype(np.float32)
        chunk1 = DocumentChunk(
            id="exact_f32",
            doc_id="doc_1",
            text="Exact float32",
            metadata={},
            embedding=exact_float32.tolist(),
        )
        repo.add_chunks([chunk1])

        retrieved1 = repo.get_chunk("exact_f32")
        retrieved_arr = np.array(retrieved1.embedding, dtype=np.float32)
        # Bitwise exactness check
        assert exact_float32.tobytes() == retrieved_arr.tobytes(), "Bitwise mismatch in float32 roundtrip!"

        # 2. Extreme dynamic range (subnormal, large, tiny)
        extremes = np.array([
            0.0,
            -0.0,
            1e-35,       # near float32 minimum positive normal
            1e35,        # near float32 maximum normal
            -1e35,
            float(np.finfo(np.float32).tiny),
            float(np.finfo(np.float32).max),
            float(np.finfo(np.float32).min),
            1.0 / 3.0,
            math.pi,
            math.e,
            2.0**(-10),
            2.0**10,
            0.1 + 0.2,
            -0.5,
            1.0,
        ], dtype=np.float32)

        chunk2 = DocumentChunk(
            id="extremes",
            doc_id="doc_2",
            text="Extremes",
            metadata={},
            embedding=extremes.tolist(),
        )
        repo.add_chunks([chunk2])

        retrieved2 = repo.get_chunk("extremes")
        retrieved_ext = np.array(retrieved2.embedding, dtype=np.float32)
        assert extremes.tobytes() == retrieved_ext.tobytes(), "Bitwise mismatch on extreme values!"

        # 3. Batch stacked get_all_embeddings matches individual chunks
        ids, matrix = repo.get_all_embeddings()
        assert ids == ["exact_f32", "extremes"]
        assert matrix[0].tobytes() == exact_float32.tobytes()
        assert matrix[1].tobytes() == extremes.tobytes()

        repo.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_mismatched_vector_dimension_behavior():
    """Probe vulnerability: what happens when a chunk with mismatched dimension is stored and fetched?"""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_dim_mismatch_")
    os.close(fd)
    try:
        repo = SQLiteStorageRepository(db_path=path, dimension=384)
        # Store a chunk with dimension 10 instead of 384
        bad_chunk = DocumentChunk(
            id="bad_dim_chunk",
            doc_id="doc_bad",
            text="Bad dimension",
            metadata={},
            embedding=[0.5] * 10,
        )
        normal_chunk = DocumentChunk(
            id="normal_chunk",
            doc_id="doc_good",
            text="Normal dimension",
            metadata={},
            embedding=[0.1] * 384,
        )

        # Now probe add_chunks or get_all_embeddings
        try:
            repo.add_chunks([bad_chunk, normal_chunk])
            ids, matrix = repo.get_all_embeddings()
            print(f"\n[DIM MISMATCH GET_ALL]: ids={ids}, matrix.shape={matrix.shape}")
        except Exception as e:
            print(f"\n[DIM MISMATCH EXCEPTION]: {type(e).__name__}: {e}")
            # If it crashed, verify it was caught cleanly or raised ValueError
            assert isinstance(e, ValueError)
        repo.close()
    finally:
        try:
            repo.close()
        except Exception:
            pass
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def test_storage_dimension_mismatch_silent_corruption_on_add():
    """Probe vulnerability: storing chunks with mismatched dimensions silently corrupts embeddings matrix.
    
    Chunk 1: 3 floats [1, 1, 1]
    Chunk 2: 5 floats [2, 2, 2, 2, 2]
    When dimension is 4, total bytes is 32 (a multiple of 16).
    SQLiteStorageRepository currently concatenates them and reshapes into (2, 4):
    Row 0: [1, 1, 1, 2] (stole float from c2)
    Row 1: [2, 2, 2, 2] (dropped float from c2)
    This silent data corruption must be prevented by validating dimensions at add_chunks() time!
    """
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_silent_corrupt_")
    os.close(fd)
    try:
        repo = SQLiteStorageRepository(db_path=path, dimension=4)
        c1 = DocumentChunk(id="c1", doc_id="d1", text="t1", metadata={}, embedding=[1.0, 1.0, 1.0])
        c2 = DocumentChunk(id="c2", doc_id="d2", text="t2", metadata={}, embedding=[2.0, 2.0, 2.0, 2.0, 2.0])

        with pytest.raises(ValueError, match="dimension mismatch"):
            repo.add_chunks([c1, c2])
    finally:
        try:
            repo.close()
        except Exception:
            pass
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


# ============================================================================
# 4. SQLITE BUSY/LOCKED CONDITIONS & CONNECTION RECOVERY
# ============================================================================

def test_sqlite_locked_exclusive_and_recovery():
    """Probe SQLiteStorageRepository when an external connection locks the database exclusively."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_locked_")
    os.close(fd)
    try:
        # Create and initialize the repository
        repo = SQLiteStorageRepository(db_path=path, dimension=4)
        chunk1 = DocumentChunk(id="c1", doc_id="d1", text="Text 1", metadata={}, embedding=[0.1]*4)
        repo.add_chunks([chunk1])
        assert repo.count() == 1

        # Simulate an external transaction holding EXCLUSIVE lock
        ext_conn = sqlite3.connect(path, timeout=0.1)
        ext_conn.execute("BEGIN EXCLUSIVE")

        # Now attempt to write from repo with short timeout
        # Repo connection has timeout=30.0 by default, so in a separate thread let's see:
        write_error = []
        def try_write():
            try:
                # Attempt to write while locked
                c2 = DocumentChunk(id="c2", doc_id="d2", text="Text 2", metadata={}, embedding=[0.2]*4)
                repo.add_chunks([c2])
            except Exception as e:
                write_error.append(e)

        th = threading.Thread(target=try_write)
        th.start()
        # Give it 0.2s to block on lock
        time.sleep(0.2)
        # Release the lock
        ext_conn.rollback()
        ext_conn.close()
        th.join(timeout=5.0)

        # After releasing lock, repo write should have completed or recovered!
        # Now verify connection recovery: can repo still read and write?
        assert repo.count() >= 1
        c3 = DocumentChunk(id="c3", doc_id="d3", text="Text 3", metadata={}, embedding=[0.3]*4)
        added = repo.add_chunks([c3])
        assert added == 1
        assert repo.get_chunk("c3") is not None
        repo.close()
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def test_sqlite_close_and_reopen_recovery():
    """Verify repository recovers transparently when close() is invoked and operations resume."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_reopen_")
    os.close(fd)
    try:
        repo = SQLiteStorageRepository(db_path=path, dimension=4)
        repo.add_chunks([DocumentChunk(id="c1", doc_id="d1", text="T1", metadata={}, embedding=[0.1]*4)])
        assert repo.count() == 1

        # Explicit close
        repo.close()
        assert repo._conn is None

        # Call get_chunk after close - should transparently re-open connection
        c = repo.get_chunk("c1")
        assert c is not None
        assert c.id == "c1"

        # Call count after close
        repo.close()
        assert repo.count() == 1

        # Call add_chunks after close
        repo.close()
        repo.add_chunks([DocumentChunk(id="c2", doc_id="d2", text="T2", metadata={}, embedding=[0.2]*4)])
        assert repo.count() == 2

        repo.close()
    finally:
        if os.path.exists(path):
            os.remove(path)



def test_postgres_fallback_concurrent_threads():
    """Probe PostgresStorageRepository in fallback mode under concurrent reads and writes."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_pg_fb_")
    os.close(fd)
    try:
        # Invalid port forces fallback mode
        repo = PostgresStorageRepository(
            host="127.0.0.1",
            port=59998,
            fallback_to_sqlite=True,
            sqlite_fallback_path=path,
            dimension=128,
        )
        assert repo.is_fallback is True

        n_threads = 6
        ops_per_thread = 20
        errors = []

        def worker(tid: int):
            try:
                for i in range(ops_per_thread):
                    cid = f"pg_fb_{tid}_{i}"
                    chunk = DocumentChunk(
                        id=cid,
                        doc_id=f"doc_{tid}",
                        text=f"PG fallback text {tid}-{i}",
                        metadata={"tid": tid},
                        embedding=[0.05 * tid] * 128,
                    )
                    repo.add_chunks([chunk])
                    repo.get_chunk(cid)
            except Exception as e:
                errors.append((tid, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futs = [executor.submit(worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futs)

        assert not errors, f"Postgres fallback concurrency errors: {errors}"
        assert repo.count() == n_threads * ops_per_thread
        repo.close()
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_postgres_live_concurrent_threads():
    """Probe live PostgreSQL under concurrent reads and writes if reachable."""
    try:
        repo = PostgresStorageRepository(
            host="127.0.0.1",
            port=5432,
            dbname="postgres",
            user="postgres",
            password="postgres",
            fallback_to_sqlite=False,
            dimension=128,
        )
    except Exception:
        pytest.skip("Live PostgreSQL not reachable on 127.0.0.1:5432")

    try:
        repo.clear()
        n_threads = 4
        ops_per_thread = 15
        errors = []

        def worker(tid: int):
            try:
                for i in range(ops_per_thread):
                    cid = f"pg_live_{tid}_{i}"
                    chunk = DocumentChunk(
                        id=cid,
                        doc_id=f"doc_{tid}",
                        text=f"PG live text {tid}-{i}",
                        metadata={"tid": tid},
                        embedding=[0.1] * 128,
                    )
                    repo.add_chunks([chunk])
                    c = repo.get_chunk(cid)
                    assert c is not None
            except Exception as e:
                errors.append((tid, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futs = [executor.submit(worker, i) for i in range(n_threads)]
            concurrent.futures.wait(futs)

        assert not errors, f"Postgres live concurrency errors: {errors}"
        assert repo.count() == n_threads * ops_per_thread
        repo.clear()
    finally:
        repo.close()


def test_postgres_mismatched_dimension_behavior():
    """Probe PostgresStorageRepository behavior when chunk with mismatched dimension is inserted."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="stress_pg_dim_")
    os.close(fd)
    try:
        repo = PostgresStorageRepository(
            host="127.0.0.1",
            port=59998,
            fallback_to_sqlite=True,
            sqlite_fallback_path=path,
            dimension=384,
        )
        bad_chunk = DocumentChunk(
            id="bad_pg_chunk",
            doc_id="doc_bad",
            text="Bad dimension",
            metadata={},
            embedding=[0.1] * 10,
        )
        # Should either reject at add_chunks or fail on get_all_embeddings
        try:
            repo.add_chunks([bad_chunk])
            repo.get_all_embeddings()
        except ValueError as e:
            print(f"\n[PG DIM MISMATCH CAUGHT]: {e}")
        repo.close()
    finally:
        try:
            repo.close()
        except Exception:
            pass
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
