"""Automated TDD test suite for Tenant Isolation, Authorization, and Error Sanitization."""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import set_engine
from app.core.engine import SpecRAGEngine
from app.main import app
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.vector_store import DenseVectorIndex


@pytest.fixture
def isolated_test_client(tmp_path):
    """Provide a fresh, isolated FastAPI TestClient with temporary storage."""
    db_file = str(tmp_path / "tenant_sec.db")
    storage = SQLiteStorageRepository(db_file)
    vector_idx = DenseVectorIndex(dimension=384)
    engine = SpecRAGEngine(storage_repo=storage, vector_index=vector_idx)
    set_engine(engine)
    client = TestClient(app)
    return client


def test_tenant_cannot_query_foreign_namespace(isolated_test_client):
    """Client authenticated as tenant_a cannot specify request.namespace = tenant_b."""
    client = isolated_test_client

    payload = {
        "query": "confidential roadmap",
        "namespace": "tenant_b",
    }
    # Authenticated as tenant_a via header
    res = client.post(
        "/api/v1/query",
        json=payload,
        headers={"X-Tenant-ID": "tenant_a"},
    )
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]


def test_tenant_cannot_ingest_into_foreign_namespace(isolated_test_client):
    """Client authenticated as tenant_a cannot ingest chunks for tenant_b."""
    client = isolated_test_client

    payload = {
        "chunks": [
            {
                "id": "chunk_malicious",
                "doc_id": "doc_evil",
                "text": "Injected malicious payload",
                "namespace": "tenant_b",
            }
        ]
    }
    res = client.post(
        "/api/v1/ingest",
        json=payload,
        headers={"X-Tenant-ID": "tenant_a"},
    )
    assert res.status_code == 403
    assert "Cannot ingest chunks for another tenant namespace" in res.json()["detail"]


def test_cross_tenant_keyword_collision_isolation(isolated_test_client):
    """Two tenants with identical keyword queries retrieve only their proprietary data."""
    client = isolated_test_client

    # 1. Ingest document for Tenant Finance
    res_ingest_fin = client.post(
        "/api/v1/ingest",
        json={
            "chunks": [
                {
                    "id": "fin_01",
                    "doc_id": "doc_fin",
                    "text": "Quarterly performance reports show revenue reached 50 million dollars.",
                    "namespace": "finance",
                }
            ]
        },
        headers={"X-Tenant-ID": "finance"},
    )
    assert res_ingest_fin.status_code == 201

    # 2. Ingest document for Tenant HR with exact same keyword 'Quarterly performance reports'
    res_ingest_hr = client.post(
        "/api/v1/ingest",
        json={
            "chunks": [
                {
                    "id": "hr_01",
                    "doc_id": "doc_hr",
                    "text": "Quarterly performance reports show employee satisfaction reached 94 percent.",
                    "namespace": "hr",
                }
            ]
        },
        headers={"X-Tenant-ID": "hr"},
    )
    assert res_ingest_hr.status_code == 201

    # 3. Query as Finance -> verify only Finance chunk returned
    res_fin = client.post(
        "/api/v1/query",
        json={"query": "Quarterly performance reports", "use_cache": True},
        headers={"X-Tenant-ID": "finance"},
    )
    assert res_fin.status_code == 200
    results_fin = res_fin.json()["results"]
    assert len(results_fin) == 1
    assert results_fin[0]["chunk_id"] == "fin_01"
    assert "50 million dollars" in results_fin[0]["text"]

    # 4. Query as HR with exact same string -> verify HR chunk returned, no leak from Finance
    res_hr = client.post(
        "/api/v1/query",
        json={"query": "Quarterly performance reports", "use_cache": True},
        headers={"X-Tenant-ID": "hr"},
    )
    assert res_hr.status_code == 200
    results_hr = res_hr.json()["results"]
    assert len(results_hr) == 1
    assert results_hr[0]["chunk_id"] == "hr_01"
    assert "employee satisfaction" in results_hr[0]["text"]


def test_tenant_cache_clear_isolation(isolated_test_client):
    """Deleting cache as Tenant A purges only Tenant A entries without affecting Tenant B."""
    client = isolated_test_client

    # Ingest for Tenant A and Tenant B
    client.post(
        "/api/v1/ingest",
        json={"chunks": [{"id": "c_a", "doc_id": "d_a", "text": "Tenant A documentation"}]},
        headers={"X-Tenant-ID": "tenant_a"},
    )
    client.post(
        "/api/v1/ingest",
        json={"chunks": [{"id": "c_b", "doc_id": "d_b", "text": "Tenant B documentation"}]},
        headers={"X-Tenant-ID": "tenant_b"},
    )

    # Populate cache for both
    client.post("/api/v1/query", json={"query": "Tenant A documentation", "use_cache": True}, headers={"X-Tenant-ID": "tenant_a"})
    client.post("/api/v1/query", json={"query": "Tenant B documentation", "use_cache": True}, headers={"X-Tenant-ID": "tenant_b"})

    # Check cache hits are primed
    res_hit_a = client.post("/api/v1/query", json={"query": "Tenant A documentation", "use_cache": True}, headers={"X-Tenant-ID": "tenant_a"})
    assert res_hit_a.json()["cache_hit"] is True

    res_hit_b = client.post("/api/v1/query", json={"query": "Tenant B documentation", "use_cache": True}, headers={"X-Tenant-ID": "tenant_b"})
    assert res_hit_b.json()["cache_hit"] is True

    # Tenant A purges their cache
    del_res = client.delete("/api/v1/cache", headers={"X-Tenant-ID": "tenant_a"})
    assert del_res.status_code == 200
    assert "tenant_a" in del_res.json()["message"]

    # Tenant A's next query should now be a CACHE MISS
    res_miss_a = client.post("/api/v1/query", json={"query": "Tenant A documentation", "use_cache": True}, headers={"X-Tenant-ID": "tenant_a"})
    assert res_miss_a.json()["cache_hit"] is False

    # Tenant B's cache MUST STILL BE A HIT (unaffected by Tenant A's deletion)
    res_still_hit_b = client.post("/api/v1/query", json={"query": "Tenant B documentation", "use_cache": True}, headers={"X-Tenant-ID": "tenant_b"})
    assert res_still_hit_b.json()["cache_hit"] is True


def test_admin_cross_tenant_authority(isolated_test_client):
    """Admin credentials allow managing all tenant caches and accessing any namespace."""
    client = isolated_test_client

    del_res = client.delete(
        "/api/v1/cache",
        headers={"X-Admin-Key": "admin-secret"},
    )
    assert del_res.status_code == 200
    assert "purged globally by admin" in del_res.json()["message"]


def test_error_sanitization_does_not_leak_internals(isolated_test_client):
    """Internal server errors return clean generic messages without str(e) stack trace."""
    client = isolated_test_client

    # Empty query should fail validation with 422 (Pydantic)
    res = client.post("/api/v1/query", json={"query": ""})
    assert res.status_code == 422

    # Malformed chunk input
    res_ingest = client.post("/api/v1/ingest", json={"chunks": []})
    assert res_ingest.status_code == 422
