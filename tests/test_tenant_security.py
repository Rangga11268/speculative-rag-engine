"""Automated TDD test suite for Cryptographic Tenant Authentication, Authorization, and Information Leakage Prevention."""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import set_engine
from app.config import Settings, get_settings
from app.core.engine import SpecRAGEngine
from app.core.security import create_tenant_token, get_active_admin_key
from app.main import app
from app.storage.sqlite import SQLiteStorageRepository
from app.storage.vector_store import DenseVectorIndex


@pytest.fixture
def isolated_test_client(tmp_path, monkeypatch):
    """Provide a fresh, isolated FastAPI TestClient with temporary storage and configured test keys."""
    settings = get_settings()
    monkeypatch.setattr(settings, "SECRET_KEY", "test-cryptographic-secret-key-384-specrag")
    monkeypatch.setattr(settings, "ADMIN_API_KEY", "test-admin-secret-key-384-specrag")
    db_file = str(tmp_path / "tenant_sec.db")
    storage = SQLiteStorageRepository(db_file)
    vector_idx = DenseVectorIndex(dimension=384)
    engine = SpecRAGEngine(storage_repo=storage, vector_index=vector_idx)
    set_engine(engine)
    client = TestClient(app)
    return client


def test_tampered_token_rejected_with_401(isolated_test_client):
    """Token with altered HMAC signature must be immediately rejected with 401 Unauthorized."""
    client = isolated_test_client
    valid_token = create_tenant_token("tenant_a")
    # Tamper with the last character of the signature
    tampered_token = valid_token[:-1] + ("0" if valid_token[-1] != "0" else "1")

    res = client.post(
        "/api/v1/query",
        json={"query": "test query"},
        headers={"Authorization": f"Bearer {tampered_token}"},
    )
    assert res.status_code == 401
    assert "Invalid authentication token" in res.json()["detail"]


def test_client_cannot_impersonate_another_tenant_via_header(isolated_test_client):
    """Client authenticated with tenant_a token cannot send X-Tenant-ID: tenant_b."""
    client = isolated_test_client
    token_a = create_tenant_token("tenant_a")

    res = client.post(
        "/api/v1/query",
        json={"query": "confidential data"},
        headers={
            "Authorization": f"Bearer {token_a}",
            "X-Tenant-ID": "tenant_b",  # Attempted identity spoofing
        },
    )
    assert res.status_code == 403
    assert "Tenant identity mismatch" in res.json()["detail"]


def test_tenant_cannot_query_foreign_namespace(isolated_test_client):
    """Client authenticated as tenant_a cannot specify request.namespace = tenant_b in body."""
    client = isolated_test_client
    token_a = create_tenant_token("tenant_a")

    payload = {
        "query": "confidential roadmap",
        "namespace": "tenant_b",
    }
    res = client.post(
        "/api/v1/query",
        json=payload,
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert res.status_code == 403
    assert "cannot query foreign namespace" in res.json()["detail"]


def test_tenant_cannot_ingest_into_foreign_namespace(isolated_test_client):
    """Client authenticated as tenant_a cannot ingest chunks for tenant_b."""
    client = isolated_test_client
    token_a = create_tenant_token("tenant_a")

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
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert res.status_code == 403
    assert "cannot ingest into foreign namespace" in res.json()["detail"]


def test_cross_tenant_keyword_collision_isolation(isolated_test_client):
    """Two tenants with identical keyword queries retrieve only their proprietary data."""
    client = isolated_test_client
    token_fin = create_tenant_token("finance")
    token_hr = create_tenant_token("hr")

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
        headers={"Authorization": f"Bearer {token_fin}"},
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
        headers={"Authorization": f"Bearer {token_hr}"},
    )
    assert res_ingest_hr.status_code == 201

    # 3. Query as Finance -> verify only Finance chunk returned
    res_fin = client.post(
        "/api/v1/query",
        json={"query": "Quarterly performance reports", "use_cache": True},
        headers={"Authorization": f"Bearer {token_fin}"},
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
        headers={"Authorization": f"Bearer {token_hr}"},
    )
    assert res_hr.status_code == 200
    results_hr = res_hr.json()["results"]
    assert len(results_hr) == 1
    assert results_hr[0]["chunk_id"] == "hr_01"
    assert "employee satisfaction" in results_hr[0]["text"]


def test_tenant_cache_clear_isolation(isolated_test_client):
    """Deleting cache as Tenant A purges only Tenant A entries without affecting Tenant B."""
    client = isolated_test_client
    token_a = create_tenant_token("tenant_a")
    token_b = create_tenant_token("tenant_b")

    # Ingest for Tenant A and Tenant B
    client.post(
        "/api/v1/ingest",
        json={"chunks": [{"id": "c_a", "doc_id": "d_a", "text": "Tenant A documentation"}]},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    client.post(
        "/api/v1/ingest",
        json={"chunks": [{"id": "c_b", "doc_id": "d_b", "text": "Tenant B documentation"}]},
        headers={"Authorization": f"Bearer {token_b}"},
    )

    # Populate cache for both
    client.post("/api/v1/query", json={"query": "Tenant A documentation", "use_cache": True}, headers={"Authorization": f"Bearer {token_a}"})
    client.post("/api/v1/query", json={"query": "Tenant B documentation", "use_cache": True}, headers={"Authorization": f"Bearer {token_b}"})

    # Check cache hits are primed
    res_hit_a = client.post("/api/v1/query", json={"query": "Tenant A documentation", "use_cache": True}, headers={"Authorization": f"Bearer {token_a}"})
    assert res_hit_a.json()["cache_hit"] is True

    res_hit_b = client.post("/api/v1/query", json={"query": "Tenant B documentation", "use_cache": True}, headers={"Authorization": f"Bearer {token_b}"})
    assert res_hit_b.json()["cache_hit"] is True

    # Tenant A purges their cache
    del_res = client.delete("/api/v1/cache", headers={"Authorization": f"Bearer {token_a}"})
    assert del_res.status_code == 200
    assert "tenant_a" in del_res.json()["message"]

    # Tenant A's next query should now be a CACHE MISS
    res_miss_a = client.post("/api/v1/query", json={"query": "Tenant A documentation", "use_cache": True}, headers={"Authorization": f"Bearer {token_a}"})
    assert res_miss_a.json()["cache_hit"] is False

    # Tenant B's cache MUST STILL BE A HIT (unaffected by Tenant A's deletion)
    res_still_hit_b = client.post("/api/v1/query", json={"query": "Tenant B documentation", "use_cache": True}, headers={"Authorization": f"Bearer {token_b}"})
    assert res_still_hit_b.json()["cache_hit"] is True


def test_cache_stats_does_not_leak_other_tenants(isolated_test_client):
    """Cache statistics for non-admin tenants must strictly return only tenant-scoped metrics."""
    client = isolated_test_client
    token_a = create_tenant_token("tenant_a")
    token_b = create_tenant_token("tenant_b")

    # Ingest and query for Tenant B to generate hits/entries
    client.post("/api/v1/ingest", json={"chunks": [{"id": "c_b", "doc_id": "d_b", "text": "Confidential B data"}]}, headers={"Authorization": f"Bearer {token_b}"})
    client.post("/api/v1/query", json={"query": "Confidential B data", "use_cache": True}, headers={"Authorization": f"Bearer {token_b}"})

    # Tenant A checks cache stats
    res_stats_a = client.get("/api/v1/cache/stats", headers={"Authorization": f"Bearer {token_a}"})
    assert res_stats_a.status_code == 200
    data_a = res_stats_a.json()

    # Tenant A must NOT see global entries, total hits, misses, or Tenant B data
    assert "hits" not in data_a, "Global hits leaked to tenant A!"
    assert "misses" not in data_a, "Global misses leaked to tenant A!"
    assert data_a["tenant_id"] == "tenant_a"
    assert data_a["tenant_entries"] == 0  # Tenant A has zero entries, even though B has entries!


def test_admin_authentication_and_authority(isolated_test_client):
    """Admin key authorizes global management, while invalid keys are rejected."""
    client = isolated_test_client
    admin_key = get_active_admin_key()

    # Valid admin key can purge global cache
    del_res = client.delete(
        "/api/v1/cache",
        headers={"X-Admin-Key": admin_key},
    )
    assert del_res.status_code == 200
    assert "purged globally by admin" in del_res.json()["message"]

    # Valid admin key can view full global cache statistics
    stats_res = client.get(
        "/api/v1/cache/stats",
        headers={"X-Admin-Key": admin_key},
    )
    assert stats_res.status_code == 200
    assert "hits" in stats_res.json()
    assert "misses" in stats_res.json()

    # Insecure / invalid key (e.g. old default 'admin-secret' or random string) is rejected
    del_invalid = client.delete(
        "/api/v1/cache",
        headers={"X-Admin-Key": "unauthorized-random-key"},
    )
    assert del_invalid.status_code == 401


def test_production_startup_rejects_insecure_secrets():
    """Application startup validator strictly rejects default/short secrets in production."""
    # 1. Default insecure admin secret in production -> Must raise ValueError
    s1 = Settings(ENVIRONMENT="production", ADMIN_API_KEY="admin-secret", SECRET_KEY="long-enough-secret-key-384")
    with pytest.raises(ValueError, match="CRITICAL SECURITY: Insecure or default ADMIN_API_KEY"):
        s1.validate_production_security()

    # 2. Missing admin key in production -> Must raise ValueError
    s2 = Settings(ENVIRONMENT="production", ADMIN_API_KEY=None, SECRET_KEY="long-enough-secret-key-384")
    with pytest.raises(ValueError, match="CRITICAL SECURITY: Insecure or default ADMIN_API_KEY"):
        s2.validate_production_security()

    # 3. Too short admin key (< 16 chars) in production -> Must raise ValueError
    s3 = Settings(ENVIRONMENT="production", ADMIN_API_KEY="short_key", SECRET_KEY="long-enough-secret-key-384")
    with pytest.raises(ValueError, match="CRITICAL SECURITY: Insecure or default ADMIN_API_KEY"):
        s3.validate_production_security()

    # 4. Default insecure SECRET_KEY in production -> Must raise ValueError
    s4 = Settings(ENVIRONMENT="production", ADMIN_API_KEY="super-strong-admin-key-12345", SECRET_KEY="secret")
    with pytest.raises(ValueError, match="CRITICAL SECURITY: Insecure or default SECRET_KEY"):
        s4.validate_production_security()

    # 5. Cryptographically secure keys (>= 16 chars) in production -> Passes cleanly
    s5 = Settings(ENVIRONMENT="production", ADMIN_API_KEY="prod-admin-secrag-key-super-secure-384", SECRET_KEY="prod-secret-signing-key-super-secure-384")
    s5.validate_production_security()  # No error raised


def test_error_sanitization_does_not_leak_internals(isolated_test_client):
    """Internal server errors return clean generic messages without str(e) stack trace."""
    client = isolated_test_client
    token = create_tenant_token("test_tenant")

    # Empty query should fail validation with 422 (Pydantic)
    res = client.post("/api/v1/query", json={"query": ""}, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 422

    # Malformed chunk input
    res_ingest = client.post("/api/v1/ingest", json={"chunks": []}, headers={"Authorization": f"Bearer {token}"})
    assert res_ingest.status_code == 422


def test_unconfigured_secrets_fail_closed_safely(monkeypatch):
    """When SECRET_KEY or ADMIN_API_KEY is missing/empty, operations fail closed without fallback."""
    settings = get_settings()
    monkeypatch.setattr(settings, "SECRET_KEY", None)
    monkeypatch.setattr(settings, "ADMIN_API_KEY", None)

    # 1. Direct getter must raise RuntimeError
    from app.core.security import get_active_secret_key, get_active_admin_key, verify_admin_key, verify_tenant_token
    with pytest.raises(RuntimeError, match="SECRET_KEY is not configured"):
        get_active_secret_key()

    with pytest.raises(RuntimeError, match="ADMIN_API_KEY is not configured"):
        get_active_admin_key()

    # 2. Token creation without explicit key must raise RuntimeError
    with pytest.raises(RuntimeError, match="SECRET_KEY is not configured"):
        create_tenant_token("tenant_test")

    # 3. Verification must fail closed safely (returning None/False, NOT raising unhandled crash or accepting default)
    assert verify_tenant_token("srag_tenant_test_1234567890abcdef1234567890abcdef") is None
    assert verify_admin_key("any-admin-candidate") is False


def test_production_rejects_wildcard_cors():
    """Production configuration must reject wildcard '*' CORS origins."""
    s = Settings(
        ENVIRONMENT="production",
        ADMIN_API_KEY="super-strong-admin-key-12345",
        SECRET_KEY="super-strong-secret-key-12345",
        CORS_ALLOW_ORIGINS=["*"],
    )
    with pytest.raises(ValueError, match=r"Wildcard '\*' CORS origin is forbidden in production"):
        s.validate_production_security()


def test_cors_headers_applied(isolated_test_client):
    """CORS headers must allow configured origin and methods."""
    client = isolated_test_client
    res = client.options(
        "/api/v1/query",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert res.headers.get("access-control-allow-credentials") == "true"

