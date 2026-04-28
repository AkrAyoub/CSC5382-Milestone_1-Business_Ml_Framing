from __future__ import annotations

from fastapi.testclient import TestClient

from m5_productionization.api.main import app


client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_catalog_is_available() -> None:
    response = client.get("/api/v1/catalog/instances")
    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert any(item["instance_id"] == "cap71" for item in payload)


def test_baseline_solve_catalog_instance() -> None:
    response = client.post(
        "/api/v1/solve",
        json={"mode": "baseline", "source": "catalog", "instance_id": "cap71"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_status"] == "SUCCESS"
    assert payload["baseline"]["status"] == "OK"
    assert payload["baseline"]["objective"] > 0.0


def test_batch_baseline_solve() -> None:
    response = client.post(
        "/api/v1/batch/solve",
        json={
            "mode": "baseline",
            "items": [
                {"source": "catalog", "instance_id": "cap71"},
                {"source": "catalog", "instance_id": "cap72"},
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["response_count"] == 2
    assert payload["success_count"] == 2
