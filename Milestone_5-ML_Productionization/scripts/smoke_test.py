from __future__ import annotations

from fastapi.testclient import TestClient

from m5_productionization.api.main import app


if __name__ == "__main__":
    client = TestClient(app)
    health = client.get("/healthz")
    catalog = client.get("/api/v1/catalog/instances")
    solve = client.post("/api/v1/solve", json={"mode": "baseline", "source": "catalog", "instance_id": "cap71"})
    print("health:", health.status_code, health.json())
    print("catalog_count:", len(catalog.json()))
    print("solve_status:", solve.status_code, solve.json()["overall_status"])
