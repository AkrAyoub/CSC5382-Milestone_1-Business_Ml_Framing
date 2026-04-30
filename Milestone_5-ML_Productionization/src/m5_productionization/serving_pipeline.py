from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from m5_productionization.api.main import app
from m5_productionization.mlflow_runtime import log_runtime_model
from m5_productionization.paths import M5_REPORTS_DIR, ensure_runtime_dirs
from m5_productionization.runtime import build_runtime_info


SERVING_STATUS_JSON = M5_REPORTS_DIR / "serving_pipeline_status.json"
SERVING_STATUS_TXT = M5_REPORTS_DIR / "serving_pipeline_status.txt"


@dataclass(frozen=True)
class ServingStepResult:
    name: str
    status: str
    duration_s: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_s": self.duration_s,
            "details": self.details,
        }


def _run_step(name: str, func) -> ServingStepResult:
    started = time.time()
    try:
        details = func()
        status = "SUCCESS"
    except Exception as exc:
        details = {"error": f"{type(exc).__name__}: {exc}"}
        status = "FAIL"
    return ServingStepResult(name=name, status=status, duration_s=time.time() - started, details=details)


def load_best_model_step() -> dict[str, Any]:
    runtime_info = build_runtime_info()
    selected = runtime_info.selected_candidate_from_m4 or {}
    return {
        "safe_default_mode": runtime_info.safe_default_mode.value,
        "default_llm_candidate_name": runtime_info.default_llm_candidate_name,
        "deployment_target": runtime_info.deployment_target,
        "selected_candidate_from_m4": selected,
        "candidate_count": len(runtime_info.available_candidates),
        "warnings": runtime_info.warnings,
    }


def validate_api_step() -> dict[str, Any]:
    client = TestClient(app)
    started = time.time()
    health = client.get("/healthz")
    runtime = client.get("/api/v1/runtime")
    predict = client.post(
        "/api/v1/solve",
        json={"mode": "baseline", "source": "catalog", "instance_id": "cap71"},
    )
    latency_ms = (time.time() - started) * 1000.0
    health.raise_for_status()
    runtime.raise_for_status()
    predict.raise_for_status()
    payload = predict.json()
    if payload["overall_status"] != "SUCCESS":
        raise RuntimeError(f"Unexpected solve status: {payload['overall_status']}")
    return {
        "health_status_code": health.status_code,
        "runtime_status_code": runtime.status_code,
        "predict_status_code": predict.status_code,
        "latency_ms": latency_ms,
        "baseline_objective": payload["baseline"]["objective"],
    }


def run_api_tests_step() -> dict[str, Any]:
    checks: list[tuple[str, bool, str]] = []
    client = TestClient(app)

    def record(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, condition, detail))

    health = client.get("/healthz")
    record("health endpoint returns 200", health.status_code == 200, str(health.status_code))
    record("health status is ok", health.json().get("status") == "ok", str(health.json()))

    catalog = client.get("/api/v1/catalog/instances")
    catalog_payload = catalog.json()
    record("catalog endpoint returns 200", catalog.status_code == 200, str(catalog.status_code))
    record("catalog has instances", bool(catalog_payload), f"count={len(catalog_payload)}")
    record("catalog includes cap71", any(item.get("instance_id") == "cap71" for item in catalog_payload))

    baseline = client.post("/api/v1/solve", json={"mode": "baseline", "source": "catalog", "instance_id": "cap71"})
    baseline_payload = baseline.json()
    record("baseline solve returns 200", baseline.status_code == 200, str(baseline.status_code))
    record("baseline solve succeeds", baseline_payload.get("overall_status") == "SUCCESS", str(baseline_payload.get("overall_status")))
    record("baseline objective positive", float(baseline_payload["baseline"]["objective"]) > 0.0)

    auto = client.post("/api/v1/solve", json={"mode": "auto", "source": "catalog", "instance_id": "cap71"})
    auto_payload = auto.json()
    record("auto solve returns 200", auto.status_code == 200, str(auto.status_code))
    record("auto uses production default", bool(auto_payload.get("production_default_used")), str(auto_payload.get("production_default_used")))

    batch = client.post(
        "/api/v1/batch/solve",
        json={
            "mode": "baseline",
            "items": [
                {"source": "catalog", "instance_id": "cap71"},
                {"source": "catalog", "instance_id": "cap72"},
            ],
        },
    )
    batch_payload = batch.json()
    record("batch solve returns 200", batch.status_code == 200, str(batch.status_code))
    record("batch solve succeeds", batch_payload.get("success_count") == 2, str(batch_payload))

    failed = [name for name, ok, _detail in checks if not ok]
    if failed:
        raise RuntimeError(f"API checks failed: {failed}")
    return {
        "integration_tests_total": len(checks),
        "integration_tests_passed": len(checks),
        "integration_tests_failed": 0,
        "integration_pass_rate": 1.0,
        "checks": [{"name": name, "passed": ok, "detail": detail} for name, ok, detail in checks],
    }


def register_serving_step() -> dict[str, Any]:
    model_uri, model_name = log_runtime_model(register_model=False)
    return {
        "mlflow_model_uri": model_uri,
        "registered_model_name": model_name,
        "registered_in_registry": bool(model_name),
    }


def run_serving_pipeline() -> dict[str, Any]:
    ensure_runtime_dirs()
    steps = [
        _run_step("load_best_model", load_best_model_step),
        _run_step("validate_api", validate_api_step),
        _run_step("run_api_tests", run_api_tests_step),
        _run_step("register_serving", register_serving_step),
    ]
    success = all(step.status == "SUCCESS" for step in steps)
    payload = {
        "pipeline_name": "m5_serving_pipeline",
        "success": success,
        "step_count": len(steps),
        "steps": [step.to_dict() for step in steps],
    }
    M5_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SERVING_STATUS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "Milestone 5 Serving Pipeline",
        f"success: {success}",
        "",
    ]
    for step in steps:
        lines.append(f"- {step.name}: {step.status} ({step.duration_s:.2f}s)")
        for key, value in step.details.items():
            if key != "checks":
                lines.append(f"  - {key}: {value}")
    SERVING_STATUS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload

