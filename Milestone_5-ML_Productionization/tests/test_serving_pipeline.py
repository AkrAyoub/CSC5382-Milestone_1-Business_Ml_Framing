from __future__ import annotations

from m5_productionization.serving_pipeline import run_api_tests_step, validate_api_step


def test_validate_api_step() -> None:
    result = validate_api_step()
    assert result["health_status_code"] == 200
    assert result["predict_status_code"] == 200
    assert result["baseline_objective"] > 0.0


def test_run_api_tests_step() -> None:
    result = run_api_tests_step()
    assert result["integration_tests_total"] == 12
    assert result["integration_tests_failed"] == 0
    assert result["integration_pass_rate"] == 1.0
