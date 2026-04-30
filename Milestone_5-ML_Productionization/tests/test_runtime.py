from __future__ import annotations

from pathlib import Path

from m5_productionization.api.schemas import InputSource, ServingMode, SolveRequest
from m5_productionization.catalog import read_instance_text
from m5_productionization.runtime import build_runtime_info
from m5_productionization.service import ProductionizationService


def test_runtime_info_defaults() -> None:
    runtime_info = build_runtime_info()
    assert runtime_info.safe_default_mode == ServingMode.BASELINE
    assert runtime_info.default_llm_candidate_name == "openai_gpt41_mini_finetuned"
    assert any(candidate.name == "openai_gpt41_mini_finetuned" for candidate in runtime_info.available_candidates)


def test_inline_baseline_matches_catalog_baseline() -> None:
    service = ProductionizationService()
    catalog_response = service.solve(
        SolveRequest(mode=ServingMode.BASELINE, source=InputSource.CATALOG, instance_id="cap71")
    )
    inline_response = service.solve(
        SolveRequest(
            mode=ServingMode.BASELINE,
            source=InputSource.INLINE,
            instance_text=read_instance_text("cap71"),
        )
    )
    assert catalog_response.baseline.objective is not None
    assert inline_response.baseline.objective is not None
    assert abs(catalog_response.baseline.objective - inline_response.baseline.objective) < 1e-6
