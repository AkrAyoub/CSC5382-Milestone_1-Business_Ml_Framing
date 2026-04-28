from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from m5_productionization.api.schemas import (
    BatchSolveRequest,
    BatchSolveResponse,
    CandidateExecutionResponse,
    InputSource,
    RuntimeInfoResponse,
    ServiceInfoResponse,
    SolveRequest,
    SolveResponse,
    ServingMode,
)
from m5_productionization.catalog import list_catalog_entries, resolve_catalog_entry
from m5_productionization.paths import M5_TEMP_INPUT_DIR, ensure_runtime_dirs
from m5_productionization.runtime import (
    DEPLOYMENT_TARGET,
    PRODUCTION_DEFAULT_MODE,
    build_runtime_info,
    run_baseline_runtime,
    run_llm_runtime,
)


@dataclass(frozen=True)
class MaterializedInstance:
    source: InputSource
    instance_id: str
    instance_path: Path
    best_known_objective: float | None
    cleanup_path: Path | None = None


class ProductionizationService:
    def __init__(self) -> None:
        ensure_runtime_dirs()

    def get_service_info(self) -> ServiceInfoResponse:
        return ServiceInfoResponse(
            service_name="CSC5382 M5 Production Service",
            milestone="Milestone 5 - ML Productionization",
            project_identity="AI-assisted symbolic optimization for UFLP with a deterministic solver as the trusted verifier",
            primary_serving_mode="On demand to a machine via FastAPI",
            frontend_client="Streamlit frontend calling the API",
            runtime="FastAPI/Uvicorn service with deterministic baseline plus optional self-hosted OpenAI-compatible symbolic runtime",
            deployment_target=DEPLOYMENT_TARGET,
        )

    def get_runtime_info(self) -> RuntimeInfoResponse:
        return build_runtime_info()

    def list_instances(self) -> list[dict[str, object]]:
        return list_catalog_entries()

    @contextmanager
    def _materialize_instance(self, request: SolveRequest) -> Iterator[MaterializedInstance]:
        if request.source == InputSource.CATALOG:
            entry = resolve_catalog_entry(request.instance_id or "")
            yield MaterializedInstance(
                source=request.source,
                instance_id=entry.instance_id,
                instance_path=entry.file_path,
                best_known_objective=entry.best_known,
            )
            return

        ensure_runtime_dirs()
        instance_text = request.instance_text or ""
        digest = sha1(instance_text.encode("utf-8")).hexdigest()[:12]
        temp_path = M5_TEMP_INPUT_DIR / f"inline_{digest}.txt"
        temp_path.write_text(instance_text, encoding="utf-8")
        try:
            yield MaterializedInstance(
                source=request.source,
                instance_id=f"inline_{digest}",
                instance_path=temp_path,
                best_known_objective=None,
                cleanup_path=temp_path,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def solve(self, request: SolveRequest) -> SolveResponse:
        request_id = uuid4().hex[:12]
        served_mode = PRODUCTION_DEFAULT_MODE if request.mode == ServingMode.AUTO else request.mode
        production_default_used = request.mode == ServingMode.AUTO
        warnings: list[str] = []

        with self._materialize_instance(request) as materialized:
            baseline = run_baseline_runtime(
                instance_id=materialized.instance_id,
                instance_path=materialized.instance_path,
                return_assignments=request.return_assignments,
            )

            candidate: CandidateExecutionResponse | None = None
            overall_status = "SUCCESS"
            if served_mode in (ServingMode.LLM, ServingMode.COMPARE):
                candidate = run_llm_runtime(
                    request_id=request_id,
                    instance_id=materialized.instance_id,
                    instance_path=materialized.instance_path,
                    baseline_objective=baseline.objective or 0.0,
                    return_assignments=request.return_assignments,
                    return_generated_code=request.return_generated_code,
                    candidate_name=request.candidate_name,
                )
                if candidate.status != "OK":
                    overall_status = "PARTIAL_SUCCESS"
                    warnings.extend(candidate.notes)

            if production_default_used:
                warnings.append("The request used the production-safe default runtime: deterministic_baseline.")

            return SolveResponse(
                request_id=request_id,
                request_label=request.request_label,
                served_mode=served_mode,
                requested_mode=request.mode,
                source=request.source,
                instance_id=materialized.instance_id,
                overall_status=overall_status,
                production_default_used=production_default_used,
                best_known_objective=materialized.best_known_objective,
                warnings=warnings,
                baseline=baseline,
                candidate=candidate,
            )

    def solve_batch(self, request: BatchSolveRequest) -> BatchSolveResponse:
        batch_id = uuid4().hex[:12]
        responses: list[SolveResponse] = []
        success_count = 0
        partial_count = 0
        failed_count = 0

        for item in request.items:
            response = self.solve(
                SolveRequest(
                    mode=request.mode,
                    source=item.source,
                    instance_id=item.instance_id,
                    instance_text=item.instance_text,
                    candidate_name=request.candidate_name,
                    return_assignments=request.return_assignments,
                    return_generated_code=request.return_generated_code,
                    request_label=item.request_label,
                )
            )
            responses.append(response)
            if response.overall_status == "SUCCESS":
                success_count += 1
            elif response.overall_status == "PARTIAL_SUCCESS":
                partial_count += 1
            else:
                failed_count += 1

        return BatchSolveResponse(
            batch_id=batch_id,
            requested_mode=request.mode,
            response_count=len(responses),
            success_count=success_count,
            partial_count=partial_count,
            failed_count=failed_count,
            responses=responses,
        )
