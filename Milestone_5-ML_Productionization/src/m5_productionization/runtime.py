from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from m5_productionization.api.schemas import CandidateExecutionResponse, RuntimeCandidateSummary, RuntimeInfoResponse, ServingMode
from m5_productionization.paths import (
    M4_MODEL_SELECTION_PATH,
    M5_GENERATED_CODE_DIR,
    UNCAPOPT_PATH,
    ensure_runtime_dirs,
)

from m4_model_dev.data.benchmark import parse_orlib_uncap, parse_uncapopt, solve_reference_cbc
from m4_model_dev.evaluation.generated_exec import run_generated_solver
from m4_model_dev.models.model_registry import DEFAULT_CANDIDATES, resolve_candidate_spec
from m4_model_dev.models.symbolic_generator import generate_solver_code


PRODUCTION_DEFAULT_MODE = ServingMode.BASELINE
DEFAULT_LLM_CANDIDATE_NAME = os.getenv("M5_DEFAULT_LLM_CANDIDATE", "openai_gpt41_mini_finetuned")
DEPLOYMENT_TARGET = "FastAPI + Streamlit + MLflow pyfunc runtime, deployable with Docker Compose or Hugging Face Spaces"


def _best_known_lookup() -> dict[str, float]:
    return parse_uncapopt(UNCAPOPT_PATH) if UNCAPOPT_PATH.exists() else {}


def _objective_gap_pct(candidate_objective: float | None, reference_objective: float | None) -> float | None:
    if candidate_objective is None or reference_objective is None:
        return None
    return (candidate_objective - reference_objective) / max(1.0, abs(reference_objective)) * 100.0


def _assignments_payload(assignments: list[int], return_assignments: bool) -> tuple[list[int], list[int] | None]:
    preview = assignments[:20]
    return preview, list(assignments) if return_assignments else None


def _load_selected_candidate_from_m4() -> dict[str, object] | None:
    if not M4_MODEL_SELECTION_PATH.exists():
        return None
    try:
        return json.loads(M4_MODEL_SELECTION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _has_openai_key_available() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    if os.name != "nt":
        return False
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','User') -or "
        "[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','Machine')",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except Exception:
        return False
    return bool(result.stdout.strip())


def build_runtime_info() -> RuntimeInfoResponse:
    selection_payload = _load_selected_candidate_from_m4()
    warnings: list[str] = []
    if selection_payload and float(selection_payload.get("selected_validation_success_rate", 0.0)) <= 0.0:
        warnings.append(
            "The latest non-baseline candidate selected in Milestone 4 has zero validation success; "
            "the deterministic baseline remains the production-safe default runtime."
        )
    if DEFAULT_LLM_CANDIDATE_NAME.startswith("openai") and not _has_openai_key_available():
        warnings.append(
            "The default LLM candidate uses OpenAI. Set OPENAI_API_KEY to enable live LLM or fine-tuned inference."
        )

    candidates = [
        RuntimeCandidateSummary(
            name=name,
            kind=spec.kind,
            backend=spec.backend,
            model_name=spec.model_name,
            prompt_template=spec.prompt_template,
            enabled=spec.enabled,
            notes=spec.notes,
        )
        for name, spec in sorted(DEFAULT_CANDIDATES.items())
    ]
    return RuntimeInfoResponse(
        safe_default_mode=PRODUCTION_DEFAULT_MODE,
        default_llm_candidate_name=DEFAULT_LLM_CANDIDATE_NAME,
        deployment_target=DEPLOYMENT_TARGET,
        supported_serving_modes=["baseline", "llm", "compare", "batch"],
        available_candidates=candidates,
        selected_candidate_from_m4=selection_payload,
        warnings=warnings,
    )


def run_baseline_runtime(instance_id: str, instance_path: Path, return_assignments: bool) -> CandidateExecutionResponse:
    best_known = _best_known_lookup().get(instance_id)
    started = time.time()
    solved = solve_reference_cbc(parse_orlib_uncap(instance_path), best_known=best_known)
    runtime_s = time.time() - started
    assignments_preview, assignments = _assignments_payload(solved.assignments, return_assignments)
    return CandidateExecutionResponse(
        name="deterministic_baseline",
        kind="deterministic_baseline",
        status="OK",
        objective=solved.objective,
        gap_vs_best_known_pct=solved.gap_percent,
        runtime_s=runtime_s,
        open_facilities=solved.open_facilities,
        assignment_count=len(solved.assignments),
        assignments_preview=assignments_preview,
        assignments=assignments,
    )


def _resolve_candidate(candidate_name: str | None):
    selected_name = candidate_name or DEFAULT_LLM_CANDIDATE_NAME
    overrides: dict[str, object] = {}
    if selected_name == "llm_fine_tuned":
        fine_tuned_model_name = (
            (os.getenv("M5_SELF_HOSTED_FINE_TUNED_MODEL_NAME") or "").strip()
            or (os.getenv("SELF_HOSTED_FINE_TUNED_MODEL_NAME") or "").strip()
            or (os.getenv("M5_FINE_TUNED_MODEL_NAME") or "").strip()
        )
        if fine_tuned_model_name:
            overrides = {"model_name": fine_tuned_model_name, "enabled": True}
    return resolve_candidate_spec(selected_name, overrides or None)


def run_llm_runtime(
    *,
    request_id: str,
    instance_id: str,
    instance_path: Path,
    baseline_objective: float,
    return_assignments: bool,
    return_generated_code: bool,
    candidate_name: str | None,
) -> CandidateExecutionResponse:
    ensure_runtime_dirs()
    spec = _resolve_candidate(candidate_name)
    response = CandidateExecutionResponse(
        name=spec.name,
        kind=spec.kind,
        status="UNAVAILABLE" if not spec.enabled else "PENDING",
        backend_name=spec.backend,
        model_name=spec.model_name,
        prompt_template=spec.prompt_template,
    )

    if not spec.enabled:
        response.error = "Candidate is disabled in the registry."
        response.notes.append(
            "Enable a valid candidate or provide M5_SELF_HOSTED_FINE_TUNED_MODEL_NAME for the fine-tuned slot."
        )
        return response

    started = time.time()
    try:
        generated = generate_solver_code(spec)
    except Exception as exc:
        missing_endpoint = "SELF_HOSTED_OPENAI_BASE_URL" in str(exc)
        response.status = "UNAVAILABLE" if missing_endpoint else "FAIL"
        response.error = f"{type(exc).__name__}: {exc}"
        response.runtime_s = time.time() - started
        if response.status == "UNAVAILABLE":
            response.notes.append(
                "Set SELF_HOSTED_OPENAI_BASE_URL (and optionally SELF_HOSTED_OPENAI_API_KEY) "
                "to enable the self-hosted symbolic generation path."
            )
        elif spec.backend == "openai":
            response.notes.append("Set OPENAI_API_KEY to enable the hosted OpenAI symbolic generation path.")
        return response

    code_output_dir = M5_GENERATED_CODE_DIR / request_id
    code_output_dir.mkdir(parents=True, exist_ok=True)
    code_output_path = code_output_dir / f"{spec.name}.py"
    code_output_path.write_text(generated.code, encoding="utf-8")

    exec_started = time.time()
    try:
        executed = run_generated_solver(generated.code, str(instance_path))
    except Exception as exc:
        response.status = "FAIL"
        response.error = f"{type(exc).__name__}: {exc}"
        response.runtime_s = time.time() - started
        response.generated_code = generated.code if return_generated_code else None
        response.notes.append("The generated candidate did not pass sandbox execution or solver-output validation.")
        return response

    execution_runtime_s = time.time() - exec_started
    assignments_preview, assignments = _assignments_payload(executed.assignments, return_assignments)
    response.status = "OK"
    response.objective = executed.objective
    response.gap_vs_baseline_pct = _objective_gap_pct(executed.objective, baseline_objective)
    response.runtime_s = time.time() - started
    response.open_facilities = executed.open_facilities
    response.assignment_count = len(executed.assignments)
    response.assignments_preview = assignments_preview
    response.assignments = assignments
    response.generated_code = generated.code if return_generated_code else None
    response.notes.extend(
        [
            "LLM path executed through the M4 symbolic generator and sandboxed execution runtime.",
            f"Generation and execution completed in {response.runtime_s:.3f}s (execution phase {execution_runtime_s:.3f}s).",
        ]
    )
    return response
