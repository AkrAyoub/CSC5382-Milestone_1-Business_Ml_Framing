from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    kind: str
    backend: str | None
    model_name: str | None
    prompt_template: str | None
    temperature: float
    max_tokens: int
    enabled: bool = True
    notes: str = ""


DEFAULT_CANDIDATES: dict[str, CandidateSpec] = {
    "deterministic_baseline": CandidateSpec(
        name="deterministic_baseline",
        kind="deterministic_baseline",
        backend=None,
        model_name=None,
        prompt_template=None,
        temperature=0.0,
        max_tokens=0,
        notes="Trusted CBC solver reference.",
    ),
    "llm_token_prompt_v0": CandidateSpec(
        name="llm_token_prompt_v0",
        kind="llm",
        backend="self_hosted_openai",
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        prompt_template="token_v0",
        temperature=0.0,
        max_tokens=650,
        notes="Token-stream prompt evaluated against a self-hosted OpenAI-compatible code model runtime.",
    ),
    "llm_robust_prompt_v1": CandidateSpec(
        name="llm_robust_prompt_v1",
        kind="llm",
        backend="self_hosted_openai",
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        prompt_template="robust_v1",
        temperature=0.0,
        max_tokens=850,
        notes="Improved robust parser prompt evaluated against a self-hosted OpenAI-compatible code model runtime.",
    ),
    "llm_fine_tuned": CandidateSpec(
        name="llm_fine_tuned",
        kind="llm",
        backend="self_hosted_openai",
        model_name=None,
        prompt_template="robust_v1",
        temperature=0.0,
        max_tokens=850,
        enabled=False,
        notes="Optional self-hosted fine-tuned slot resolved from environment after SFT/LoRA training.",
    ),
}


def _first_env(*names: str) -> str | None:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def _apply_runtime_env_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("backend") != "self_hosted_openai":
        return payload

    payload = dict(payload)
    payload["model_name"] = _first_env(
        "SELF_HOSTED_MODEL_NAME",
        "M4_SELF_HOSTED_MODEL_NAME",
        "M5_SELF_HOSTED_MODEL_NAME",
    ) or payload.get("model_name")

    base_url = _first_env(
        "SELF_HOSTED_OPENAI_BASE_URL",
        "M4_SELF_HOSTED_BASE_URL",
        "M5_SELF_HOSTED_BASE_URL",
    )
    if base_url:
        payload["notes"] = (
            f"{payload.get('notes', '')} Runtime endpoint: {base_url}."
        ).strip()

    if payload.get("name") == "llm_fine_tuned":
        fine_tuned_model_name = _first_env(
            "SELF_HOSTED_FINE_TUNED_MODEL_NAME",
            "M4_SELF_HOSTED_FINE_TUNED_MODEL_NAME",
            "M5_SELF_HOSTED_FINE_TUNED_MODEL_NAME",
        )
        if fine_tuned_model_name:
            payload["model_name"] = fine_tuned_model_name
            payload["enabled"] = True

    return payload


def list_supported_candidate_names() -> list[str]:
    return sorted(DEFAULT_CANDIDATES.keys())


def _merged_candidate_payload(candidate_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if candidate_name not in DEFAULT_CANDIDATES:
        raise ValueError(f"Unsupported candidate '{candidate_name}'. Supported: {list_supported_candidate_names()}")

    payload = asdict(DEFAULT_CANDIDATES[candidate_name])
    if overrides:
        payload.update({key: value for key, value in overrides.items() if value is not None})
    return _apply_runtime_env_defaults(payload)


def resolve_candidate_spec(candidate_name: str, overrides: dict[str, Any] | None = None) -> CandidateSpec:
    payload = _merged_candidate_payload(candidate_name, overrides)
    return CandidateSpec(
        name=str(payload["name"]),
        kind=str(payload["kind"]),
        backend=str(payload["backend"]) if payload["backend"] else None,
        model_name=str(payload["model_name"]) if payload["model_name"] else None,
        prompt_template=str(payload["prompt_template"]) if payload["prompt_template"] else None,
        temperature=float(payload.get("temperature", 0.0)),
        max_tokens=int(payload.get("max_tokens", 650)),
        enabled=bool(payload.get("enabled", True)),
        notes=str(payload.get("notes", "")),
    )


def resolve_single_candidate_config(config: dict[str, Any]) -> CandidateSpec:
    candidate_block = dict(config.get("candidate", {}))
    candidate_name = str(candidate_block.pop("name", "deterministic_baseline"))
    return resolve_candidate_spec(candidate_name, candidate_block)


def resolve_comparison_candidates(config: dict[str, Any]) -> list[CandidateSpec]:
    comparison = dict(config.get("comparison", {}))
    candidate_names = comparison.get("candidate_names", ["deterministic_baseline", "llm_robust_prompt_v1"])
    results: list[CandidateSpec] = []
    for name in candidate_names:
        overrides = comparison.get(str(name), {})
        results.append(resolve_candidate_spec(str(name), dict(overrides)))
    return results


def serialize_candidate_spec(spec: CandidateSpec, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        __import__("json").dumps(asdict(spec), indent=2),
        encoding="utf-8",
    )
    return output_path
