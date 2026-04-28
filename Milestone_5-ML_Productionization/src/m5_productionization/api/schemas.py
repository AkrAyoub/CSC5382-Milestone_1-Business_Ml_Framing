from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ServingMode(str, Enum):
    AUTO = "auto"
    BASELINE = "baseline"
    LLM = "llm"
    COMPARE = "compare"


class InputSource(str, Enum):
    CATALOG = "catalog"
    INLINE = "inline"


class SolveRequest(BaseModel):
    mode: ServingMode = ServingMode.AUTO
    source: InputSource = InputSource.CATALOG
    instance_id: str | None = None
    instance_text: str | None = None
    candidate_name: str | None = None
    return_assignments: bool = False
    return_generated_code: bool = False
    request_label: str | None = None

    @model_validator(mode="after")
    def validate_source_fields(self) -> "SolveRequest":
        if self.source == InputSource.CATALOG and not self.instance_id:
            raise ValueError("instance_id is required when source='catalog'.")
        if self.source == InputSource.INLINE and not self.instance_text:
            raise ValueError("instance_text is required when source='inline'.")
        return self


class BatchSolveItem(BaseModel):
    source: InputSource = InputSource.CATALOG
    instance_id: str | None = None
    instance_text: str | None = None
    request_label: str | None = None

    @model_validator(mode="after")
    def validate_source_fields(self) -> "BatchSolveItem":
        if self.source == InputSource.CATALOG and not self.instance_id:
            raise ValueError("instance_id is required when source='catalog'.")
        if self.source == InputSource.INLINE and not self.instance_text:
            raise ValueError("instance_text is required when source='inline'.")
        return self


class BatchSolveRequest(BaseModel):
    mode: ServingMode = ServingMode.BASELINE
    candidate_name: str | None = None
    return_assignments: bool = False
    return_generated_code: bool = False
    items: list[BatchSolveItem] = Field(default_factory=list)


class CandidateExecutionResponse(BaseModel):
    name: str
    kind: str
    status: str
    backend_name: str | None = None
    model_name: str | None = None
    prompt_template: str | None = None
    objective: float | None = None
    gap_vs_best_known_pct: float | None = None
    gap_vs_baseline_pct: float | None = None
    runtime_s: float = 0.0
    open_facilities: list[int] = Field(default_factory=list)
    assignment_count: int = 0
    assignments_preview: list[int] = Field(default_factory=list)
    assignments: list[int] | None = None
    generated_code: str | None = None
    error: str | None = None
    notes: list[str] = Field(default_factory=list)


class SolveResponse(BaseModel):
    request_id: str
    request_label: str | None = None
    served_mode: ServingMode
    requested_mode: ServingMode
    source: InputSource
    instance_id: str
    overall_status: str
    production_default_used: bool = False
    best_known_objective: float | None = None
    warnings: list[str] = Field(default_factory=list)
    baseline: CandidateExecutionResponse
    candidate: CandidateExecutionResponse | None = None


class BatchSolveResponse(BaseModel):
    batch_id: str
    requested_mode: ServingMode
    response_count: int
    success_count: int
    partial_count: int
    failed_count: int
    responses: list[SolveResponse] = Field(default_factory=list)


class RuntimeCandidateSummary(BaseModel):
    name: str
    kind: str
    backend: str | None = None
    model_name: str | None = None
    prompt_template: str | None = None
    enabled: bool
    notes: str = ""


class RuntimeInfoResponse(BaseModel):
    safe_default_mode: ServingMode
    default_llm_candidate_name: str
    deployment_target: str
    supported_serving_modes: list[str]
    available_candidates: list[RuntimeCandidateSummary]
    selected_candidate_from_m4: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class ServiceInfoResponse(BaseModel):
    service_name: str
    milestone: str
    project_identity: str
    primary_serving_mode: str
    frontend_client: str
    runtime: str
    deployment_target: str
