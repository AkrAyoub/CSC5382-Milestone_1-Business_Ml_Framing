from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


SPACE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SPACE_ROOT
for parent in [SPACE_ROOT, *SPACE_ROOT.parents]:
    if (parent / "Milestone_5-ML_Productionization").exists():
        REPO_ROOT = parent
        break

M4_SRC = REPO_ROOT / "Milestone_4-Model_Dev" / "src"
M5_SRC = REPO_ROOT / "Milestone_5-ML_Productionization" / "src"
for path in [M5_SRC, M4_SRC]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m5_productionization.api.schemas import InputSource, ServingMode, SolveRequest
from m5_productionization.service import ProductionizationService


@st.cache_resource
def get_service() -> ProductionizationService:
    return ProductionizationService()


def render_candidate(title: str, payload) -> None:
    st.subheader(title)
    st.write({"status": payload.status, "name": payload.name, "kind": payload.kind})
    if payload.objective is not None:
        st.metric("Objective", f"{payload.objective:.3f}")
    if payload.gap_vs_best_known_pct is not None:
        st.metric("Gap vs best known (%)", f"{payload.gap_vs_best_known_pct:.6f}")
    if payload.gap_vs_baseline_pct is not None:
        st.metric("Gap vs baseline (%)", f"{payload.gap_vs_baseline_pct:.6f}")
    st.write("Open facilities:", payload.open_facilities)
    st.write("Assignment preview:", payload.assignments_preview)
    if payload.error:
        st.error(payload.error)
    for note in payload.notes:
        st.caption(note)


def main() -> None:
    service = get_service()
    st.title("UFLP Production Solver")
    st.caption("Milestone 5 deployed Streamlit client. Baseline solving is available without secrets; LLM mode uses OPENAI_API_KEY.")

    runtime = service.get_runtime_info()
    st.info(f"Safe default: {runtime.safe_default_mode.value} | Default LLM candidate: {runtime.default_llm_candidate_name}")
    for warning in runtime.warnings:
        st.warning(warning)

    catalog = service.list_instances()
    candidate_names = [candidate.name for candidate in runtime.available_candidates if candidate.kind == "llm"]

    selected_instance = st.selectbox(
        "Instance",
        catalog,
        format_func=lambda item: f"{item['instance_id']} (m={item['facility_count_m']}, n={item['customer_count_n']})",
    )
    mode = st.selectbox("Serving mode", ["auto", "baseline", "llm", "compare"], index=0)
    candidate = st.selectbox(
        "LLM candidate",
        candidate_names,
        index=candidate_names.index(runtime.default_llm_candidate_name)
        if runtime.default_llm_candidate_name in candidate_names
        else 0,
    )
    return_assignments = st.checkbox("Return full assignments", value=False)
    return_generated_code = st.checkbox("Return generated code", value=False)

    if st.button("Solve", type="primary"):
        request = SolveRequest(
            mode=ServingMode(mode),
            source=InputSource.CATALOG,
            instance_id=str(selected_instance["instance_id"]),
            candidate_name=candidate,
            return_assignments=return_assignments,
            return_generated_code=return_generated_code,
        )
        response = service.solve(request)
        st.success(f"Status: {response.overall_status}")
        for warning in response.warnings:
            st.warning(warning)
        render_candidate("Baseline", response.baseline)
        if response.candidate:
            render_candidate("Candidate", response.candidate)
        st.json(response.model_dump(mode="json"))


if __name__ == "__main__":
    main()
