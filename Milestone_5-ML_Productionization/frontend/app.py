from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("M5_API_BASE_URL", "http://localhost:8000")


def _safe_get_json(url: str) -> Any:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _post_json(url: str, payload: dict[str, Any]) -> Any:
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def render_candidate_block(title: str, payload: dict[str, Any] | None) -> None:
    st.subheader(title)
    if not payload:
        st.info("No candidate payload returned.")
        return

    st.write({"status": payload.get("status"), "name": payload.get("name"), "kind": payload.get("kind")})
    if payload.get("objective") is not None:
        st.metric(f"{title} objective", f"{payload['objective']:.3f}")
    if payload.get("gap_vs_best_known_pct") is not None:
        st.metric(f"{title} gap vs best known (%)", f"{payload['gap_vs_best_known_pct']:.6f}")
    if payload.get("gap_vs_baseline_pct") is not None:
        st.metric(f"{title} gap vs baseline (%)", f"{payload['gap_vs_baseline_pct']:.6f}")
    st.write("Open facilities:", payload.get("open_facilities", []))
    st.write("Assignment preview:", payload.get("assignments_preview", []))
    if payload.get("error"):
        st.error(payload["error"])
    if payload.get("generated_code"):
        st.code(payload["generated_code"], language="python")
    if payload.get("notes"):
        for note in payload["notes"]:
            st.caption(note)


def main() -> None:
    st.title("Milestone 5 - Productionized UFLP Service Client")
    st.caption(
        "This client calls the Milestone 5 FastAPI service. The deterministic baseline is the production-safe "
        "default runtime, while the LLM-assisted path remains available as an optional served mode."
    )

    with st.sidebar:
        api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)
        st.markdown("**Recommended local flow**")
        st.code("python scripts/run_api.py")
        st.code("python scripts/run_client.py")

    try:
        runtime_info = _safe_get_json(f"{api_base_url}/api/v1/runtime")
        catalog = _safe_get_json(f"{api_base_url}/api/v1/catalog/instances")
    except Exception as exc:
        st.error(f"Could not reach the API service: {type(exc).__name__}: {exc}")
        st.stop()

    st.info(
        f"Safe default mode: {runtime_info['safe_default_mode']} | "
        f"Default LLM candidate: {runtime_info['default_llm_candidate_name']}"
    )
    for warning in runtime_info.get("warnings", []):
        st.warning(warning)

    candidate_names = [candidate["name"] for candidate in runtime_info["available_candidates"] if candidate["kind"] == "llm"]

    tab_catalog, tab_inline = st.tabs(["Catalog Instance", "Inline Instance"])

    with tab_catalog:
        selected_instance = st.selectbox(
            "Catalog instance",
            options=catalog,
            format_func=lambda item: f"{item['instance_id']} (m={item['facility_count_m']}, n={item['customer_count_n']})",
        )
        mode = st.selectbox("Serving mode", options=["auto", "baseline", "llm", "compare"], index=0)
        candidate_name = st.selectbox("LLM candidate", options=candidate_names, index=candidate_names.index(runtime_info["default_llm_candidate_name"]) if runtime_info["default_llm_candidate_name"] in candidate_names else 0)
        return_assignments = st.checkbox("Return full assignments", value=False)
        return_generated_code = st.checkbox("Return generated code", value=False)

        if st.button("Solve catalog instance", type="primary"):
            payload = {
                "mode": mode,
                "source": "catalog",
                "instance_id": selected_instance["instance_id"],
                "candidate_name": candidate_name,
                "return_assignments": return_assignments,
                "return_generated_code": return_generated_code,
            }
            try:
                response = _post_json(f"{api_base_url}/api/v1/solve", payload)
            except Exception as exc:
                st.error(f"API request failed: {type(exc).__name__}: {exc}")
            else:
                st.success(f"Request completed with overall status: {response['overall_status']}")
                for warning in response.get("warnings", []):
                    st.warning(warning)
                render_candidate_block("Baseline", response.get("baseline"))
                render_candidate_block("Candidate", response.get("candidate"))
                st.json(response)

    with tab_inline:
        mode_inline = st.selectbox("Inline serving mode", options=["baseline", "llm", "compare"], index=0)
        candidate_name_inline = st.selectbox("Inline LLM candidate", options=candidate_names, index=candidate_names.index(runtime_info["default_llm_candidate_name"]) if runtime_info["default_llm_candidate_name"] in candidate_names else 0, key="inline_candidate")
        instance_text = st.text_area("Paste raw OR-Library instance text", height=300)
        return_assignments_inline = st.checkbox("Return full assignments", value=False, key="inline_assignments")
        return_generated_code_inline = st.checkbox("Return generated code", value=False, key="inline_code")

        if st.button("Solve inline instance"):
            payload = {
                "mode": mode_inline,
                "source": "inline",
                "instance_text": instance_text,
                "candidate_name": candidate_name_inline,
                "return_assignments": return_assignments_inline,
                "return_generated_code": return_generated_code_inline,
            }
            try:
                response = _post_json(f"{api_base_url}/api/v1/solve", payload)
            except Exception as exc:
                st.error(f"API request failed: {type(exc).__name__}: {exc}")
            else:
                st.success(f"Request completed with overall status: {response['overall_status']}")
                for warning in response.get("warnings", []):
                    st.warning(warning)
                render_candidate_block("Baseline", response.get("baseline"))
                render_candidate_block("Candidate", response.get("candidate"))
                st.json(response)


if __name__ == "__main__":
    main()
