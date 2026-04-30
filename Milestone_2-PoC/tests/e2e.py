from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import pytest


def _runtime_context() -> tuple[Path, Optional[Path], List[Path], float]:
    this_file = Path(__file__).resolve()
    milestone_root = this_file.parent.parent
    project_root = milestone_root.parent

    sys.path.insert(0, str(milestone_root))

    data_raw = project_root / "data" / "raw"
    if not data_raw.exists():
        raise RuntimeError(f"data/raw not found at expected location: {data_raw}")

    optfile = data_raw / "uncapopt.txt"
    optfile_path = optfile if optfile.exists() else None
    tol = float(os.getenv("E2E_TOL", "1e-6"))
    instances = _pick_instances(data_raw)
    return data_raw, optfile_path, instances, tol


def _pick_instances(data_raw: Path) -> List[Path]:
    preferred = ["cap71.txt", "cap74.txt", "cap131.txt"]
    picks: List[Path] = []
    for name in preferred:
        path = data_raw / name
        if path.exists():
            picks.append(path)

    if len(picks) >= 3:
        return picks[:3]

    for path in sorted(data_raw.glob("cap*.txt")):
        if path not in picks:
            picks.append(path)
        if len(picks) >= 3:
            break

    if len(picks) < 3:
        raise RuntimeError(f"Need at least 3 instances in {data_raw} (cap*.txt). Found {len(picks)}")
    return picks[:3]


def _assert_close(a: float, b: float, tol: float, label: str) -> None:
    denom = max(1.0, abs(b))
    rel = abs(a - b) / denom
    if rel > tol:
        raise AssertionError(f"{label}: not close. a={a} b={b} rel_err={rel} tol={tol}")


def _reports_dir() -> Path:
    reports = Path(__file__).resolve().parent.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports


def _write_e2e_report(report: dict[str, Any]) -> None:
    reports = _reports_dir()
    json_path = reports / "e2e_results.json"
    txt_path = reports / "e2e_results.txt"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: list[str] = [
        "Milestone 2 End-to-End Validation Evidence",
        "",
        f"Generated at: {report['generated_at']}",
        f"Tolerance: {report['tolerance']}",
        "",
        "Baseline validation:",
    ]
    for item in report["baseline"]["instances"]:
        lines.append(
            "  "
            f"{item['instance']}: objective={item['objective']:.3f}, "
            f"best_known={item['best_known'] if item['best_known'] is not None else 'N/A'}, "
            f"gap_percent={item['gap_percent'] if item['gap_percent'] is not None else 'N/A'}, "
            f"runtime_s={item['runtime_s']:.3f}, "
            f"open_facilities={item['open_facility_count']}, "
            f"assignments={item['assignment_count']}"
        )

    lines.extend(
        [
            f"Baseline status: {report['baseline']['status']}",
            "",
            "LLM verification:",
        ]
    )
    llm = report["llm_verification"]
    lines.append(f"  status={llm['status']}")
    lines.append(f"  enabled={llm['enabled']}")
    if llm.get("instance"):
        lines.append(f"  instance={llm['instance']}")
    if llm.get("backend_name"):
        lines.append(f"  backend={llm['backend_name']}")
    if llm.get("model_name"):
        lines.append(f"  model={llm['model_name']}")
    if llm.get("baseline_objective") is not None:
        lines.append(f"  baseline_objective={llm['baseline_objective']:.3f}")
    if llm.get("llm_objective") is not None:
        lines.append(f"  llm_objective={llm['llm_objective']:.3f}")
    if llm.get("gap_vs_baseline_pct") is not None:
        lines.append(f"  gap_vs_baseline_pct={llm['gap_vs_baseline_pct']:.6f}")
    if llm.get("open_facility_count") is not None:
        lines.append(f"  open_facilities={llm['open_facility_count']}")
    if llm.get("assignment_count") is not None:
        lines.append(f"  assignments={llm['assignment_count']}")
    if llm.get("trace"):
        lines.append("  trace:")
        for step in llm["trace"]:
            duration = step["duration_s"]
            duration_text = "N/A" if duration is None else f"{duration:.3f}s"
            lines.append(f"    - {step['name']}: {step['status']} ({duration_text})")
    if llm.get("error"):
        lines.append(f"  error={llm['error']}")

    lines.extend(
        [
            "",
            "Validation commands:",
            '  $env:E2E_ENABLE_LLM="0"; python tests\\e2e.py',
            "  python -m pytest tests\\e2e.py -q",
            '  $env:E2E_ENABLE_LLM="1"; $env:GROQ_API_KEY="YOUR_KEY"; python tests\\e2e.py',
            "",
        ]
    )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def _empty_llm_report(status: str, enabled: bool, error: str | None = None) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "status": status,
        "instance": None,
        "backend_name": None,
        "model_name": None,
        "baseline_objective": None,
        "llm_objective": None,
        "gap_vs_baseline_pct": None,
        "open_facility_count": None,
        "assignment_count": None,
        "trace": [],
        "error": error,
    }


def _build_report(
    baseline_instances: list[dict[str, Any]],
    tol: float,
    llm_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tolerance": tol,
        "baseline": {
            "status": "OK",
            "instances": baseline_instances,
        },
        "llm_verification": llm_report
        or _empty_llm_report(
            status="SKIPPED",
            enabled=False,
            error="Optional LLM verification skipped. Set E2E_ENABLE_LLM=1 to run it.",
        ),
    }


def _run_baseline_scenario(instances: List[Path], optfile: Optional[Path], tol: float) -> list[dict[str, Any]]:
    from src.baseline_solver import run_baseline

    print("\n[E2E] Baseline scenario on 3 instances")
    results: list[dict[str, Any]] = []
    for path in instances:
        res = run_baseline(str(path), str(optfile) if optfile else None)
        if res.best_known is not None and res.gap_percent is not None:
            if abs(res.gap_percent) > (tol * 100.0):
                raise AssertionError(
                    f"Baseline gap too large for {path.name}: gap%={res.gap_percent} tol%={tol*100.0}"
                )

        item = {
            "instance": path.name,
            "instance_path": str(path),
            "objective": res.objective,
            "best_known": res.best_known,
            "gap_percent": res.gap_percent,
            "runtime_s": res.runtime_s,
            "open_facility_count": len(res.open_facilities),
            "open_facilities": res.open_facilities,
            "assignment_count": len(res.assignments),
            "assignments_preview": res.assignments[:20],
        }
        results.append(item)

        print(
            f"  {path.name}: obj={res.objective:.3f} "
            f"best={res.best_known if res.best_known is not None else 'N/A'} "
            f"gap%={res.gap_percent if res.gap_percent is not None else 'N/A'} "
            f"rt={res.runtime_s:.3f}s "
            f"open_facilities={len(res.open_facilities)} "
            f"assignments={len(res.assignments)}"
        )

    print("[E2E] Baseline scenario - OK")
    return results


def _run_optional_llm_verification(instance_path: Path, optfile: Optional[Path], tol: float) -> dict[str, Any]:
    from src.pipeline_trace import PipelineTrace
    from src.poc_pipeline import run_poc_scenario

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Optional LLM verification requires GROQ_API_KEY. Set it (and optionally GROQ_MODEL) then rerun."
        )

    print("\n[E2E] Optional LLM verification scenario (1 instance)")
    trace = PipelineTrace()
    result = run_poc_scenario(
        str(instance_path),
        str(optfile) if optfile else None,
        enable_llm=True,
        trace=trace,
    )

    if result.llm.status != "OK":
        print("\n[E2E] Optional LLM verification failed. Trace steps:")
        for step in trace.steps:
            print(f" - {step.name}: {step.status} {f'err={step.error}' if step.error else ''}")
        raise AssertionError(f"Optional LLM verification failed: {result.llm.error}")

    if result.llm.objective is None:
        raise AssertionError("LLM verification returned OK but llm.objective is None")

    _assert_close(result.llm.objective, result.baseline.objective, tol, "LLM vs baseline objective")
    if result.llm.gap_vs_baseline_pct is not None and abs(result.llm.gap_vs_baseline_pct) > (tol * 100.0):
        raise AssertionError(
            f"LLM gap vs baseline too large: {result.llm.gap_vs_baseline_pct}% tol%={tol*100.0}"
        )

    print(
        f"  {instance_path.name}: baseline={result.baseline.objective:.3f} "
        f"llm={result.llm.objective:.3f} "
        f"gap%={result.llm.gap_vs_baseline_pct:.6f} "
        f"backend={result.llm.backend_name} "
        f"model={result.llm.model_name} "
        f"open_facilities={len(result.llm.open_facilities)} "
        f"assignments={len(result.llm.assignments)}"
    )
    print("[E2E] Optional LLM verification - OK")
    return {
        "enabled": True,
        "status": "OK",
        "instance": instance_path.name,
        "instance_path": str(instance_path),
        "backend_name": result.llm.backend_name,
        "model_name": result.llm.model_name,
        "baseline_objective": result.baseline.objective,
        "llm_objective": result.llm.objective,
        "gap_vs_baseline_pct": result.llm.gap_vs_baseline_pct,
        "open_facility_count": len(result.llm.open_facilities),
        "open_facilities": result.llm.open_facilities,
        "assignment_count": len(result.llm.assignments),
        "assignments_preview": result.llm.assignments[:20],
        "trace": [
            {
                "name": step.name,
                "status": step.status,
                "duration_s": step.duration_s,
                "error": step.error,
                "artifact_keys": sorted(step.artifacts.keys()),
            }
            for step in trace.steps
        ],
        "error": None,
    }


def test_baseline_scenario() -> None:
    _, optfile_path, instances, tol = _runtime_context()
    baseline_results = _run_baseline_scenario(instances, optfile_path, tol)
    _write_e2e_report(_build_report(baseline_results, tol))


def test_optional_llm_verification() -> None:
    data_raw, optfile_path, instances, tol = _runtime_context()

    enable_llm = os.getenv("E2E_ENABLE_LLM", "1").strip() == "1"
    if not enable_llm:
        pytest.skip("Optional LLM verification disabled. Set E2E_ENABLE_LLM=1 to run it.")

    if not os.getenv("GROQ_API_KEY", "").strip():
        pytest.skip("Optional LLM verification requires GROQ_API_KEY.")

    llm_instance_name = os.getenv("E2E_LLM_INSTANCE", instances[0].name)
    llm_instance = data_raw / llm_instance_name
    if not llm_instance.exists():
        raise RuntimeError(f"E2E_LLM_INSTANCE={llm_instance_name} not found in {data_raw}")
    baseline_results = _run_baseline_scenario(instances, optfile_path, tol)
    llm_report = _run_optional_llm_verification(llm_instance, optfile_path, tol)
    _write_e2e_report(_build_report(baseline_results, tol, llm_report=llm_report))


def main() -> None:
    data_raw, optfile_path, instances, tol = _runtime_context()

    baseline_results = _run_baseline_scenario(instances, optfile_path, tol)

    enable_llm = os.getenv("E2E_ENABLE_LLM", "1").strip() == "1"
    llm_report = None
    if enable_llm:
        llm_instance_name = os.getenv("E2E_LLM_INSTANCE", instances[0].name)
        llm_instance = data_raw / llm_instance_name
        if not llm_instance.exists():
            raise RuntimeError(f"E2E_LLM_INSTANCE={llm_instance_name} not found in {data_raw}")
        llm_report = _run_optional_llm_verification(llm_instance, optfile_path, tol)
    else:
        print("\n[E2E] Optional LLM verification skipped. Set E2E_ENABLE_LLM=1 to run it.")

    _write_e2e_report(_build_report(baseline_results, tol, llm_report=llm_report))
    print(f"\n[E2E] Results written to {_reports_dir() / 'e2e_results.txt'}")
    print(f"[E2E] JSON results written to {_reports_dir() / 'e2e_results.json'}")
    print("\n ALL E2E TESTS PASSED")


if __name__ == "__main__":
    main()
