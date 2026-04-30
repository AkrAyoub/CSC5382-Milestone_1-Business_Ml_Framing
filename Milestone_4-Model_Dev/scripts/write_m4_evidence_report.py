from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


M4_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = M4_ROOT / "reports"
ARTIFACTS_DIR = M4_ROOT / "artifacts"
EVIDENCE_JSON = REPORTS_DIR / "m4_evidence_report.json"
EVIDENCE_TXT = REPORTS_DIR / "m4_evidence_report.txt"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _run_command(args: list[str], timeout_s: int = 120) -> dict[str, Any]:
    result = subprocess.run(
        args,
        cwd=M4_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return {
        "command": " ".join(args),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "success": result.returncode == 0,
    }


def _latest_emissions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"available": False}
    latest = rows[-1]
    return {
        "available": True,
        "timestamp": latest.get("timestamp", ""),
        "emissions_kg": float(latest.get("emissions") or 0.0),
        "energy_consumed_kwh": float(latest.get("energy_consumed") or 0.0),
        "duration_s": float(latest.get("duration") or 0.0),
        "country_iso_code": latest.get("country_iso_code", ""),
    }


def _comparison_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    active_rows = [
        row
        for row in rows
        if row.get("split") == "val"
        and row.get("candidate_kind") != "deterministic_baseline"
        and int(float(row.get("attempted_instances") or 0)) > int(float(row.get("skipped_instances") or 0))
    ]
    fallback_rows = [row for row in rows if row.get("generation_backend_name") == "template_fallback"]
    candidate_rows = [
        {
            "candidate_name": row.get("candidate_name", ""),
            "split": row.get("split", ""),
            "generation_success_rate": float(row.get("generation_success_rate") or 0.0),
            "execution_success_rate": float(row.get("execution_success_rate") or 0.0),
            "exact_match_rate": float(row.get("exact_match_rate") or 0.0),
            "attempted_instances": int(float(row.get("attempted_instances") or 0)),
        }
        for row in rows
        if row.get("candidate_kind") != "deterministic_baseline"
    ]
    return {
        "available": True,
        "row_count": len(rows),
        "validation_candidate_count": len(active_rows),
        "uses_template_fallback": bool(fallback_rows),
        "template_fallback_rows": len(fallback_rows),
        "candidate_rows": candidate_rows,
    }


def _single_candidate_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "available": True,
        "row_count": len(rows),
        "candidate_name": rows[0].get("candidate_name", "") if rows else "",
        "generation_backend_name": rows[0].get("generation_backend_name", "") if rows else "",
        "successful_instances": sum(int(float(row.get("successful_instances") or 0)) for row in rows),
        "failed_instances": sum(int(float(row.get("failed_instances") or 0)) for row in rows),
        "skipped_instances": sum(int(float(row.get("skipped_instances") or 0)) for row in rows),
    }


def build_report() -> dict[str, Any]:
    tests = _run_command([sys.executable, "scripts/run_tests.py"])
    zenml_status = _read_json(REPORTS_DIR / "zenml_status.json", {})
    run_manifest = _read_json(REPORTS_DIR / "run_manifest.json", {})
    model_selection = _read_json(REPORTS_DIR / "model_selection.json", {})
    robust_spec = _read_json(ARTIFACTS_DIR / "llm_robust_prompt_v1_spec.json", {})
    fine_tuned_spec = _read_json(ARTIFACTS_DIR / "llm_fine_tuned_spec.json", {})
    openai_job = _read_json(REPORTS_DIR / "openai_finetune_job.json", {})
    openai_manifest = _read_json(M4_ROOT / "data" / "openai_finetune" / "openai_finetune_manifest.json", {})
    comparison = _comparison_summary(REPORTS_DIR / "model_comparison.csv")
    single_candidate = _single_candidate_summary(REPORTS_DIR / "evaluation" / "single_candidate_metrics.csv")
    emissions = _latest_emissions(REPORTS_DIR / "emissions.csv")
    openai_fine_tuned_model = openai_job.get("fine_tuned_model", "")
    openai_fine_tuned_succeeded = openai_job.get("status") == "succeeded" and bool(openai_fine_tuned_model)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tests": tests,
        "zenml": zenml_status,
        "mlflow": {
            "logged": bool(run_manifest.get("mlflow_logged")),
            "registered_model_name": run_manifest.get("registered_model_name", ""),
            "artifact_log_errors": run_manifest.get("mlflow_artifact_log_errors", []),
        },
        "model_versioning": {
            "registered_model_name": run_manifest.get("registered_model_name", ""),
            "candidate_spec_path": run_manifest.get("candidate_spec_path", ""),
            "robust_candidate_spec": robust_spec,
            "fine_tuned_candidate_spec": fine_tuned_spec,
            "fine_tuned_enabled": bool(fine_tuned_spec.get("enabled")) or openai_fine_tuned_succeeded,
        },
        "openai_finetuning": {
            "manifest_available": bool(openai_manifest),
            "train_records": openai_manifest.get("splits", {}).get("train", {}).get("record_count"),
            "validation_records": openai_manifest.get("splits", {}).get("val", {}).get("record_count"),
            "job_artifact_available": bool(openai_job),
            "job_id": openai_job.get("job", {}).get("id", ""),
            "status": openai_job.get("status", ""),
            "fine_tuned_model": openai_job.get("fine_tuned_model", ""),
        },
        "comparison": {
            "selection": model_selection,
            "summary": comparison,
        },
        "single_candidate_live_run": {
            "summary": single_candidate,
            "human_summary": _read_text(REPORTS_DIR / "summary.txt"),
        },
        "energy": emissions,
        "known_limitations": [
            "The OpenAI fine-tuning dataset is intentionally small because this milestone is a proof of MLOps training and evaluation integration, not final production model scaling.",
            "Template fallback rows, if any, are explicitly marked with generation_backend_name=template_fallback in comparison outputs.",
        ],
    }


def _format_report(report: dict[str, Any]) -> str:
    selection = report["comparison"]["selection"]
    comparison = report["comparison"]["summary"]
    live = report["single_candidate_live_run"]["summary"]
    energy = report["energy"]
    mlflow = report["mlflow"]
    model_versioning = report["model_versioning"]
    openai_ft = report["openai_finetuning"]
    lines = [
        "Milestone 4 Evidence Report",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "Verification:",
        f"- Tests passed: {report['tests']['success']}",
        f"- ZenML success: {bool(report['zenml'].get('success'))}",
        f"- MLflow logged: {mlflow['logged']}",
        f"- Registered model name: {mlflow['registered_model_name'] or 'not recorded'}",
        "",
        "Model versioning:",
        f"- Candidate spec path: {model_versioning['candidate_spec_path'] or 'not recorded'}",
        f"- Fine-tuned candidate enabled: {model_versioning['fine_tuned_enabled']}",
        "",
        "OpenAI fine-tuning track:",
        f"- Fine-tuning data manifest available: {openai_ft['manifest_available']}",
        f"- Train records: {openai_ft.get('train_records', 'not recorded')}",
        f"- Validation records: {openai_ft.get('validation_records', 'not recorded')}",
        f"- Job artifact available: {openai_ft['job_artifact_available']}",
        f"- Job status: {openai_ft.get('status') or 'not launched'}",
        f"- Fine-tuned model: {openai_ft.get('fine_tuned_model') or 'not available yet'}",
        "",
        "Model comparison:",
        f"- Selected candidate: {selection.get('selected_candidate_name', 'not recorded')}",
        f"- Selected validation success rate: {selection.get('selected_validation_success_rate', 'not recorded')}",
        f"- Uses template fallback in comparison outputs: {comparison.get('uses_template_fallback', False)}",
        f"- Template fallback rows: {comparison.get('template_fallback_rows', 0)}",
    ]
    for row in comparison.get("candidate_rows", []):
        lines.append(
            "- "
            f"{row['candidate_name']} {row['split']}: "
            f"generation={row['generation_success_rate']:.4f}, "
            f"execution={row['execution_success_rate']:.4f}, "
            f"exact_match={row['exact_match_rate']:.4f}, "
            f"attempted={row['attempted_instances']}"
        )
    lines.extend([
        "",
        "Live single-candidate run:",
        f"- Candidate: {live.get('candidate_name', 'not recorded')}",
        f"- Backend: {live.get('generation_backend_name', 'not recorded')}",
        f"- Successful instances: {live.get('successful_instances', 0)}",
        f"- Failed instances: {live.get('failed_instances', 0)}",
        f"- Skipped instances: {live.get('skipped_instances', 0)}",
        "",
        "Energy:",
        f"- Available: {energy.get('available', False)}",
        f"- Latest emissions kg: {energy.get('emissions_kg', 'not recorded')}",
        f"- Latest energy kWh: {energy.get('energy_consumed_kwh', 'not recorded')}",
        "",
        "Known limitations:",
    ])
    for limitation in report["known_limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "Test command output:",
            report["tests"]["stdout"] or report["tests"]["stderr"],
            "",
        ]
    )
    return "\n".join(lines)


def write_report() -> dict[str, Any]:
    report = build_report()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    EVIDENCE_TXT.write_text(_format_report(report), encoding="utf-8")
    return report


def main() -> None:
    report = write_report()
    print("Milestone 4 evidence report written:")
    print(f"- {EVIDENCE_TXT}")
    print(f"- {EVIDENCE_JSON}")
    print(json.dumps({
        "tests_passed": report["tests"]["success"],
        "zenml_success": bool(report["zenml"].get("success")),
        "mlflow_logged": report["mlflow"]["logged"],
        "registered_model_name": report["mlflow"]["registered_model_name"],
    }, indent=2))


if __name__ == "__main__":
    main()
