from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .common import read_json_file, write_json_file, write_text_file
except ImportError:
    from common import read_json_file, write_json_file, write_text_file

try:
    from ..paths import M3_ROOT, REPORTS_DIR, VALIDATION_DIR, ZENML_STATUS_JSON
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import M3_ROOT, REPORTS_DIR, VALIDATION_DIR, ZENML_STATUS_JSON


PIPELINE_RESULTS_JSON = REPORTS_DIR / "m3_pipeline_results.json"
PIPELINE_RESULTS_TXT = REPORTS_DIR / "m3_pipeline_results.txt"


def _run_command(args: list[str], *, cwd: Path = M3_ROOT, timeout_s: int = 120) -> dict[str, Any]:
    result = subprocess.run(
        args,
        cwd=cwd,
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


def build_pipeline_report() -> dict[str, Any]:
    validation_summary = read_json_file(VALIDATION_DIR / "validation_summary.json")
    anomalies = read_json_file(VALIDATION_DIR / "anomalies.json")
    zenml_status = read_json_file(ZENML_STATUS_JSON) if ZENML_STATUS_JSON.exists() else {
        "success": False,
        "details": "ZenML status file not found.",
    }

    dvc_status = _run_command([sys.executable, "-m", "dvc", "status"])
    dvc_dag = _run_command([sys.executable, "-m", "dvc", "dag"])
    feast_demo = _run_command([sys.executable, "feature_repo/run_feature_store_demo.py"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation": validation_summary,
        "anomalies": anomalies,
        "dvc": {
            "status": dvc_status,
            "dag": dvc_dag,
        },
        "feast": {
            "historical_retrieval_demo": feast_demo,
        },
        "zenml": zenml_status,
        "summary": {
            "raw_instances": validation_summary["raw_layer"]["instance_count"],
            "processed_instances": validation_summary["processed_layer"]["instances"],
            "processed_facilities": validation_summary["processed_layer"]["facilities"],
            "processed_customers": validation_summary["processed_layer"]["customers"],
            "assignment_costs": validation_summary["processed_layer"]["assignment_costs"],
            "instance_features": validation_summary["feature_layer"]["instance_features"],
            "facility_features": validation_summary["feature_layer"]["facility_features"],
            "customer_features": validation_summary["feature_layer"]["customer_features"],
            "symbolic_sft_train": validation_summary["training_layer"]["symbolic_sft_train"],
            "symbolic_sft_val": validation_summary["training_layer"]["symbolic_sft_val"],
            "symbolic_sft_test": validation_summary["training_layer"]["symbolic_sft_test"],
            "anomaly_count": validation_summary["anomaly_count"],
            "dvc_up_to_date": dvc_status["success"] and "up to date" in dvc_status["stdout"].lower(),
            "feast_demo_passed": feast_demo["success"],
            "zenml_success": bool(zenml_status.get("success")),
        },
    }


def _format_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Milestone 3 Pipeline Results",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "Data artifacts:",
        f"- Raw benchmark instances: {summary['raw_instances']}",
        f"- Processed instances: {summary['processed_instances']}",
        f"- Processed facilities: {summary['processed_facilities']}",
        f"- Processed customers: {summary['processed_customers']}",
        f"- Assignment-cost rows: {summary['assignment_costs']}",
        f"- Instance feature rows: {summary['instance_features']}",
        f"- Facility feature rows: {summary['facility_features']}",
        f"- Customer feature rows: {summary['customer_features']}",
        f"- Symbolic SFT splits: train={summary['symbolic_sft_train']}, val={summary['symbolic_sft_val']}, test={summary['symbolic_sft_test']}",
        "",
        "Validation:",
        f"- Status: {report['validation']['status']}",
        f"- Anomaly count: {summary['anomaly_count']}",
        "",
        "DVC:",
        f"- Status command passed: {report['dvc']['status']['success']}",
        f"- Pipeline up to date: {summary['dvc_up_to_date']}",
        f"- DAG command passed: {report['dvc']['dag']['success']}",
        "",
        "Feature store:",
        f"- Feast historical retrieval demo passed: {summary['feast_demo_passed']}",
        "",
        "ZenML:",
        f"- ZenML pipeline success: {summary['zenml_success']}",
        f"- Details: {report['zenml'].get('details', '')}",
        "",
        "DVC status output:",
        report["dvc"]["status"]["stdout"] or report["dvc"]["status"]["stderr"],
        "",
        "Feast demo output:",
        report["feast"]["historical_retrieval_demo"]["stdout"]
        or report["feast"]["historical_retrieval_demo"]["stderr"],
        "",
    ]
    return "\n".join(lines)


def write_pipeline_report() -> dict[str, Any]:
    report = build_pipeline_report()
    write_json_file(PIPELINE_RESULTS_JSON, report)
    write_text_file(PIPELINE_RESULTS_TXT, _format_text(report))
    return report


def main() -> None:
    report = write_pipeline_report()
    print("Milestone 3 pipeline report written:")
    print(f"- {PIPELINE_RESULTS_TXT}")
    print(f"- {PIPELINE_RESULTS_JSON}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
