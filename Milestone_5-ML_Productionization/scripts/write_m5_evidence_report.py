from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


M5_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = M5_ROOT.parent
REPORTS_DIR = M5_ROOT / "reports"
EVIDENCE_JSON = REPORTS_DIR / "m5_evidence_report.json"
EVIDENCE_TXT = REPORTS_DIR / "m5_evidence_report.txt"


def _run(args: list[str], timeout_s: int = 240, cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(
        args,
        cwd=cwd or M5_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return {
        "command": " ".join(args),
        "returncode": result.returncode,
        "success": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    pytest_result = _run([sys.executable, "-m", "pytest", "tests", "-q"])
    smoke_result = _run([sys.executable, "scripts/smoke_test.py"])
    serving_pipeline_result = _run([sys.executable, "scripts/run_serving_pipeline.py"], timeout_s=360)
    hf_bundle_result = _run([sys.executable, "scripts/deploy_huggingface_space.py"])
    compose_config_result = _run(["docker", "compose", "-f", "deployment/docker-compose.yml", "config"], timeout_s=240)
    package_result = _run([sys.executable, "scripts/package_mlflow_runtime.py"], timeout_s=360)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requirements_coverage": {
            "architecture_drawing": (M5_ROOT / "assets" / "architecture.png").exists(),
            "serving_modes": ["on-demand-machine", "on-demand-human", "batch"],
            "model_service_api": (M5_ROOT / "src" / "m5_productionization" / "api" / "main.py").exists(),
            "front_end_client": (M5_ROOT / "frontend" / "app.py").exists(),
            "containerization": (M5_ROOT / "deployment" / "docker-compose.yml").exists(),
            "cicd": (REPO_ROOT / ".github" / "workflows" / "milestone5-ci-cd.yml").exists(),
            "hosting_assets": (M5_ROOT / "deployment" / "huggingface_space" / "app.py").exists(),
            "model_serving_runtime": (M5_ROOT / "src" / "m5_productionization" / "mlflow_runtime.py").exists(),
        },
        "commands": {
            "pytest": pytest_result,
            "smoke_test": smoke_result,
            "serving_pipeline": serving_pipeline_result,
            "huggingface_bundle": hf_bundle_result,
            "docker_compose_config": compose_config_result,
            "mlflow_packaging": package_result,
        },
        "serving_pipeline_status": _read_json(REPORTS_DIR / "serving_pipeline_status.json", {}),
        "batch_smoke_summary": _read_json(REPORTS_DIR / "batch_smoke_summary.json", {}),
        "live_finetuned_serving_smoke": _read_json(REPORTS_DIR / "live_finetuned_serving_smoke.json", {}),
        "manual_steps_required": [
            "To publish the Hugging Face Space, set HF_TOKEN and HF_SPACE_ID, then run M5_DEPLOY_HF=1 python scripts/deploy_huggingface_space.py.",
            "To make the deployed Space support live LLM/fine-tuned mode, add OPENAI_API_KEY as a Hugging Face Space secret.",
            "To activate the VM/DigitalOcean CD path, configure M5_DEPLOY_HOST, M5_DEPLOY_USER, and M5_DEPLOY_SSH_KEY GitHub repository secrets.",
        ],
    }


def _format(report: dict[str, Any]) -> str:
    lines = [
        "Milestone 5 Evidence Report",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "Requirement coverage:",
    ]
    for key, value in report["requirements_coverage"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Verification commands:")
    for name, result in report["commands"].items():
        lines.append(f"- {name}: success={result['success']} returncode={result['returncode']}")
    pipeline = report.get("serving_pipeline_status", {})
    if pipeline:
        lines.extend(["", "Serving pipeline:"])
        lines.append(f"- success: {pipeline.get('success')}")
        for step in pipeline.get("steps", []):
            lines.append(f"- {step.get('name')}: {step.get('status')} ({step.get('duration_s'):.2f}s)")
    live = report.get("live_finetuned_serving_smoke", {})
    if live:
        lines.extend(["", "Live fine-tuned serving smoke:"])
        lines.append(f"- overall_status: {live.get('overall_status')}")
        candidate = live.get("candidate") or {}
        lines.append(f"- candidate: {candidate.get('name')}")
        lines.append(f"- candidate_status: {candidate.get('status')}")
        lines.append(f"- model_name: {candidate.get('model_name')}")
        lines.append(f"- gap_vs_baseline_pct: {candidate.get('gap_vs_baseline_pct')}")
    lines.extend(["", "Manual steps required for public hosting:"])
    for step in report["manual_steps_required"]:
        lines.append(f"- {step}")
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    EVIDENCE_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    EVIDENCE_TXT.write_text(_format(report), encoding="utf-8")
    print(f"Wrote {EVIDENCE_TXT}")
    print(f"Wrote {EVIDENCE_JSON}")


if __name__ == "__main__":
    main()
