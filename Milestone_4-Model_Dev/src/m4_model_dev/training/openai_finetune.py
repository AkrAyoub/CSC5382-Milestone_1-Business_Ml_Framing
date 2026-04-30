from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m4_model_dev.data.build_benchmark_dataset import build_benchmark_dataset
from m4_model_dev.data.build_reference_solutions import build_reference_solutions
from m4_model_dev.data.build_sft_dataset import build_sft_dataset
from m4_model_dev.data.make_splits import build_grouped_splits
from m4_model_dev.paths import (
    M4_DATASETS_DIR,
    M4_OPENAI_FT_DIR,
    M4_REFERENCE_DIR,
    M4_REPORTS_DIR,
    M4_SFT_DIR,
    M4_SPLITS_DIR,
    ensure_runtime_dirs,
)


OPENAI_SYSTEM_MESSAGE = (
    "You are an expert operations research engineer. "
    "Return only safe Python code for an OR-Tools CBC UFLP solver. "
    "Do not include commentary, markdown, prints, filesystem writes, or unsafe imports."
)
DEFAULT_BASE_MODEL = "gpt-4.1-mini-2025-04-14"
OPENAI_FINE_TUNE_JOB_PATH = M4_REPORTS_DIR / "openai_finetune_job.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _prepare_sft_assets() -> dict[str, Path]:
    existing = {
        "manifest": M4_SFT_DIR / "sft_manifest.json",
        "train_jsonl": M4_SFT_DIR / "sft_train.jsonl",
        "val_jsonl": M4_SFT_DIR / "sft_val.jsonl",
        "test_jsonl": M4_SFT_DIR / "sft_test.jsonl",
    }
    if all(path.exists() for path in existing.values()):
        return existing

    reference_path = build_reference_solutions(M4_REFERENCE_DIR / "reference_solutions.csv")
    dataset_path = build_benchmark_dataset(reference_path=reference_path, output_path=M4_DATASETS_DIR / "benchmark_instances.csv")
    split_path = build_grouped_splits(dataset_path=dataset_path, output_path=M4_SPLITS_DIR / "instance_splits.csv")
    sft_paths = build_sft_dataset(
        split_path=split_path,
        dataset_path=dataset_path,
        reference_path=reference_path,
        output_dir=M4_SFT_DIR,
    )
    return {
        "reference_path": reference_path,
        "dataset_path": dataset_path,
        "split_path": split_path,
        **sft_paths,
    }


def _to_openai_chat_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if isinstance(messages, list) and len(messages) >= 3:
        normalized = []
        for message in messages:
            normalized.append({"role": str(message["role"]), "content": str(message["content"])})
        if normalized[0]["role"] == "system":
            normalized[0]["content"] = OPENAI_SYSTEM_MESSAGE
        return {"messages": normalized}

    prompt = str(record.get("prompt") or "")
    response = str(record.get("response") or "")
    if not prompt or not response:
        raise ValueError("SFT record must contain either messages or prompt/response fields.")

    return {
        "messages": [
            {"role": "system", "content": OPENAI_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def export_openai_finetune_files() -> dict[str, Any]:
    ensure_runtime_dirs()
    sft_assets = _prepare_sft_assets()
    train_source_records = _read_jsonl(sft_assets["train_jsonl"])
    val_source_records = _read_jsonl(sft_assets["val_jsonl"])
    test_source_records = _read_jsonl(sft_assets["test_jsonl"])

    # OpenAI fine-tuning benefits from at least a modest number of training
    # examples. The project split is tiny, so we use train+val for fine-tuning
    # and keep the original test split as the validation file for the hosted job.
    openai_train_records = [_to_openai_chat_record(record) for record in [*train_source_records, *val_source_records]]
    openai_val_records = [_to_openai_chat_record(record) for record in test_source_records]
    openai_test_records = [_to_openai_chat_record(record) for record in test_source_records]

    train_path = _write_jsonl(openai_train_records, M4_OPENAI_FT_DIR / "openai_train.jsonl")
    val_path = _write_jsonl(openai_val_records, M4_OPENAI_FT_DIR / "openai_val.jsonl")
    test_path = _write_jsonl(openai_test_records, M4_OPENAI_FT_DIR / "openai_test.jsonl")

    outputs: dict[str, Any] = {
        "train": {
            "source_path": f"{sft_assets['train_jsonl']} + {sft_assets['val_jsonl']}",
            "openai_path": str(train_path),
            "record_count": len(openai_train_records),
            "source_splits": ["train", "val"],
        },
        "val": {
            "source_path": str(sft_assets["test_jsonl"]),
            "openai_path": str(val_path),
            "record_count": len(openai_val_records),
            "source_splits": ["test"],
        },
        "test": {
            "source_path": str(sft_assets["test_jsonl"]),
            "openai_path": str(test_path),
            "record_count": len(openai_test_records),
            "source_splits": ["test"],
        },
    }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "format": "OpenAI chat fine-tuning JSONL",
        "base_model": DEFAULT_BASE_MODEL,
        "system_message": OPENAI_SYSTEM_MESSAGE,
        "splits": outputs,
    }
    manifest_path = _write_json(manifest, M4_OPENAI_FT_DIR / "openai_finetune_manifest.json")
    return {"manifest_path": manifest_path, **manifest}


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is not installed. Install Milestone 4 requirements first.") from exc

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        _hydrate_openai_key_from_windows_environment()

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in the shell before launching fine-tuning.")
    return OpenAI()


def _hydrate_openai_key_from_windows_environment() -> None:
    if os.name != "nt" or (os.getenv("OPENAI_API_KEY") or "").strip():
        return
    command = (
        "$v=[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','User'); "
        "if([string]::IsNullOrWhiteSpace($v)){$v=[Environment]::GetEnvironmentVariable('OPENAI_API_KEY','Machine')}; "
        "if($v){[Console]::Out.Write($v)}"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return
    value = result.stdout.strip()
    if value:
        os.environ["OPENAI_API_KEY"] = value


def _job_to_dict(job: Any) -> dict[str, Any]:
    if hasattr(job, "model_dump"):
        return job.model_dump()
    if hasattr(job, "to_dict"):
        return job.to_dict()
    return json.loads(job.model_dump_json())


def create_openai_finetune_job(
    *,
    model: str = DEFAULT_BASE_MODEL,
    suffix: str = "uflp-symbolic-solver",
) -> dict[str, Any]:
    exported = export_openai_finetune_files()
    client = _client()
    train_path = Path(exported["splits"]["train"]["openai_path"])
    val_path = Path(exported["splits"]["val"]["openai_path"])

    with train_path.open("rb") as train_handle:
        train_file = client.files.create(file=train_handle, purpose="fine-tune")
    with val_path.open("rb") as val_handle:
        val_file = client.files.create(file=val_handle, purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        model=model,
        training_file=train_file.id,
        validation_file=val_file.id,
        suffix=suffix,
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": model,
        "training_file_id": train_file.id,
        "validation_file_id": val_file.id,
        "job": _job_to_dict(job),
        "fine_tuned_model": getattr(job, "fine_tuned_model", None),
        "status": getattr(job, "status", ""),
    }
    _write_json(payload, OPENAI_FINE_TUNE_JOB_PATH)
    return payload


def refresh_openai_finetune_status(job_id: str | None = None) -> dict[str, Any]:
    client = _client()
    previous = {}
    if OPENAI_FINE_TUNE_JOB_PATH.exists():
        previous = json.loads(OPENAI_FINE_TUNE_JOB_PATH.read_text(encoding="utf-8"))
    active_job_id = job_id or previous.get("job", {}).get("id")
    if not active_job_id:
        raise RuntimeError("No fine-tuning job id was provided and no saved job artifact exists.")

    job = client.fine_tuning.jobs.retrieve(active_job_id)
    job_payload = _job_to_dict(job)
    payload = {
        **previous,
        "last_checked_at": datetime.now(timezone.utc).isoformat(),
        "job": job_payload,
        "fine_tuned_model": job_payload.get("fine_tuned_model"),
        "status": job_payload.get("status", ""),
    }
    _write_json(payload, OPENAI_FINE_TUNE_JOB_PATH)
    return payload


def wait_for_openai_finetune(job_id: str | None = None, poll_seconds: int = 60, max_wait_minutes: int = 120) -> dict[str, Any]:
    deadline = time.time() + max_wait_minutes * 60
    terminal = {"succeeded", "failed", "cancelled"}
    payload = refresh_openai_finetune_status(job_id)
    while str(payload.get("status", "")).lower() not in terminal:
        if time.time() >= deadline:
            return payload
        time.sleep(poll_seconds)
        payload = refresh_openai_finetune_status(job_id)
    return payload
