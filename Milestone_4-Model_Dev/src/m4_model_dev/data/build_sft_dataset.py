from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.data.benchmark import build_reference_solver_module_source
from m4_model_dev.paths import M4_SFT_DIR


ROBUST_PROMPT_TEMPLATE = """
Write ONE self-contained Python module that solves the OR-Library UNCAPACITATED Facility Location Problem (UFLP) with OR-Tools CBC.

Requirements:
- define solve(instance_path: str) -> dict
- parse both OR-Library variants using line-based parsing and numeric-token extraction
- use solver = pywraplp.Solver.CreateSolver("CBC")
- return keys: objective, open_facilities, assignments
- no prints, no filesystem writes, no unsafe imports
""".strip()


def _training_text(prompt: str, response: str) -> str:
    return (
        "### Instruction\n"
        f"{prompt}\n\n"
        "### Response\n"
        f"{response}\n"
    )


def _messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are an expert operations research engineer."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def _build_record(
    row: dict[str, Any],
    *,
    prompt: str,
    response: str,
) -> dict[str, Any]:
    best_known = row.get("best_known")
    if pd.isna(best_known):
        best_known = None
    reference_objective = row.get("reference_objective")
    if pd.isna(reference_objective):
        reference_objective = None
    reference_gap_percent = row.get("reference_gap_percent")
    if pd.isna(reference_gap_percent):
        reference_gap_percent = None

    return {
        "instance_id": str(row["instance_id"]),
        "instance_path": str(row["instance_path"]),
        "split": str(row["split"]),
        "facility_count_m": int(row["facility_count_m"]),
        "customer_count_n": int(row["customer_count_n"]),
        "best_known_objective": best_known,
        "reference_objective": reference_objective,
        "reference_gap_percent": reference_gap_percent,
        "prompt_template": "robust_v1",
        "source": "reference_solver_teacher",
        "validation_status": "passed",
        "execution_status": "passed",
        "exact_match_with_baseline": 1,
        "repair_notes": "",
        "prompt": prompt,
        "response": response,
        "text": _training_text(prompt, response),
        "messages": _messages(prompt, response),
    }


def build_sft_dataset(
    split_path: Path,
    *,
    dataset_path: Path | None = None,
    reference_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    output_dir = output_dir or M4_SFT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(split_path)
    merged_df = split_df.copy()

    if dataset_path is not None and dataset_path.exists():
        dataset_df = pd.read_csv(dataset_path)
        merged_df = merged_df.merge(
            dataset_df[
                [
                    "instance_id",
                    "instance_path",
                    "facility_count_m",
                    "customer_count_n",
                    "best_known",
                ]
            ],
            on=["instance_id", "instance_path", "facility_count_m", "customer_count_n"],
            how="left",
        )

    if reference_path is not None and reference_path.exists():
        reference_df = pd.read_csv(reference_path).rename(
            columns={
                "objective": "reference_objective",
                "gap_percent": "reference_gap_percent",
            }
        )
        merged_df = merged_df.merge(
            reference_df[["instance_id", "reference_objective", "reference_gap_percent"]],
            on="instance_id",
            how="left",
        )

    teacher_response = build_reference_solver_module_source()

    outputs: dict[str, Path] = {}
    manifest_payload: dict[str, dict[str, Any]] = {}

    for split_name, group_df in merged_df.groupby("split", sort=True):
        records = [
            _build_record(row.to_dict(), prompt=ROBUST_PROMPT_TEMPLATE, response=teacher_response)
            for _, row in group_df.sort_values("instance_id").iterrows()
        ]

        jsonl_path = output_dir / f"sft_{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for payload in records:
                handle.write(json.dumps(payload) + "\n")

        csv_path = output_dir / f"sft_{split_name}.csv"
        pd.DataFrame(records).drop(columns=["messages"]).to_csv(csv_path, index=False)

        stats_path = output_dir / f"sft_{split_name}_stats.json"
        stats_payload = {
            "split": split_name,
            "record_count": len(records),
            "instance_ids": [record["instance_id"] for record in records],
            "facility_count_min": min(record["facility_count_m"] for record in records) if records else 0,
            "facility_count_max": max(record["facility_count_m"] for record in records) if records else 0,
            "customer_count_min": min(record["customer_count_n"] for record in records) if records else 0,
            "customer_count_max": max(record["customer_count_n"] for record in records) if records else 0,
            "schema_fields": list(records[0].keys()) if records else [],
        }
        stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

        outputs[f"{split_name}_jsonl"] = jsonl_path
        outputs[f"{split_name}_csv"] = csv_path
        outputs[f"{split_name}_stats"] = stats_path
        manifest_payload[split_name] = {
            "jsonl_path": str(jsonl_path),
            "csv_path": str(csv_path),
            "stats_path": str(stats_path),
            "record_count": len(records),
        }

    manifest_path = output_dir / "sft_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    outputs["manifest"] = manifest_path
    return outputs
