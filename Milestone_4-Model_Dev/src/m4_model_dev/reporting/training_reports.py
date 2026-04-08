from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.utils.io import write_dataframe, write_json, write_text


def build_confusion_report(metrics: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    return {
        split_name: {
            key: split_metrics[key]
            for key in ("tp", "fp", "tn", "fn", "threshold", "support")
        }
        for split_name, split_metrics in metrics.items()
    }


def build_run_manifest(
    config_path: Path,
    labels_path: Path,
    dataset_path: Path,
    merged_dataset_path: Path,
    split_path: Path,
    model_family: str,
    model_artifact_path: Path,
    feature_columns: list[str],
    threshold: float,
    metrics: dict[str, dict[str, float | int]],
    metadata: dict[str, Any],
    uses_scaled_features: bool,
) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "labels_path": str(labels_path),
        "dataset_path": str(dataset_path),
        "merged_dataset_path": str(merged_dataset_path),
        "split_path": str(split_path),
        "model_family": model_family,
        "model_artifact_path": str(model_artifact_path),
        "feature_columns": feature_columns,
        "selected_threshold": threshold,
        "metrics": metrics,
        "metadata": metadata,
        "uses_scaled_features": uses_scaled_features,
        "mlflow_artifact_log_errors": [],
    }


def write_training_reports(
    metrics: dict[str, dict[str, float | int]],
    confusion_report: dict[str, dict[str, float | int]],
    feature_importance_rows: list[dict[str, float | str]],
    run_manifest: dict[str, Any],
    metrics_path: Path,
    confusion_path: Path,
    manifest_path: Path,
    feature_importance_path: Path,
) -> None:
    write_json(metrics, metrics_path)
    write_json(confusion_report, confusion_path)
    write_json(run_manifest, manifest_path)
    if feature_importance_rows:
        write_dataframe(pd.DataFrame(feature_importance_rows), feature_importance_path)


def render_split_metrics(split_name: str, split_metrics: dict[str, float | int]) -> str:
    return (
        f"{split_name}: "
        f"accuracy={split_metrics['accuracy']:.4f}, "
        f"precision={split_metrics['precision']:.4f}, "
        f"recall={split_metrics['recall']:.4f}, "
        f"f1={split_metrics['f1']:.4f}, "
        f"roc_auc={split_metrics['roc_auc']:.4f}, "
        f"log_loss={split_metrics['log_loss']:.4f}, "
        f"tp={split_metrics['tp']}, fp={split_metrics['fp']}, "
        f"tn={split_metrics['tn']}, fn={split_metrics['fn']}"
    )


def render_training_summary(
    config: dict[str, Any],
    feature_columns: list[str],
    threshold: float,
    metrics: dict[str, dict[str, float | int]],
    split_counts: dict[str, int],
    model_artifact_path: Path,
    emissions_path: Path | None,
    mlflow_logged: bool,
    registered_model_name: str | None,
    artifact_log_errors: list[str],
) -> str:
    model_family = str(config.get("model_family", "logreg_numpy"))
    lines = [
        "Milestone 4 training summary",
        f"run_name: {config.get('run_name', 'm4-run')}",
        f"model_family: {model_family}",
        f"feature_count: {len(feature_columns)}",
        f"selected_threshold: {threshold:.2f}",
        f"model_artifact_path: {model_artifact_path}",
        f"mlflow_logged: {mlflow_logged}",
        f"registered_model_name: {registered_model_name or 'none'}",
        f"codecarbon_emissions_logged: {bool(emissions_path and emissions_path.exists())}",
        f"mlflow_artifact_copy_warnings: {len(artifact_log_errors)}",
        "split_counts:",
    ]
    for split_name, count in split_counts.items():
        lines.append(f"  - {split_name}: {count}")
    lines.append("metrics:")
    for split_name, split_metrics in metrics.items():
        lines.append(f"  - {render_split_metrics(split_name, split_metrics)}")
    if artifact_log_errors:
        lines.append("mlflow_artifact_copy_errors:")
        for error in artifact_log_errors:
            lines.append(f"  - {error}")
    return "\n".join(lines) + "\n"


def write_summary(summary_text: str, summary_path: Path) -> None:
    write_text(summary_text, summary_path)
