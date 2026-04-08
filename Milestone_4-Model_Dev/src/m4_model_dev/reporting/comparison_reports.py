from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.utils.io import write_dataframe, write_json, write_text


def flatten_comparison_metrics(
    model_name: str,
    threshold: float,
    metrics: dict[str, dict[str, float | int]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"model_name": model_name, "selected_threshold": threshold}
    row.update(metadata)
    for split_name, split_metrics in metrics.items():
        for metric_name, metric_value in split_metrics.items():
            row[f"{split_name}_{metric_name}"] = metric_value
    return row


def order_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    # Model selection stays on validation metrics only to keep the test split
    # reserved for final offline evaluation.
    return results_df.sort_values(
        ["val_f1", "val_roc_auc", "val_precision", "model_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def render_comparison_summary(results_df: pd.DataFrame) -> str:
    ordered = order_results_df(results_df)
    lines = [
        "Milestone 4 model comparison summary",
        "",
        "Ranking by validation F1, then validation ROC-AUC:",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"- {row['model_name']}: "
            f"val_f1={row['val_f1']:.4f}, "
            f"test_f1={row['test_f1']:.4f}, "
            f"val_roc_auc={row['val_roc_auc']:.4f}, "
            f"test_roc_auc={row['test_roc_auc']:.4f}, "
            f"threshold={row['selected_threshold']:.2f}"
        )
    if not ordered.empty:
        best_row = ordered.iloc[0]
        lines.extend(
            [
                "",
                f"Selected model by validation only: {best_row['model_name']}",
                f"Best validation F1: {best_row['val_f1']:.4f}",
                f"Validation ROC-AUC for selected model: {best_row['val_roc_auc']:.4f}",
                f"Observed held-out test F1: {best_row['test_f1']:.4f}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_selection_payload(best_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_rule": "highest validation F1, tie-broken by validation ROC-AUC, then validation precision, then model_name",
        "selected_model_family": best_row["model_name"],
        "selected_threshold": best_row["selected_threshold"],
        "selected_validation_f1": best_row["val_f1"],
        "selected_validation_roc_auc": best_row["val_roc_auc"],
        "selected_test_f1": best_row["test_f1"],
    }


def write_comparison_reports(
    results_df: pd.DataFrame,
    full_metrics: dict[str, dict[str, dict[str, float | int]]],
    feature_columns: list[str],
    config_path: Path,
    csv_path: Path,
    json_path: Path,
    summary_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    ordered = order_results_df(results_df)
    best_row = ordered.iloc[0].to_dict()

    write_dataframe(ordered, csv_path)
    write_json(
        {
            "config_path": str(config_path),
            "feature_columns": feature_columns,
            "results": ordered.to_dict(orient="records"),
            "full_metrics": full_metrics,
        },
        json_path,
    )
    write_text(render_comparison_summary(ordered), summary_path)
    selection_payload = build_selection_payload(best_row)
    write_json(selection_payload, selection_path)
    return selection_payload
