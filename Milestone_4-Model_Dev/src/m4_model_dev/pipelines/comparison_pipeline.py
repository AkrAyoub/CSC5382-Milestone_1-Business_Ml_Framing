from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.evaluation.metrics import binary_classification_report
from m4_model_dev.models.logistic_numpy import StandardScaler
from m4_model_dev.models.model_registry import SUPPORTED_MODEL_NAMES, fit_model_bundle, predict_scores
from m4_model_dev.paths import M4_CONFIGS_DIR, M4_REPORTS_DIR, ensure_runtime_dirs
from m4_model_dev.pipelines.training_pipeline import (
    _build_runtime_dataset,
    prepare_feature_columns,
    select_threshold,
)
from m4_model_dev.tracking.mlflow_utils import log_comparison_run
from m4_model_dev.utils.config import load_yaml_config
from m4_model_dev.utils.io import write_json, write_text


DEFAULT_COMPARISON_MODELS = [
    "dummy_prior",
    "logreg_numpy",
    "logreg_sklearn",
    "random_forest",
    "hist_gradient_boosting",
]


def order_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    # Keep model selection on validation metrics only.
    return results_df.sort_values(
        ["val_f1", "val_roc_auc", "val_precision", "model_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _flatten_metrics(
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


def _render_comparison_summary(results_df: pd.DataFrame) -> str:
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
    if not results_df.empty:
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


def run_model_comparison_pipeline(config_path: Path | None = None) -> dict[str, Any]:
    ensure_runtime_dirs()
    config_path = config_path or (M4_CONFIGS_DIR / "compare_models.yaml")
    config = load_yaml_config(config_path)
    random_seed = int(config.get("random_seed", 42))

    _, _, _, merged_df, feature_columns = _build_runtime_dataset(config)
    feature_columns = prepare_feature_columns(merged_df)
    train_df = merged_df[merged_df["split"] == "train"].copy()
    val_df = merged_df[merged_df["split"] == "val"].copy()
    test_df = merged_df[merged_df["split"] == "test"].copy()

    x_train_raw = train_df[feature_columns].to_numpy(dtype=float)
    x_val_raw = val_df[feature_columns].to_numpy(dtype=float)
    x_test_raw = test_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["is_open"].to_numpy(dtype=int)
    y_val = val_df["is_open"].to_numpy(dtype=int)
    y_test = test_df["is_open"].to_numpy(dtype=int)

    scaler = StandardScaler().fit(x_train_raw)
    x_train_scaled = scaler.transform(x_train_raw)
    x_val_scaled = scaler.transform(x_val_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    threshold_grid = [float(v) for v in config["comparison"]["threshold_grid"]]
    model_names = config.get("comparison", {}).get("model_names", DEFAULT_COMPARISON_MODELS)

    rows: list[dict[str, Any]] = []
    full_metrics: dict[str, dict[str, dict[str, float | int]]] = {}

    for model_name in model_names:
        if model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(f"Unsupported comparison model '{model_name}'")

        bundle = fit_model_bundle(
            model_name=model_name,
            config=config,
            x_train_raw=x_train_raw,
            x_train_scaled=x_train_scaled,
            y_train=y_train,
            scaler=scaler,
            random_seed=random_seed,
        )
        train_scores = predict_scores(bundle, x_train_raw, x_train_scaled)
        val_scores = predict_scores(bundle, x_val_raw, x_val_scaled)
        test_scores = predict_scores(bundle, x_test_raw, x_test_scaled)

        best_threshold, val_report = select_threshold(y_val.tolist(), val_scores, threshold_grid)
        metrics = {
            "train": binary_classification_report(y_train.tolist(), train_scores, best_threshold),
            "val": val_report,
            "test": binary_classification_report(y_test.tolist(), test_scores, best_threshold),
        }
        full_metrics[model_name] = metrics
        rows.append(_flatten_metrics(model_name, best_threshold, metrics, bundle.get("metadata", {})))

    results_df = order_results_df(pd.DataFrame(rows))
    best_row = results_df.iloc[0].to_dict()

    csv_path = M4_REPORTS_DIR / "model_comparison.csv"
    json_path = M4_REPORTS_DIR / "model_comparison.json"
    summary_path = M4_REPORTS_DIR / "model_comparison_summary.txt"
    selection_path = M4_REPORTS_DIR / "model_selection.json"

    results_df.to_csv(csv_path, index=False)
    write_json(
        {
            "config_path": str(config_path),
            "feature_columns": feature_columns,
            "results": results_df.to_dict(orient="records"),
            "full_metrics": full_metrics,
        },
        json_path,
    )
    write_text(_render_comparison_summary(results_df), summary_path)
    write_json(
        {
            "selection_rule": "highest validation F1, tie-broken by validation ROC-AUC, then validation precision, then model_name",
            "selected_model_family": best_row["model_name"],
            "selected_threshold": best_row["selected_threshold"],
            "selected_validation_f1": best_row["val_f1"],
            "selected_validation_roc_auc": best_row["val_roc_auc"],
            "selected_test_f1": best_row["test_f1"],
        },
        selection_path,
    )

    mlflow_logged = log_comparison_run(config, results_df, [csv_path, json_path, summary_path, selection_path])

    return {
        "results_df": results_df,
        "csv_path": csv_path,
        "json_path": json_path,
        "summary_path": summary_path,
        "selection_path": selection_path,
        "mlflow_logged": mlflow_logged,
    }
