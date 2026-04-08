from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.evaluation.metrics import binary_classification_report
from m4_model_dev.models.model_registry import SUPPORTED_MODEL_NAMES, fit_model_bundle, predict_scores
from m4_model_dev.paths import M4_CONFIGS_DIR, M4_REPORTS_DIR, ensure_runtime_dirs
from m4_model_dev.pipelines.contracts import ComparisonPipelineResult, TrainingInputs
from m4_model_dev.pipelines.training_pipeline import prepare_training_inputs, select_threshold
from m4_model_dev.reporting.comparison_reports import (
    flatten_comparison_metrics,
    order_results_df,
    write_comparison_reports,
)
from m4_model_dev.reporting.figures import write_comparison_figures
from m4_model_dev.tracking.mlflow_utils import log_comparison_run
from m4_model_dev.utils.config import load_yaml_config


DEFAULT_COMPARISON_MODELS = [
    "dummy_prior",
    "logreg_numpy",
    "logreg_sklearn",
    "random_forest",
    "hist_gradient_boosting",
]


def _evaluate_candidate(
    model_name: str,
    config: dict[str, Any],
    training_inputs: TrainingInputs,
    random_seed: int,
) -> dict[str, Any]:
    bundle = fit_model_bundle(
        model_name=model_name,
        config=config,
        x_train_raw=training_inputs["x_train_raw"],
        x_train_scaled=training_inputs["x_train_scaled"],
        y_train=training_inputs["y_train"],
        scaler=training_inputs["scaler"],
        random_seed=random_seed,
    )
    train_scores = predict_scores(bundle, training_inputs["x_train_raw"], training_inputs["x_train_scaled"])
    val_scores = predict_scores(bundle, training_inputs["x_val_raw"], training_inputs["x_val_scaled"])
    test_scores = predict_scores(bundle, training_inputs["x_test_raw"], training_inputs["x_test_scaled"])

    threshold_grid = [float(v) for v in config["comparison"]["threshold_grid"]]
    best_threshold, val_report = select_threshold(training_inputs["y_val"].tolist(), val_scores, threshold_grid)
    metrics = {
        "train": binary_classification_report(training_inputs["y_train"].tolist(), train_scores, best_threshold),
        "val": val_report,
        "test": binary_classification_report(training_inputs["y_test"].tolist(), test_scores, best_threshold),
    }
    return {
        "row": flatten_comparison_metrics(
            model_name,
            best_threshold,
            metrics,
            bundle.get("metadata", {}),
        ),
        "metrics": metrics,
    }


def run_model_comparison_pipeline(config_path: Path | None = None) -> ComparisonPipelineResult:
    ensure_runtime_dirs()
    config_path = config_path or (M4_CONFIGS_DIR / "compare_models.yaml")
    config = load_yaml_config(config_path)
    random_seed = int(config.get("random_seed", 42))
    training_inputs = prepare_training_inputs(config)
    feature_columns = training_inputs["feature_columns"]

    model_names = config.get("comparison", {}).get("model_names", DEFAULT_COMPARISON_MODELS)

    rows: list[dict[str, Any]] = []
    full_metrics: dict[str, dict[str, dict[str, float | int]]] = {}

    for model_name in model_names:
        if model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(f"Unsupported comparison model '{model_name}'")

        candidate_result = _evaluate_candidate(model_name, config, training_inputs, random_seed)
        full_metrics[model_name] = candidate_result["metrics"]
        rows.append(candidate_result["row"])

    results_df = order_results_df(pd.DataFrame(rows))

    csv_path = M4_REPORTS_DIR / "model_comparison.csv"
    json_path = M4_REPORTS_DIR / "model_comparison.json"
    summary_path = M4_REPORTS_DIR / "model_comparison_summary.txt"
    selection_path = M4_REPORTS_DIR / "model_selection.json"

    write_comparison_reports(
        results_df=results_df,
        full_metrics=full_metrics,
        feature_columns=feature_columns,
        config_path=config_path,
        csv_path=csv_path,
        json_path=json_path,
        summary_path=summary_path,
        selection_path=selection_path,
    )
    figure_paths = write_comparison_figures(results_df, M4_REPORTS_DIR)

    mlflow_logged = log_comparison_run(
        config,
        results_df,
        [csv_path, json_path, summary_path, selection_path, *figure_paths.values()],
    )

    return {
        "results_df": results_df,
        "csv_path": csv_path,
        "json_path": json_path,
        "summary_path": summary_path,
        "selection_path": selection_path,
        "figure_paths": figure_paths,
        "mlflow_logged": mlflow_logged,
    }
