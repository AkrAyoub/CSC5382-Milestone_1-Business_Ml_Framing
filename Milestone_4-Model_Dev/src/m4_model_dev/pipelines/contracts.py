from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from m4_model_dev.models.logistic_numpy import StandardScaler


class TrainingInputs(TypedDict):
    labels_path: Path
    dataset_path: Path
    split_path: Path
    merged_dataset_path: Path
    feature_columns: list[str]
    split_counts: dict[str, int]
    x_train_raw: np.ndarray
    x_val_raw: np.ndarray
    x_test_raw: np.ndarray
    x_train_scaled: np.ndarray
    x_val_scaled: np.ndarray
    x_test_scaled: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


class TrainedModel(TypedDict):
    bundle: dict[str, Any]
    model_family: str
    emissions_path: Path | None


class EvaluationResults(TypedDict):
    threshold: float
    metrics: dict[str, dict[str, float | int]]
    confusion_report: dict[str, dict[str, float | int]]
    feature_importance_rows: list[dict[str, float | str]]


class ComparisonPipelineResult(TypedDict):
    results_df: Any
    csv_path: Path
    json_path: Path
    summary_path: Path
    selection_path: Path
    figure_paths: dict[str, Path]
    mlflow_logged: bool


class TrainingPipelineResult(TypedDict):
    config: dict[str, Any]
    config_path: Path
    dataset_path: Path
    split_path: Path
    metrics_path: Path
    summary_path: Path
    manifest_path: Path
    confusion_path: Path
    feature_importance_path: Path | None
    emissions_path: Path | None
    model_artifact_path: Path
    figure_paths: dict[str, Path]
    mlflow_logged: bool
    registered_model_name: str | None
    metrics: dict[str, dict[str, float | int]]
    model_family: str
    model_bundle: dict[str, Any]
    feature_columns: list[str]
    threshold: float
