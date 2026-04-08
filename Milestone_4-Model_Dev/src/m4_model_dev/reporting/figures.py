from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _save_figure(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _metric_frame(metrics: dict[str, dict[str, float | int]]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for split_name, split_metrics in metrics.items():
        row: dict[str, float | str] = {"split": split_name}
        for metric_name in DISPLAY_METRICS:
            row[metric_name] = float(split_metrics[metric_name])
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_confusion_axis(ax, split_name: str, split_metrics: dict[str, float | int]) -> None:
    matrix = np.array(
        [
            [int(split_metrics["tn"]), int(split_metrics["fp"])],
            [int(split_metrics["fn"]), int(split_metrics["tp"])],
        ]
    )
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{split_name.upper()} confusion")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")


def write_training_figures(
    metrics: dict[str, dict[str, float | int]],
    feature_importance_rows: list[dict[str, float | str]],
    model_family: str,
    threshold: float,
    output_dir: Path,
) -> dict[str, Path]:
    metric_df = _metric_frame(metrics)
    figure_paths: dict[str, Path] = {}

    metrics_path = output_dir / "training_metrics.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(metric_df["split"]))
    bar_width = 0.14
    for idx, metric_name in enumerate(DISPLAY_METRICS):
        values = metric_df[metric_name].to_numpy(dtype=float)
        ax.bar(positions + idx * bar_width, values, width=bar_width, label=metric_name.upper())
    ax.set_xticks(positions + bar_width * (len(DISPLAY_METRICS) - 1) / 2.0, metric_df["split"].str.upper().tolist())
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_family} split metrics")
    ax.legend(ncols=3, fontsize=8)
    figure_paths["training_metrics"] = _save_figure(fig, metrics_path)

    confusion_path = output_dir / "training_confusion_matrices.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, split_name in zip(axes, ["train", "val", "test"]):
        _plot_confusion_axis(axis, split_name, metrics[split_name])
    figure_paths["training_confusion_matrices"] = _save_figure(fig, confusion_path)

    if feature_importance_rows:
        feature_importance_path = output_dir / "feature_importance.png"
        top_rows = feature_importance_rows[:10]
        names = [str(row["feature_name"]) for row in reversed(top_rows)]
        values = [float(row["importance"]) for row in reversed(top_rows)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(names, values, color="#2f6b8a")
        ax.set_xlabel("Importance")
        ax.set_title("Top feature importances")
        figure_paths["feature_importance"] = _save_figure(fig, feature_importance_path)

    dashboard_path = output_dir / "training_dashboard.png"
    has_importance = bool(feature_importance_rows)
    fig, axes = plt.subplots(3 if has_importance else 2, 1, figsize=(11, 12 if has_importance else 9))
    axes = np.atleast_1d(axes)

    for idx, metric_name in enumerate(["f1", "roc_auc", "precision", "recall"]):
        values = metric_df[metric_name].to_numpy(dtype=float)
        axes[0].bar(positions + idx * 0.18, values, width=0.18, label=metric_name.upper())
    axes[0].set_xticks(positions + 0.27, metric_df["split"].str.upper().tolist())
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"Training dashboard: {model_family} (threshold={threshold:.2f})")
    axes[0].legend(ncols=4, fontsize=8)

    _plot_confusion_axis(axes[1], "val", metrics["val"])

    if has_importance:
        top_rows = feature_importance_rows[:8]
        names = [str(row["feature_name"]) for row in reversed(top_rows)]
        values = [float(row["importance"]) for row in reversed(top_rows)]
        axes[2].barh(names, values, color="#3c8c4a")
        axes[2].set_xlabel("Importance")
        axes[2].set_title("Top features")

    figure_paths["training_dashboard"] = _save_figure(fig, dashboard_path)
    return figure_paths


def write_comparison_figures(results_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    ordered = results_df.reset_index(drop=True)
    figure_paths: dict[str, Path] = {}
    model_names = ordered["model_name"].tolist()
    y_positions = np.arange(len(model_names))

    validation_path = output_dir / "comparison_validation_metrics.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y_positions - 0.18, ordered["val_f1"].to_numpy(dtype=float), height=0.35, label="Validation F1")
    ax.barh(y_positions + 0.18, ordered["val_roc_auc"].to_numpy(dtype=float), height=0.35, label="Validation ROC-AUC")
    ax.set_yticks(y_positions, model_names)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("Model comparison on validation metrics")
    ax.legend()
    figure_paths["comparison_validation_metrics"] = _save_figure(fig, validation_path)

    test_path = output_dir / "comparison_test_metrics.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y_positions - 0.18, ordered["test_f1"].to_numpy(dtype=float), height=0.35, label="Test F1")
    ax.barh(y_positions + 0.18, ordered["test_roc_auc"].to_numpy(dtype=float), height=0.35, label="Test ROC-AUC")
    ax.set_yticks(y_positions, model_names)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("Held-out test metrics by model")
    ax.legend()
    figure_paths["comparison_test_metrics"] = _save_figure(fig, test_path)

    dashboard_path = output_dir / "comparison_dashboard.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    x_positions = np.arange(len(model_names))
    axes[0].bar(x_positions - 0.18, ordered["val_f1"].to_numpy(dtype=float), width=0.35, label="Val F1")
    axes[0].bar(x_positions + 0.18, ordered["test_f1"].to_numpy(dtype=float), width=0.35, label="Test F1")
    axes[0].set_xticks(x_positions, model_names, rotation=20, ha="right")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("F1 comparison across model families")
    axes[0].legend()

    axes[1].bar(x_positions - 0.18, ordered["val_roc_auc"].to_numpy(dtype=float), width=0.35, label="Val ROC-AUC")
    axes[1].bar(x_positions + 0.18, ordered["test_roc_auc"].to_numpy(dtype=float), width=0.35, label="Test ROC-AUC")
    axes[1].set_xticks(x_positions, model_names, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Score")
    axes[1].set_title("ROC-AUC comparison across model families")
    axes[1].legend()

    figure_paths["comparison_dashboard"] = _save_figure(fig, dashboard_path)
    return figure_paths
