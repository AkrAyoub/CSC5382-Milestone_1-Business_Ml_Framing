from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from m4_model_dev.models.logistic_numpy import (
    LogisticTrainingConfig,
    NumpyLogisticRegression,
    StandardScaler,
)


SKLEARN_MODEL_NAMES = {
    "dummy_prior",
    "logreg_sklearn",
    "random_forest",
    "hist_gradient_boosting",
}

SUPPORTED_MODEL_NAMES = SKLEARN_MODEL_NAMES | {"logreg_numpy"}


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    y_int = y.astype(int)
    counts = np.bincount(y_int, minlength=2)
    total = float(len(y))
    weights = np.ones(len(y), dtype=float)
    for cls in (0, 1):
        if counts[cls] > 0:
            weights[y_int == cls] = total / (2.0 * counts[cls])
    return weights


def build_model_training_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    training_config = dict(config.get("training", {}))
    model_specific_config = config.get("comparison", {}).get(model_name, {})
    if isinstance(model_specific_config, dict):
        training_config.update(model_specific_config)
    return training_config


def _build_bundle(
    model_name: str,
    model: Any,
    uses_scaled_features: bool,
    scaler: StandardScaler | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model": model,
        "uses_scaled_features": uses_scaled_features,
        "scaler": scaler,
        "metadata": metadata,
    }


def make_numpy_training_config(training_config: dict[str, Any]) -> LogisticTrainingConfig:
    return LogisticTrainingConfig(
        learning_rate=float(training_config.get("learning_rate", 0.05)),
        epochs=int(training_config.get("epochs", 3000)),
        l2_strength=float(training_config.get("l2_strength", 0.001)),
        beta1=float(training_config.get("beta1", 0.9)),
        beta2=float(training_config.get("beta2", 0.999)),
        epsilon=float(training_config.get("epsilon", 1e-8)),
    )


def fit_model_bundle(
    model_name: str,
    config: dict[str, Any],
    x_train_raw: np.ndarray,
    x_train_scaled: np.ndarray,
    y_train: np.ndarray,
    scaler: StandardScaler,
    random_seed: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"model_name": model_name}
    training_config = build_model_training_config(config, model_name)

    if model_name == "dummy_prior":
        model = DummyClassifier(strategy="prior")
        model.fit(x_train_raw, y_train)
        return _build_bundle(model_name, model, False, None, metadata)

    if model_name == "logreg_sklearn":
        model = LogisticRegression(
            max_iter=int(training_config.get("max_iter", 5000)),
            class_weight=str(training_config.get("class_weight", "balanced")),
            random_state=random_seed,
        )
        model.fit(x_train_scaled, y_train)
        return _build_bundle(model_name, model, True, scaler, metadata)

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(training_config.get("n_estimators", 400)),
            max_depth=(
                int(training_config["max_depth"])
                if training_config.get("max_depth") is not None
                else None
            ),
            min_samples_leaf=int(training_config.get("min_samples_leaf", 3)),
            class_weight=str(training_config.get("class_weight", "balanced")),
            random_state=random_seed,
        )
        model.fit(x_train_raw, y_train)
        metadata["feature_importances_available"] = True
        return _build_bundle(model_name, model, False, None, metadata)

    if model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=float(training_config.get("learning_rate", 0.05)),
            max_depth=(
                int(training_config["max_depth"])
                if training_config.get("max_depth") is not None
                else None
            ),
            max_iter=int(training_config.get("max_iter", 300)),
            # Disable internal validation splits unless explicitly requested.
            early_stopping=bool(training_config.get("early_stopping", False)),
            random_state=random_seed,
        )
        model.fit(x_train_raw, y_train, sample_weight=balanced_sample_weight(y_train))
        metadata["early_stopping"] = bool(training_config.get("early_stopping", False))
        return _build_bundle(model_name, model, False, None, metadata)

    if model_name == "logreg_numpy":
        training_cfg = make_numpy_training_config(training_config)
        model = NumpyLogisticRegression(training_cfg).fit(x_train_scaled, y_train.astype(float))
        metadata["loss_final"] = model.loss_history_[-1] if model.loss_history_ else None
        metadata["training_config"] = asdict(training_cfg)
        return _build_bundle(model_name, model, True, scaler, metadata)

    raise ValueError(f"Unsupported model family: {model_name}")


def predict_scores(bundle: dict[str, Any], x_raw: np.ndarray, x_scaled: np.ndarray) -> list[float]:
    model = bundle["model"]
    if bundle["model_name"] == "logreg_numpy":
        return model.predict_proba(x_scaled).tolist()
    features = x_scaled if bundle.get("uses_scaled_features", False) else x_raw
    return model.predict_proba(features)[:, 1].tolist()


def save_model_bundle(
    bundle: dict[str, Any],
    feature_columns: list[str],
    threshold: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = bundle["model_name"]

    if model_name == "logreg_numpy":
        model = bundle["model"]
        scaler = bundle["scaler"]
        output_path = output_dir / "logreg_numpy_model.npz"
        np.savez(
            output_path,
            weights=model.weights_,
            bias=np.array([model.bias_], dtype=float),
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            feature_columns=np.array(feature_columns, dtype=object),
            threshold=np.array([threshold], dtype=float),
            loss_history=np.array(model.loss_history_, dtype=float),
        )
        return output_path

    output_path = output_dir / f"{model_name}_model.joblib"
    payload = {
        "model_name": model_name,
        "model": bundle["model"],
        "feature_columns": feature_columns,
        "threshold": threshold,
        "uses_scaled_features": bundle.get("uses_scaled_features", False),
        "scaler": bundle.get("scaler"),
        "metadata": bundle.get("metadata", {}),
    }
    joblib.dump(payload, output_path)
    return output_path


def extract_feature_importance(bundle: dict[str, Any], feature_columns: list[str]) -> list[dict[str, float | str]]:
    model = bundle["model"]
    if not hasattr(model, "feature_importances_"):
        return []
    importances = getattr(model, "feature_importances_")
    rows: list[dict[str, float | str]] = []
    for feature_name, importance in zip(feature_columns, importances):
        rows.append(
            {
                "feature_name": feature_name,
                "importance": float(importance),
            }
        )
    rows.sort(key=lambda item: float(item["importance"]), reverse=True)
    return rows
