from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from m4_model_dev.models.model_registry import SKLEARN_MODEL_NAMES
from m4_model_dev.paths import M4_MLFLOW_DIR, M4_ROOT


MLFLOW_DB_PATH = M4_ROOT / "mlflow.db"


def get_tracking_uri() -> str:
    return f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"


def get_registry_uri() -> str:
    return get_tracking_uri()


def _default_experiment_artifact_uri(experiment_name: str) -> str:
    sanitized = experiment_name.replace(" ", "-").lower()
    artifact_root = M4_MLFLOW_DIR / sanitized
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root.resolve().as_uri()


def configure_mlflow():
    import mlflow
    from mlflow.tracking import MlflowClient

    M4_MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_registry_uri(get_registry_uri())
    return mlflow, MlflowClient()


def get_or_create_experiment(client, experiment_name: str) -> str:
    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(
        name=experiment_name,
        artifact_location=_default_experiment_artifact_uri(experiment_name),
    )


def _artifact_uri_to_path(artifact_uri: str) -> Path | None:
    if not artifact_uri.startswith("file:"):
        return None
    parsed = urlparse(artifact_uri)
    return Path(unquote(parsed.path.lstrip("/")))


def _ensure_run_artifact_dir(mlflow) -> None:
    active_run = mlflow.active_run()
    if active_run is None:
        return
    artifact_path = _artifact_uri_to_path(active_run.info.artifact_uri)
    if artifact_path is None:
        return
    # Local SQLite-backed runs do not always materialize the directory eagerly.
    artifact_path.mkdir(parents=True, exist_ok=True)
    (artifact_path / "supporting_artifacts").mkdir(parents=True, exist_ok=True)


def _tracking_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("tracking", {}).get("enable_mlflow", False))


def _log_training_params(
    mlflow,
    config: dict[str, Any],
    config_path: Path,
    model_name: str,
    feature_columns: list[str],
    threshold: float,
) -> None:
    mlflow.log_param("config_path", str(config_path))
    mlflow.log_param("model_family", model_name)
    mlflow.log_param("feature_set", config.get("feature_set", "facility_plus_instance"))
    mlflow.log_param("feature_count", len(feature_columns))
    mlflow.log_param("selected_threshold", threshold)
    for name, value in config.get("training", {}).items():
        if isinstance(value, list):
            continue
        mlflow.log_param(name, value)


def _log_split_metrics(mlflow, metrics: dict[str, dict[str, float | int]]) -> None:
    for split_name, split_metrics in metrics.items():
        for metric_name, metric_value in split_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"{split_name}_{metric_name}", float(metric_value))


def _log_supporting_artifacts(mlflow, artifacts: list[Path]) -> list[str]:
    artifact_log_errors: list[str] = []
    for artifact in artifacts:
        if not artifact.exists():
            continue
        try:
            mlflow.log_artifact(str(artifact), artifact_path="supporting_artifacts")
        except Exception as exc:
            try:
                # Fall back to the run root on Windows when nested local file
                # artifact paths are rejected by the local store backend.
                mlflow.log_artifact(str(artifact))
            except Exception as fallback_exc:
                artifact_log_errors.append(f"{artifact.name}: {fallback_exc}")
    return artifact_log_errors


def _log_sklearn_bundle(
    mlflow,
    bundle: dict[str, Any],
    feature_columns: list[str],
    threshold: float,
    register_model: bool,
    registered_model_name: str | None,
) -> str | None:
    import mlflow.sklearn

    active_registered_model_name = registered_model_name
    mlflow.sklearn.log_model(
        sk_model=bundle["model"],
        name="model",
        registered_model_name=active_registered_model_name if register_model else None,
    )
    mlflow.log_dict(
        {
            "feature_columns": feature_columns,
            "threshold": threshold,
            "uses_scaled_features": bundle.get("uses_scaled_features", False),
        },
        "model/model_metadata.json",
    )
    if bundle.get("uses_scaled_features", False) and bundle.get("scaler") is not None:
        mlflow.log_dict(
            {
                "scaler_mean": bundle["scaler"].mean_.tolist(),
                "scaler_scale": bundle["scaler"].scale_.tolist(),
            },
            "model/scaler.json",
        )
    if register_model and active_registered_model_name is None:
        active_registered_model_name = "registered-sklearn-model"
    return active_registered_model_name


def log_training_run(
    config: dict[str, Any],
    metrics: dict[str, dict[str, float | int]],
    artifacts: list[Path],
    bundle: dict[str, Any],
    feature_columns: list[str],
    threshold: float,
    config_path: Path,
    register_model: bool,
    registered_model_name: str | None,
) -> dict[str, Any]:
    if not _tracking_enabled(config):
        return {"mlflow_logged": False, "registered_model_name": None, "artifact_log_errors": []}

    try:
        mlflow, client = configure_mlflow()
    except ImportError:
        return {"mlflow_logged": False, "registered_model_name": None, "artifact_log_errors": []}

    experiment_name = str(config.get("tracking", {}).get("experiment_name", "milestone4-facility-opening"))
    experiment_id = get_or_create_experiment(client, experiment_name)
    run_name = str(config.get("run_name", "m4-run"))
    model_name = bundle["model_name"]
    active_registered_model_name = registered_model_name

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        _ensure_run_artifact_dir(mlflow)
        _log_training_params(mlflow, config, config_path, model_name, feature_columns, threshold)
        _log_split_metrics(mlflow, metrics)

        if model_name in SKLEARN_MODEL_NAMES:
            active_registered_model_name = _log_sklearn_bundle(
                mlflow=mlflow,
                bundle=bundle,
                feature_columns=feature_columns,
                threshold=threshold,
                register_model=register_model,
                registered_model_name=active_registered_model_name,
            )
        artifact_log_errors = _log_supporting_artifacts(mlflow, artifacts)

    return {
        "mlflow_logged": True,
        "registered_model_name": active_registered_model_name,
        "artifact_log_errors": artifact_log_errors,
    }


def log_comparison_run(config: dict[str, Any], results_df, artifacts: list[Path]) -> bool:
    if not _tracking_enabled(config):
        return False

    try:
        mlflow, client = configure_mlflow()
    except ImportError:
        return False

    experiment_name = str(config.get("tracking", {}).get("experiment_name", "milestone4-model-comparison"))
    experiment_id = get_or_create_experiment(client, experiment_name)

    with mlflow.start_run(run_name=config.get("run_name", "m4-compare"), experiment_id=experiment_id):
        _ensure_run_artifact_dir(mlflow)
        mlflow.log_param("comparison_models", ",".join(results_df["model_name"].tolist()))
        for _, row in results_df.iterrows():
            with mlflow.start_run(run_name=str(row["model_name"]), nested=True):
                mlflow.log_param("model_name", str(row["model_name"]))
                mlflow.log_param("selected_threshold", float(row["selected_threshold"]))
                for key, value in row.items():
                    if key == "model_name":
                        continue
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, float(value))

        _log_supporting_artifacts(mlflow, artifacts)
    return True
