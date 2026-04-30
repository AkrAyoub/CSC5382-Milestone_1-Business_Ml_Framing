from __future__ import annotations

from typing import Any

import pandas as pd

from m5_productionization.api.schemas import InputSource, ServingMode, SolveRequest
import shutil

from m5_productionization.paths import M5_MLFLOW_DIR, M5_RUNTIME_DIR, ensure_runtime_dirs
from m5_productionization.service import ProductionizationService


MLFLOW_DB_PATH = M5_MLFLOW_DIR / "mlflow.db"
PACKAGED_MODEL_DIR = M5_RUNTIME_DIR / "mlflow_runtime_model"


def get_tracking_uri() -> str:
    return f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"


def _default_experiment_artifact_uri(experiment_name: str) -> str:
    sanitized = experiment_name.replace(" ", "-").lower()
    artifact_root = M5_MLFLOW_DIR / sanitized
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root.resolve().as_uri()


def configure_mlflow(*, create_client: bool = False):
    import mlflow

    ensure_runtime_dirs()
    M5_MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_registry_uri(get_tracking_uri())
    if create_client:
        from mlflow.tracking import MlflowClient

        return mlflow, MlflowClient()
    return mlflow, None


def _build_pyfunc_model_class(mlflow):
    class UFLPProductionPyFuncModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context) -> None:
            self.service = ProductionizationService()

        def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
            input_df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
            rows: list[dict[str, object]] = []
            for payload in input_df.to_dict(orient="records"):
                request = SolveRequest(
                    mode=ServingMode(str(payload.get("mode", "baseline"))),
                    source=InputSource(str(payload.get("source", "catalog"))),
                    instance_id=payload.get("instance_id"),
                    instance_text=payload.get("instance_text"),
                    candidate_name=payload.get("candidate_name"),
                    return_assignments=bool(payload.get("return_assignments", False)),
                    return_generated_code=bool(payload.get("return_generated_code", False)),
                    request_label=payload.get("request_label"),
                )
                response = self.service.solve(request)
                rows.append(
                    {
                        "request_id": response.request_id,
                        "instance_id": response.instance_id,
                        "served_mode": response.served_mode.value,
                        "overall_status": response.overall_status,
                        "baseline_objective": response.baseline.objective,
                        "candidate_status": response.candidate.status if response.candidate else "",
                        "candidate_name": response.candidate.name if response.candidate else "",
                        "candidate_objective": response.candidate.objective if response.candidate else None,
                        "production_default_used": response.production_default_used,
                    }
                )
            return pd.DataFrame(rows)

    return UFLPProductionPyFuncModel


def predict_with_service(model_input) -> pd.DataFrame:
    service = ProductionizationService()
    input_df = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
    rows: list[dict[str, object]] = []
    for payload in input_df.to_dict(orient="records"):
        request = SolveRequest(
            mode=ServingMode(str(payload.get("mode", "baseline"))),
            source=InputSource(str(payload.get("source", "catalog"))),
            instance_id=payload.get("instance_id"),
            instance_text=payload.get("instance_text"),
            candidate_name=payload.get("candidate_name"),
            return_assignments=bool(payload.get("return_assignments", False)),
            return_generated_code=bool(payload.get("return_generated_code", False)),
            request_label=payload.get("request_label"),
        )
        response = service.solve(request)
        rows.append(
            {
                "request_id": response.request_id,
                "instance_id": response.instance_id,
                "served_mode": response.served_mode.value,
                "overall_status": response.overall_status,
                "baseline_objective": response.baseline.objective,
                "candidate_status": response.candidate.status if response.candidate else "",
                "candidate_name": response.candidate.name if response.candidate else "",
                "candidate_objective": response.candidate.objective if response.candidate else None,
                "production_default_used": response.production_default_used,
            }
        )
    return pd.DataFrame(rows)


def log_runtime_model(
    *,
    run_name: str = "m5-production-runtime",
    registered_model_name: str = "m5-uflp-production-runtime",
    register_model: bool = False,
) -> tuple[str, str]:
    import mlflow.pyfunc

    mlflow, _client = configure_mlflow()
    ensure_runtime_dirs()
    if PACKAGED_MODEL_DIR.exists():
        shutil.rmtree(PACKAGED_MODEL_DIR)

    model_class = _build_pyfunc_model_class(mlflow)
    input_example = pd.DataFrame(
        [{"mode": "baseline", "source": "catalog", "instance_id": "cap71"}]
    )
    mlflow.pyfunc.save_model(
        path=str(PACKAGED_MODEL_DIR),
        python_model=model_class(),
        input_example=input_example,
    )
    model_uri = PACKAGED_MODEL_DIR.resolve().as_uri()

    if register_model:
        mlflow.register_model(model_uri, registered_model_name)
        return model_uri, registered_model_name

    return model_uri, ""
