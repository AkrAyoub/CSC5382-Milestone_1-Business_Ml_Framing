from __future__ import annotations

from fastapi import FastAPI, HTTPException

from m5_productionization.api.schemas import BatchSolveRequest, BatchSolveResponse, RuntimeInfoResponse, ServiceInfoResponse, SolveRequest, SolveResponse
from m5_productionization.service import ProductionizationService


service = ProductionizationService()

app = FastAPI(
    title="CSC5382 M5 Production Service",
    version="0.1.0",
    description=(
        "Productionization layer for the AI-assisted symbolic optimization project. "
        "The deterministic baseline is the production-safe default runtime, while the LLM-assisted path remains available as an optional served mode."
    ),
)


@app.get("/", response_model=ServiceInfoResponse, tags=["service"])
def get_service_info() -> ServiceInfoResponse:
    return service.get_service_info()


@app.get("/healthz", tags=["service"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/runtime", response_model=RuntimeInfoResponse, tags=["runtime"])
def get_runtime_info() -> RuntimeInfoResponse:
    return service.get_runtime_info()


@app.get("/api/v1/catalog/instances", tags=["catalog"])
def list_catalog_instances() -> list[dict[str, object]]:
    return service.list_instances()


@app.post("/api/v1/solve", response_model=SolveResponse, tags=["serving"])
def solve_instance(request: SolveRequest) -> SolveResponse:
    try:
        return service.solve(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/v1/batch/solve", response_model=BatchSolveResponse, tags=["serving"])
def solve_batch(request: BatchSolveRequest) -> BatchSolveResponse:
    try:
        return service.solve_batch(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
