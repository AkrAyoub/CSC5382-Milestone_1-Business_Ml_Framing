# Milestone 5 - Model Productionization

This milestone productionizes the Milestone 1 system by turning the validated UFLP workflow into a served application. The production-safe default remains the deterministic CBC baseline, while the LLM-assisted symbolic path is exposed as an optional runtime for side-by-side evaluation rather than as the only production path.

### Table of Contents

- [1. Setup, Usage, and Deployment Guide](#1-setup-usage-and-deployment-guide)
  - [1.1 Repository Structure](#11-repository-structure)
  - [1.2 Installation and Setup](#12-installation-and-setup)
  - [1.3 Running the API Service](#13-running-the-api-service)
  - [1.4 Running the Front-End Client](#14-running-the-front-end-client)
  - [1.5 Running the Batch Serving Path](#15-running-the-batch-serving-path)
  - [1.6 Packaging the MLflow Runtime](#16-packaging-the-mlflow-runtime)
  - [1.7 Containerized Execution](#17-containerized-execution)
  - [1.8 Troubleshooting](#18-troubleshooting)
- [2. Milestone 5 - Model Productionization](#2-milestone-5---model-productionization)
  - [2.1 ML System Architecture](#21-ml-system-architecture)
  - [2.2 Application Development](#22-application-development)
    - [2.2.1 Proper Implementation of a Serving Mode](#221-proper-implementation-of-a-serving-mode)
    - [2.2.2 Model Service Development](#222-model-service-development)
    - [2.2.3 Front-End Client Development](#223-front-end-client-development)
  - [2.3 Integration and Deployment](#23-integration-and-deployment)
    - [2.3.1 Packaging and Containerization](#231-packaging-and-containerization)
    - [2.3.2 Integration with a CI/CD Pipeline](#232-integration-with-a-cicd-pipeline)
    - [2.3.3 Hosting the Application](#233-hosting-the-application)
  - [2.4 Model Serving](#24-model-serving)
    - [2.4.1 Model Serving Runtime](#241-model-serving-runtime)
- [3. References](#3-references)

## Quick Links

- Live Hugging Face Space: https://huggingface.co/spaces/AkrAyoub/uflp-production-solver
- Architecture drawing: [assets/architecture.png](assets/architecture.png)
- API app: [src/m5_productionization/api/main.py](src/m5_productionization/api/main.py)
- Service layer: [src/m5_productionization/service.py](src/m5_productionization/service.py)
- Runtime adapter: [src/m5_productionization/runtime.py](src/m5_productionization/runtime.py)
- Front-end client: [frontend/app.py](frontend/app.py)
- MLflow runtime packaging: [src/m5_productionization/mlflow_runtime.py](src/m5_productionization/mlflow_runtime.py)
- Compose deployment: [deployment/docker-compose.yml](deployment/docker-compose.yml)
- GitHub Actions workflow: [../.github/workflows/milestone5-ci-cd.yml](../.github/workflows/milestone5-ci-cd.yml)
- API contract: [docs/service_contract.md](docs/service_contract.md)
- CI/CD and deployment notes: [docs/cicd_and_deployment.md](docs/cicd_and_deployment.md)
- Docker setup on Windows: [docs/docker_setup_windows.md](docs/docker_setup_windows.md)
- Batch serving evidence: [reports/batch_smoke_summary.json](reports/batch_smoke_summary.json)
- Serving pipeline evidence: [reports/serving_pipeline_status.txt](reports/serving_pipeline_status.txt)
- Live fine-tuned serving smoke: [reports/live_finetuned_serving_smoke.json](reports/live_finetuned_serving_smoke.json)
- M5 evidence report: [reports/m5_evidence_report.txt](reports/m5_evidence_report.txt)
- Hugging Face Space bundle: [deployment/huggingface_space/](deployment/huggingface_space/)
- Shared dataset: [../data/raw/](../data/raw/)
- Milestone 4 backend base: [../Milestone_4-Model_Dev/README.md](../Milestone_4-Model_Dev/README.md)

## 1. Setup, Usage, and Deployment Guide

### 1.1 Repository Structure

Milestone 5 is organized around an API-first production stack:

- [src/m5_productionization/api/](src/m5_productionization/api/) - FastAPI routes and request/response schemas
- [src/m5_productionization/service.py](src/m5_productionization/service.py) - request handling, input materialization, and orchestration
- [src/m5_productionization/runtime.py](src/m5_productionization/runtime.py) - deterministic baseline runtime and optional LLM runtime
- [src/m5_productionization/catalog.py](src/m5_productionization/catalog.py) - shared benchmark catalog access
- [src/m5_productionization/mlflow_runtime.py](src/m5_productionization/mlflow_runtime.py) - MLflow-packaged serving runtime
- [frontend/app.py](frontend/app.py) - Streamlit frontend client
- [scripts/run_api.py](scripts/run_api.py) - local API runner
- [scripts/run_client.py](scripts/run_client.py) - local frontend runner
- [scripts/run_batch_job.py](scripts/run_batch_job.py) - batch-serving smoke workflow
- [deployment/](deployment/) - Dockerfiles, Compose stack, and deployment notes
- [tests/](tests/) - API and runtime tests

### 1.2 Installation and Setup

From the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
cd Milestone_5-ML_Productionization
pip install -r requirements.txt
pip install -e .
```

System requirement for the containerized deployment path:

- Docker Desktop with Docker Compose. Windows setup is documented in [docs/docker_setup_windows.md](docs/docker_setup_windows.md).

Optional environment variables:

```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:SELF_HOSTED_OPENAI_BASE_URL="http://localhost:8001/v1"
$env:SELF_HOSTED_OPENAI_API_KEY="local-dev-key"
$env:SELF_HOSTED_MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
$env:M5_SELF_HOSTED_FINE_TUNED_MODEL_NAME="optional-fine-tuned-model-id"
```

Notes:

- The deterministic baseline path does not require any API key.
- The production-safe default mode is `baseline`, even when the optional LLM path is available.
- The milestone reuses the shared benchmark data in [../data/raw/](../data/raw/).
- The default optional LLM path now targets the real OpenAI fine-tuned model from Milestone 4: `openai_gpt41_mini_finetuned`.

### 1.3 Running the API Service

Run the FastAPI service locally:

```powershell
python scripts/run_api.py
```

Important endpoints:

- `GET /healthz`
- `GET /api/v1/runtime`
- `GET /api/v1/catalog/instances`
- `POST /api/v1/solve`
- `POST /api/v1/batch/solve`

### 1.4 Running the Front-End Client

Start the Streamlit frontend client:

```powershell
python scripts/run_client.py
```

The client calls the FastAPI backend and supports:

- catalog instance selection
- inline raw-instance solving
- `auto`, `baseline`, `llm`, and `compare` modes
- optional full assignments and generated-code return payloads

### 1.5 Running the Batch Serving Path

Run the batch-serving smoke workflow:

```powershell
python scripts/run_batch_job.py
```

This executes a baseline batch job over three benchmark instances and writes:

- [reports/batch_smoke_summary.json](reports/batch_smoke_summary.json)

### 1.6 Packaging the MLflow Runtime

Package the production service as an MLflow pyfunc runtime:

```powershell
python scripts/package_mlflow_runtime.py
```

This packages the service as a local MLflow pyfunc runtime under `.m5_runtime/mlflow_runtime_model`. Registry registration is optional and can be enabled later with `M5_REGISTER_RUNTIME_MODEL=1` when the environment supports it cleanly.

### 1.6.1 Running the Serving Pipeline Evidence

Run the production serving validation pipeline:

```powershell
python scripts/run_serving_pipeline.py
```

This verifies the service as a serving runtime:

1. loads the selected M4 model/candidate metadata
2. validates `/healthz`, `/api/v1/runtime`, and `/api/v1/solve`
3. runs 12 API integration checks
4. packages the service as an MLflow pyfunc runtime

Outputs:

- [reports/serving_pipeline_status.json](reports/serving_pipeline_status.json)
- [reports/serving_pipeline_status.txt](reports/serving_pipeline_status.txt)

### 1.7 Containerized Execution

Run the API and frontend with Docker Compose:

```powershell
cd deployment
docker compose up --build
```

If Docker is not installed yet, complete [docs/docker_setup_windows.md](docs/docker_setup_windows.md) first.

If you also want the local self-hosted OpenAI-compatible model server in the same stack, enable the extra profile:

```powershell
cd deployment
docker compose --profile selfhosted up --build
```

Artifacts:

- [deployment/Dockerfile.api](deployment/Dockerfile.api)
- [deployment/Dockerfile.client](deployment/Dockerfile.client)
- [deployment/docker-compose.yml](deployment/docker-compose.yml)

### 1.7.1 Hugging Face Space Deployment

A Streamlit Hugging Face Space bundle is included in [deployment/huggingface_space/](deployment/huggingface_space/).

Build the bundle locally:

```powershell
python scripts/deploy_huggingface_space.py
```

This writes a deployable Space directory to:

```text
D:\csc5382_m5_hf_space_build
```

To publish it, set credentials and run:

```powershell
$env:HF_TOKEN="hf_..."
$env:HF_SPACE_ID="your-username/uflp-production-solver"
$env:M5_DEPLOY_HF="1"
python scripts/deploy_huggingface_space.py
```

After upload, configure `OPENAI_API_KEY` as a Hugging Face Space secret to enable live fine-tuned LLM mode. Baseline mode works without the OpenAI secret.

### 1.8 Troubleshooting

- If `m4_model_dev` imports fail, ensure M5 is installed with `pip install -e .`.
- If the optional LLM path returns `UNAVAILABLE`, set `SELF_HOSTED_OPENAI_BASE_URL` and, if needed, `SELF_HOSTED_OPENAI_API_KEY`.
- If the client cannot reach the backend, confirm the API base URL and that `python scripts/run_api.py` is already running.
- Validate the Compose file before deployment with `docker compose -f deployment/docker-compose.yml config`.

## 2. Milestone 5 - Model Productionization

This milestone turns the project into a served application while preserving the project identity fixed in Milestone 1. The deterministic OR-Tools/CBC solver is the production-safe default runtime, and the OpenAI fine-tuned symbolic generation path from Milestone 4 is available as an optional mode for controlled comparison and demos.

### 2.1 ML System Architecture

The architecture is documented in [assets/architecture.png](assets/architecture.png). The key production components are:

- a machine-facing FastAPI service
- a human-facing Streamlit client
- a runtime selector that defaults to the deterministic CBC baseline
- an optional hosted OpenAI fine-tuned runtime built on the M4 symbolic-generation stack
- MLflow runtime packaging for serving evidence
- Docker Compose deployment and GitHub Actions CI/CD

The architecture highlights the full production path:

1. Human users interact with the Streamlit client.
2. Machine clients call the FastAPI service directly.
3. The service resolves catalog or inline input into a UFLP instance.
4. The runtime selector chooses `auto`, `baseline`, `llm`, or `compare`.
5. OR-Tools/CBC computes the trusted baseline.
6. Optional OpenAI base/fine-tuned LLM candidates generate solver code, which is sandboxed and verified.
7. Results are returned through structured API schemas and can also be packaged as an MLflow pyfunc runtime.

This architecture keeps the milestone aligned with Milestone 1: the solver remains the trusted optimization engine and verification layer, while the LLM remains a symbolic model/code generator.

### 2.2 Application Development

#### 2.2.1 Proper Implementation of a Serving Mode

Milestone 5 implements three serving modes:

- `on demand to a machine`: FastAPI endpoints in [src/m5_productionization/api/main.py](src/m5_productionization/api/main.py)
- `on demand to a human`: Streamlit client in [frontend/app.py](frontend/app.py)
- `batch`: synchronous batch-serving workflow through `POST /api/v1/batch/solve` and [scripts/run_batch_job.py](scripts/run_batch_job.py)

The default `auto` path resolves to the deterministic baseline, which makes the public serving behavior safe and predictable.

The selectable runtime modes are:

- `auto`: production-safe default, currently the deterministic baseline.
- `baseline`: deterministic OR-Tools/CBC only.
- `llm`: selected LLM candidate plus baseline verification context.
- `compare`: baseline and selected LLM candidate side by side.

This covers both interactive and programmatic serving. The API is the production machine interface, the Streamlit app is the human interface, and the batch endpoint/script demonstrates grouped inference/serving.

#### 2.2.2 Model Service Development

The service core is implemented in:

- [src/m5_productionization/service.py](src/m5_productionization/service.py)
- [src/m5_productionization/runtime.py](src/m5_productionization/runtime.py)
- [src/m5_productionization/catalog.py](src/m5_productionization/catalog.py)

It supports:

- catalog-instance solving from the shared benchmark
- inline raw-instance solving
- baseline-only execution
- optional LLM candidate execution with sandbox validation
- consistent request/response contracts for API and MLflow runtime packaging

The current service contract is documented in [docs/service_contract.md](docs/service_contract.md).

The service supports two input sources:

- `catalog`: solve one of the known OR-Library benchmark instances from [../data/raw/](../data/raw/).
- `inline`: solve raw UFLP instance text supplied directly in the request.

The service always materializes input, runs the trusted baseline, and then optionally runs the selected LLM candidate. Failed candidate execution returns a structured partial-success response rather than silently replacing the baseline result.

#### 2.2.3 Front-End Client Development

The frontend client in [frontend/app.py](frontend/app.py) reuses the user-facing pattern of Milestone 2, but now calls the production API instead of running the full PoC locally. It allows:

- selecting catalog instances
- pasting inline instance text
- choosing serving mode and candidate
- inspecting baseline and optional candidate outputs
- viewing structured JSON responses for debugging and demos

The frontend is intentionally a client, not a second copy of the solver logic. This is important for productionization: the Streamlit app sends requests to the FastAPI service, and the service owns parsing, runtime selection, solving, candidate execution, and response formatting.

### 2.3 Integration and Deployment

#### 2.3.1 Packaging and Containerization

Packaging assets:

- [pyproject.toml](pyproject.toml)
- [requirements.txt](requirements.txt)
- [src/m5_productionization/mlflow_runtime.py](src/m5_productionization/mlflow_runtime.py)

Containerization assets:

- [deployment/Dockerfile.api](deployment/Dockerfile.api)
- [deployment/Dockerfile.client](deployment/Dockerfile.client)
- [deployment/docker-compose.yml](deployment/docker-compose.yml)

These files package the API and client separately while keeping them deployable together as one stack.

Docker usage:

- `Dockerfile.api` builds the FastAPI/Uvicorn backend image.
- `Dockerfile.client` builds the Streamlit frontend image.
- `docker-compose.yml` runs the backend and frontend together.
- The optional `selfhosted` profile adds a local OpenAI-compatible model-serving container.

Docker Desktop setup for Windows is documented in [docs/docker_setup_windows.md](docs/docker_setup_windows.md).

#### 2.3.2 Integration with a CI/CD Pipeline

CI/CD is implemented in [../.github/workflows/milestone5-ci-cd.yml](../.github/workflows/milestone5-ci-cd.yml).

The workflow performs:

1. dependency installation
2. pytest execution
3. API smoke verification
4. Docker image builds
5. optional SSH-driven deployment when repository secrets are configured

This gives the milestone both CI evidence and a deployment-ready CD path.

The workflow validates that the production code can be installed, tested, smoke-tested, and containerized from a clean GitHub Actions environment. The deploy job uses the same Docker Compose target documented for manual hosting.

#### 2.3.3 Hosting the Application

The selected hosting targets are:

- Docker Compose deployment on a Linux VM / DigitalOcean-style host, documented in [deployment/README.md](deployment/README.md) and [docs/cicd_and_deployment.md](docs/cicd_and_deployment.md)
- Hugging Face Spaces Streamlit deployment using [deployment/huggingface_space/](deployment/huggingface_space/) and [scripts/deploy_huggingface_space.py](scripts/deploy_huggingface_space.py)

The repository contains the deployment stack and the GitHub Actions deploy job for a Docker Compose host. Configure the documented SSH and self-hosted model secrets in GitHub Actions, then run the workflow from the `master` branch to deploy the API and client.

The same Compose stack can be tested locally through Docker Desktop. In local mode:

- FastAPI is exposed on `http://localhost:8000`.
- Streamlit is exposed on `http://localhost:8501`.
- API docs are exposed on `http://localhost:8000/docs`.

Verified hosting/deployment evidence:

- Hugging Face bundle build succeeds locally.
- GitHub Actions includes a `deploy-huggingface-space` job gated by `HF_TOKEN` and `HF_SPACE_ID` secrets.
- GitHub Actions includes an SSH-based Compose deployment job gated by VM secrets.

### 2.4 Model Serving

#### 2.4.1 Model Serving Runtime

The serving runtime combines:

- FastAPI/Uvicorn request serving
- direct runtime execution through [src/m5_productionization/runtime.py](src/m5_productionization/runtime.py)
- MLflow pyfunc packaging through [src/m5_productionization/mlflow_runtime.py](src/m5_productionization/mlflow_runtime.py)

The runtime intentionally uses the deterministic baseline as the production-safe default. The optional model path now targets the hosted OpenAI fine-tuned model selected from Milestone 4 outputs.

Live fine-tuned serving evidence:

- candidate: `openai_gpt41_mini_finetuned`
- model: `ft:gpt-4.1-mini-2025-04-14:aui:uflp-symbolic-solver:DaENP2ZH`
- candidate status: `OK`
- gap versus baseline on the smoke instance: `0.0%`
- saved output: [reports/live_finetuned_serving_smoke.json](reports/live_finetuned_serving_smoke.json)

Runtime evidence:

- API smoke test: [scripts/smoke_test.py](scripts/smoke_test.py)
- Batch serving smoke job: [scripts/run_batch_job.py](scripts/run_batch_job.py)
- MLflow runtime packaging: [scripts/package_mlflow_runtime.py](scripts/package_mlflow_runtime.py)
- Serving pipeline evidence: [scripts/run_serving_pipeline.py](scripts/run_serving_pipeline.py)
- Runtime tests: [tests/](tests/)

Alignment with previous milestones:

- Uses the same UFLP benchmark and raw data contract from M1.
- Extends the M2 Streamlit PoC into a service/client architecture.
- Consumes the M3 canonical data source and benchmark structure.
- Reuses the M4 symbolic-generation runtime direction and candidate naming.
- Keeps OR-Tools/CBC as the default trusted production runtime.

## 3. References

- Shared OR-Library benchmark: [../data/raw/](../data/raw/)
- FastAPI documentation: https://fastapi.tiangolo.com/
- Streamlit documentation: https://docs.streamlit.io/
- MLflow documentation: https://mlflow.org/
- Docker documentation: https://docs.docker.com/
- GitHub Actions documentation: https://docs.github.com/actions
