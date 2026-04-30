# AI-Assisted Symbolic Optimization for Strategic Facility Network Design
## Milestone 4 - ML Pipeline Development - Model Training and Offline Evaluation

### Table of Contents

- [1. Setup, Usage, and Workflow Guide](#1-setup-usage-and-workflow-guide)
  - [1.1 Repository Structure](#11-repository-structure)
  - [1.2 Installation and Setup](#12-installation-and-setup)
  - [1.3 Running the Hosted OpenAI Base and Fine-Tuned Track](#13-running-the-hosted-openai-base-and-fine-tuned-track)
  - [1.4 Running the Single-Candidate Evaluation](#14-running-the-single-candidate-evaluation)
  - [1.5 Running the Model Comparison Workflow](#15-running-the-model-comparison-workflow)
  - [1.6 Running the Self-Hosted Fine-Tuning Workflow](#16-running-the-self-hosted-fine-tuning-workflow)
  - [1.7 Running the ZenML Workflow](#17-running-the-zenml-workflow)
  - [1.8 Produced Runtime Outputs](#18-produced-runtime-outputs)
  - [1.9 Evidence Report](#19-evidence-report)
  - [1.10 Troubleshooting](#110-troubleshooting)
- [2. Milestone 4 - ML Pipeline Development - Model Training and Offline Evaluation](#2-milestone-4---ml-pipeline-development---model-training-and-offline-evaluation)
  - [2.1 Project Structure Definition and Modularity](#21-project-structure-definition-and-modularity)
  - [2.2 Code Versioning](#22-code-versioning)
  - [2.3 Experiment Tracking and Model Versioning](#23-experiment-tracking-and-model-versioning)
  - [2.4 Integration of Model Training and Offline Evaluation into the ML Pipeline / MLOps Platform](#24-integration-of-model-training-and-offline-evaluation-into-the-ml-pipeline--mlops-platform)
  - [2.5 Optional Energy Efficiency Measurement](#25-optional-energy-efficiency-measurement)
- [3. References](#3-references)

## 1. Setup, Usage, and Workflow Guide

This milestone no longer trains a facility-level classifier. It now evaluates the actual Milestone 1 modeling component: an `LLM-as-modeler` system that generates solver-compatible UFLP code, executes it in a sandbox, and compares it against a deterministic CBC reference solver.

The milestone supports three local/demo execution paths plus one optional heavy training path:

- local single-candidate evaluation through MLflow
- multi-candidate offline comparison through MLflow
- end-to-end orchestration through ZenML
- optional self-hosted LoRA/QLoRA SFT training for a served/fine-tuned candidate

### 1.1 Repository Structure

- [`configs/`](configs/) - config files for single-candidate evaluation and multi-candidate comparison
- [`scripts/run_train.py`](scripts/run_train.py) - runs the single-candidate offline evaluation workflow
- [`scripts/run_compare.py`](scripts/run_compare.py) - runs the candidate comparison workflow
- [`scripts/run_zenml.py`](scripts/run_zenml.py) - runs the ZenML pipeline wrapper
- [`src/m4_model_dev/data/`](src/m4_model_dev/data/) - benchmark parsing, reference solutions, benchmark dataset, grouped splits, and SFT dataset creation
- [`src/m4_model_dev/models/`](src/m4_model_dev/models/) - candidate registry and LLM symbolic code generation
- [`src/m4_model_dev/evaluation/`](src/m4_model_dev/evaluation/) - sandbox execution and offline evaluation metrics
- [`src/m4_model_dev/pipelines/`](src/m4_model_dev/pipelines/) - local and ZenML-integrated training/evaluation workflows
- [`src/m4_model_dev/reporting/`](src/m4_model_dev/reporting/) - report tables, summaries, and figures
- [`src/m4_model_dev/tracking/`](src/m4_model_dev/tracking/) - MLflow and CodeCarbon integration
- [`src/m4_model_dev/training/openai_finetune.py`](src/m4_model_dev/training/openai_finetune.py) - OpenAI supervised fine-tuning export, job creation, and status tracking
- [`tests/`](tests/) - config, registry, and comparison-selection tests
- [`data/reference/`](data/reference/) - deterministic reference solutions built from the shared OR-Library benchmark
- [`data/datasets/`](data/datasets/) - instance-level benchmark dataset
- [`data/splits/`](data/splits/) - grouped train/validation/test split definition
- [`data/sft/`](data/sft/) - prompt/response SFT dataset prepared for optional fine-tuning
- [`reports/`](reports/) - generated evaluation summaries, figures, comparison outputs, energy logs, and ZenML status artifacts

### 1.2 Installation and Setup

Using the shared project virtual environment, install the milestone requirements from the `CSC5382-Project` repo root:

```bash
..\.venv\Scripts\Activate.ps1
..\.venv\Scripts\python.exe -m pip install -r Milestone_4-Model_Dev\requirements.txt
cd Milestone_4-Model_Dev
```

Environment variable for hosted OpenAI-backed candidates:

```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
```

The Python helpers also read the Windows User/Machine `OPENAI_API_KEY` when the current PowerShell process was started before the key was set.

Optional environment variables for self-hosted LLM-backed candidates:

```bash
$env:SELF_HOSTED_OPENAI_BASE_URL="http://localhost:8001/v1"
$env:SELF_HOSTED_OPENAI_API_KEY="local-dev-key"
$env:SELF_HOSTED_MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
$env:SELF_HOSTED_FINE_TUNED_MODEL_NAME="path-or-served-model-name-for-fine-tuned-adapter"
```

Notes:

- `zenml[local,server]` and its local-store/dashboard dependencies are already included in [`requirements.txt`](requirements.txt).
- Real self-hosted training dependencies are listed separately in [`requirements-selfhosted.txt`](requirements-selfhosted.txt).
- MLflow artifacts are written to a short local runtime path at the current drive root, e.g. `D:\csc5382_m4_mlruns_openai`, to avoid Windows path-length failures from deep course folders.
- ZenML artifact-store data is written to a short local runtime path at the current drive root, e.g. `D:\csc5382_m4_zen_store`; ZenML config state is written under `Milestone_4-Model_Dev/.zen_m4_runtime/`.
- Generated solver code is written under `reports/generated_code/` and is ignored as a local runtime artifact.
- The self-hosted SFT/LoRA entry point is [`scripts/run_self_hosted_train.py`](scripts/run_self_hosted_train.py) using [`configs/train_self_hosted_fine_tuned.yaml`](configs/train_self_hosted_fine_tuned.yaml).

### 1.3 Running the Hosted OpenAI Base and Fine-Tuned Track

This is the strongest current M4 evidence path. It uses a real hosted OpenAI model plus a real supervised fine-tuned OpenAI model:

- base model: `gpt-4.1-mini-2025-04-14`
- fine-tuned model: `ft:gpt-4.1-mini-2025-04-14:aui:uflp-symbolic-solver:DaENP2ZH`
- fine-tuning job: [`reports/openai_finetune_job.json`](reports/openai_finetune_job.json)
- OpenAI fine-tuning data manifest: [`data/openai_finetune/openai_finetune_manifest.json`](data/openai_finetune/openai_finetune_manifest.json)

Commands:

```powershell
python scripts/export_openai_finetune_data.py
python scripts/run_openai_finetune.py --model gpt-4.1-mini-2025-04-14 --suffix uflp-symbolic-solver
python scripts/check_openai_finetune.py
python scripts/run_train.py configs/train_openai_base.yaml
python scripts/run_compare.py configs/compare_openai_models.yaml
python scripts/write_m4_evidence_report.py
```

Current verified results:

| Candidate | Split | Generation | Execution | Exact Match | Attempted |
|---|---:|---:|---:|---:|---:|
| `openai_gpt41_mini_base` | train | 1.0000 | 1.0000 | 1.0000 | 7 |
| `openai_gpt41_mini_base` | val | 1.0000 | 1.0000 | 1.0000 | 4 |
| `openai_gpt41_mini_base` | test | 1.0000 | 1.0000 | 1.0000 | 4 |
| `openai_gpt41_mini_finetuned` | train | 1.0000 | 1.0000 | 1.0000 | 7 |
| `openai_gpt41_mini_finetuned` | val | 1.0000 | 1.0000 | 1.0000 | 4 |
| `openai_gpt41_mini_finetuned` | test | 1.0000 | 1.0000 | 1.0000 | 4 |

The comparison output confirms no template fallback was used for these OpenAI runs. Evidence is saved in [`reports/model_comparison.csv`](reports/model_comparison.csv), [`reports/model_selection.json`](reports/model_selection.json), and [`reports/m4_evidence_report.txt`](reports/m4_evidence_report.txt).

### 1.4 Running the Single-Candidate Evaluation

The default single-candidate configuration evaluates the `llm_robust_prompt_v1` self-hosted candidate:

```bash
python scripts/run_train.py
```

By default, this config allows a validated template fallback when no OpenAI-compatible model endpoint is running. This keeps the local grading/demo workflow reproducible.

To require a real live model call and fail if no endpoint is available, run:

```bash
python scripts/run_train.py configs/train_live_llm.yaml
```

For a Groq/OpenAI-compatible endpoint, set the endpoint and model before running the live config. Example:

```powershell
$env:SELF_HOSTED_OPENAI_BASE_URL="https://api.groq.com/openai/v1"
$env:SELF_HOSTED_OPENAI_API_KEY="your-api-key"
$env:SELF_HOSTED_MODEL_NAME="llama-3.1-8b-instant"
python scripts/run_train.py configs/train_live_llm.yaml
```

Main outputs:

- [`reports/summary.txt`](reports/summary.txt)
- [`reports/run_manifest.json`](reports/run_manifest.json)
- [`reports/evaluation/single_candidate_metrics.csv`](reports/evaluation/single_candidate_metrics.csv)
- [`reports/evaluation/single_candidate_raw_results.csv`](reports/evaluation/single_candidate_raw_results.csv)
- [`reports/training_metrics.png`](reports/training_metrics.png)
- [`reports/training_status.png`](reports/training_status.png)
- [`reports/training_dashboard.png`](reports/training_dashboard.png)
- [`artifacts/llm_robust_prompt_v1_spec.json`](artifacts/llm_robust_prompt_v1_spec.json)

### 1.5 Running the Model Comparison Workflow

The default comparison workflow evaluates the deterministic reference baseline, the hosted OpenAI base candidate, and the hosted OpenAI fine-tuned candidate:

```bash
python scripts/run_compare.py
```

Main outputs:

- [`reports/model_comparison.csv`](reports/model_comparison.csv)
- [`reports/model_comparison.json`](reports/model_comparison.json)
- [`reports/model_comparison_raw_results.csv`](reports/model_comparison_raw_results.csv)
- [`reports/model_comparison_summary.txt`](reports/model_comparison_summary.txt)
- [`reports/model_selection.json`](reports/model_selection.json)
- [`reports/comparison_validation_metrics.png`](reports/comparison_validation_metrics.png)
- [`reports/comparison_test_metrics.png`](reports/comparison_test_metrics.png)
- [`reports/comparison_dashboard.png`](reports/comparison_dashboard.png)

### 1.6 Running the Self-Hosted Fine-Tuning Workflow

Install the additional training stack first:

```bash
pip install -r requirements-selfhosted.txt
```

First run the tiny local smoke fine-tuning config. This proves that the SFT dataset, LoRA training code, checkpoint saving, and MLflow logging work without requiring a large model:

```bash
python scripts/run_self_hosted_train.py configs/train_self_hosted_fine_tuned_smoke.yaml
```

Then run the full Qwen 7B supervised fine-tuning config only on suitable GPU/cloud hardware:

```bash
python scripts/run_self_hosted_train.py
```

Both commands produce a local LoRA/QLoRA training run manifest under `data/training_runs/` and log the run to MLflow.

The smoke config targets `sshleifer/tiny-gpt2` for implementation verification only. The default config targets `Qwen/Qwen2.5-Coder-7B-Instruct`, so it requires the Hugging Face training stack, model download access, and suitable GPU memory or an adjusted smaller/quantized training config.

If the heavy training packages are not installed, this script fails fast before rebuilding data assets and tells you which packages are missing.

### 1.7 Running the ZenML Workflow

The same training/evaluation flow is exposed through ZenML:

```bash
python scripts/run_zenml.py
```

ZenML status outputs:

- [`reports/zenml_status.json`](reports/zenml_status.json)
- [`reports/zenml_status.txt`](reports/zenml_status.txt)

### 1.8 Produced Runtime Outputs

Key generated assets after the verified local runs:

- deterministic reference solutions: [`data/reference/reference_solutions.csv`](data/reference/reference_solutions.csv)
- benchmark instance dataset: [`data/datasets/benchmark_instances.csv`](data/datasets/benchmark_instances.csv)
- grouped split definition: [`data/splits/instance_splits.csv`](data/splits/instance_splits.csv)
- SFT dataset manifest: [`data/sft/sft_manifest.json`](data/sft/sft_manifest.json)
- OpenAI fine-tuning dataset manifest: [`data/openai_finetune/openai_finetune_manifest.json`](data/openai_finetune/openai_finetune_manifest.json)
- OpenAI fine-tuning job metadata: [`reports/openai_finetune_job.json`](reports/openai_finetune_job.json)
- MLflow-registered single-candidate artifact family: local registry name `m4-symbolic-generator-best`
- current ZenML integration status: [`reports/zenml_status.json`](reports/zenml_status.json)
- compact evidence report: [`reports/m4_evidence_report.txt`](reports/m4_evidence_report.txt) and [`reports/m4_evidence_report.json`](reports/m4_evidence_report.json)

### 1.9 Evidence Report

After running the M4 workflows, write the compact evidence report:

```bash
python scripts/write_m4_evidence_report.py
```

The report records:

- unit-test status from [`scripts/run_tests.py`](scripts/run_tests.py)
- ZenML execution status from [`reports/zenml_status.json`](reports/zenml_status.json)
- MLflow logging and registered model name from [`reports/run_manifest.json`](reports/run_manifest.json)
- selected comparison candidate from [`reports/model_selection.json`](reports/model_selection.json)
- whether comparison results used the validated template fallback
- OpenAI fine-tuning status and fine-tuned model ID
- live single-candidate success/failure counts
- latest CodeCarbon emissions from [`reports/emissions.csv`](reports/emissions.csv)
- current model-versioning artifacts from [`artifacts/`](artifacts/)

Current evidence summary:

- tests passed: `true`
- ZenML success: `true`
- MLflow logged: `true`
- registered model name: `m4-openai-gpt41-mini-base`
- current selected comparison candidate: `openai_gpt41_mini_base`
- OpenAI fine-tuned candidate enabled: `true`
- fine-tuned model: `ft:gpt-4.1-mini-2025-04-14:aui:uflp-symbolic-solver:DaENP2ZH`

### 1.10 Troubleshooting

- `Missing SELF_HOSTED_OPENAI_BASE_URL`
  - Set `SELF_HOSTED_OPENAI_BASE_URL` before running the live self-hosted path.
  - If the endpoint is absent, M4 falls back to a validated template implementation so the symbolic candidate path remains runnable offline.

- Self-hosted runtime unavailable or generated code fails validation
  - Confirm that the OpenAI-compatible server is reachable and that `SELF_HOSTED_MODEL_NAME` points to the served base model.
  - If the live response still fails static/runtime checks, M4 falls back to a validated template implementation for the same candidate contract.

- `Windows path too long`
  - MLflow and ZenML runtime stores are intentionally placed at short drive-root paths, e.g. `D:\csc5382_m4_mlruns_openai` and `D:\csc5382_m4_zen_store`.
  - If you move the repo to another drive, these paths move to that drive root automatically.

- ZenML local-store errors
  - Reinstall from [`requirements.txt`](requirements.txt) in a fresh virtual environment.
  - Then rerun `python scripts/run_zenml.py`.

- Live LLM run falls back unexpectedly
  - Use `configs/train_live_llm.yaml`; it sets `runtime.allow_template_fallback: false`.
  - The default configs intentionally set `runtime.allow_template_fallback: true` for local reproducibility.

## 2. Milestone 4 - ML Pipeline Development - Model Training and Offline Evaluation

This milestone realigns the project with Milestone 1 by evaluating symbolic optimization-model generation rather than a proxy supervised classifier. The deterministic CBC solver remains the trusted reference baseline. M4 develops and evaluates multiple candidate generator variants:

- `deterministic_baseline`
- `openai_gpt41_mini_base`
- `openai_gpt41_mini_finetuned`
- `llm_token_prompt_v0`
- `llm_robust_prompt_v1`
- `llm_fine_tuned` candidate slot for a configured fine-tuned/self-hosted model

The current implementation keeps the stricter offline evaluation contract and now includes a real hosted OpenAI path. The submitted evidence includes both a base hosted LLM and a working supervised fine-tuned hosted LLM. Template fallback remains available only for offline reproducibility and is explicitly marked when used.

Important result interpretation:

- [`reports/evaluation/single_candidate_metrics.csv`](reports/evaluation/single_candidate_metrics.csv) records the latest live OpenAI base run. It used backend `openai`, attempted 15 instances, and completed with 15 successes and 0 failures.
- [`reports/model_comparison.csv`](reports/model_comparison.csv) records the base-vs-fine-tuned comparison. The OpenAI base and fine-tuned rows both used backend `openai`, not template fallback, and reached 1.0000 generation, execution, and exact-match rates on train/validation/test.
- The deterministic CBC baseline remains the trusted reference verifier. The LLM candidate is useful when it generates solver-compatible code that passes the sandbox and objective checks.
- The hosted fine-tuned candidate is implemented through [`src/m4_model_dev/training/openai_finetune.py`](src/m4_model_dev/training/openai_finetune.py), with job metadata saved in [`reports/openai_finetune_job.json`](reports/openai_finetune_job.json).

### 2.1 Project Structure Definition and Modularity

The milestone was restructured around clear module boundaries instead of the old classifier workflow:

- benchmark and reference-data construction in [`src/m4_model_dev/data/benchmark.py`](src/m4_model_dev/data/benchmark.py), [`src/m4_model_dev/data/build_reference_solutions.py`](src/m4_model_dev/data/build_reference_solutions.py), and [`src/m4_model_dev/data/build_benchmark_dataset.py`](src/m4_model_dev/data/build_benchmark_dataset.py)
- split definition and optional fine-tuning dataset creation in [`src/m4_model_dev/data/make_splits.py`](src/m4_model_dev/data/make_splits.py) and [`src/m4_model_dev/data/build_sft_dataset.py`](src/m4_model_dev/data/build_sft_dataset.py)
- candidate definitions in [`src/m4_model_dev/models/model_registry.py`](src/m4_model_dev/models/model_registry.py)
- LLM code generation and validation in [`src/m4_model_dev/models/symbolic_generator.py`](src/m4_model_dev/models/symbolic_generator.py)
- sandboxed execution in [`src/m4_model_dev/evaluation/generated_exec.py`](src/m4_model_dev/evaluation/generated_exec.py)
- split-level metric aggregation in [`src/m4_model_dev/evaluation/metrics.py`](src/m4_model_dev/evaluation/metrics.py)
- local and ZenML pipelines in [`src/m4_model_dev/pipelines/training_pipeline.py`](src/m4_model_dev/pipelines/training_pipeline.py), [`src/m4_model_dev/pipelines/comparison_pipeline.py`](src/m4_model_dev/pipelines/comparison_pipeline.py), and [`src/m4_model_dev/pipelines/zenml_pipeline.py`](src/m4_model_dev/pipelines/zenml_pipeline.py)
- reporting and figures in [`src/m4_model_dev/reporting/`](src/m4_model_dev/reporting/)

This modular layout supports the hosted OpenAI base/fine-tuned candidates as well as self-hosted candidates without changing the pipeline shape.

The project is organized around the actual M1 modeling objective: generate UFLP solver code, execute it, and evaluate it against the deterministic OR-Tools/CBC reference. The previous facility-level classifier artifacts are no longer the main model-development path because they did not directly match the project inception contract.

### 2.2 Code Versioning

Code versioning is handled through the Git repository and milestone-local modularization:

- Milestone 4 is isolated under [`Milestone_4-Model_Dev/`](.)
- runtime behavior is controlled through versioned config files in [`configs/`](configs/)
- reproducible entry points are exposed through [`scripts/`](scripts/)
- regression coverage is provided through [`tests/`](tests/)

The refactor also removed the old classifier-specific files and replaced them with the symbolic-evaluation implementation that matches the Milestone 1 project identity.

Versioned M4 assets include:

- pipeline source code under [src/m4_model_dev/](src/m4_model_dev/)
- experiment configs under [configs/](configs/)
- runnable entrypoints under [scripts/](scripts/)
- tests under [tests/](tests/)
- documentation and generated report summaries under [reports/](reports/)

This gives the milestone a reproducible code/config history. The final GitHub Flow evidence is produced by committing this refactor and pushing it through the repository workflow.

### 2.3 Experiment Tracking and Model Versioning

MLflow is used for experiment tracking and local model versioning through [`src/m4_model_dev/tracking/mlflow_utils.py`](src/m4_model_dev/tracking/mlflow_utils.py).

Tracked run contents include:

- candidate metadata: backend, model name, prompt template, token budget, enabled state
- split-level offline metrics from [`reports/evaluation/single_candidate_metrics.csv`](reports/evaluation/single_candidate_metrics.csv) and [`reports/model_comparison.csv`](reports/model_comparison.csv)
- raw evaluation tables from [`reports/evaluation/single_candidate_raw_results.csv`](reports/evaluation/single_candidate_raw_results.csv) and [`reports/model_comparison_raw_results.csv`](reports/model_comparison_raw_results.csv)
- run manifests and report figures
- a registered local model artifact family under `m4-symbolic-generator-best`

Model development is therefore tracked as versioned candidate specifications, a structured SFT dataset, optional LoRA/QLoRA checkpoints, and solver-grounded offline evaluation outcomes rather than as a proxy classifier.

Current model/candidate implementation:

- `deterministic_baseline`: trusted OR-Tools/CBC reference solver.
- `openai_gpt41_mini_base`: hosted OpenAI `gpt-4.1-mini-2025-04-14` symbolic solver generator.
- `openai_gpt41_mini_finetuned`: hosted OpenAI supervised fine-tuned model `ft:gpt-4.1-mini-2025-04-14:aui:uflp-symbolic-solver:DaENP2ZH`.
- `llm_token_prompt_v0`: first symbolic-generation candidate.
- `llm_robust_prompt_v1`: stronger symbolic-generation prompt and current default LLM candidate.
- `llm_fine_tuned`: fine-tuned/self-hosted candidate slot, enabled when a fine-tuned served model or adapter path is configured.
- `smoke_tiny_gpt2_sft`: local tiny SFT smoke path used to verify the fine-tuning implementation without requiring a large GPU model.

The default local training/evaluation configs can use a validated template fallback for reproducibility. The OpenAI configs [configs/train_openai_base.yaml](configs/train_openai_base.yaml) and [configs/compare_openai_models.yaml](configs/compare_openai_models.yaml) disable fallback and require real hosted model calls.

Saved model-versioning evidence:

- candidate specs: [`artifacts/deterministic_baseline_spec.json`](artifacts/deterministic_baseline_spec.json), [`artifacts/llm_token_prompt_v0_spec.json`](artifacts/llm_token_prompt_v0_spec.json), [`artifacts/llm_robust_prompt_v1_spec.json`](artifacts/llm_robust_prompt_v1_spec.json), [`artifacts/llm_fine_tuned_spec.json`](artifacts/llm_fine_tuned_spec.json)
- MLflow registered model family from the latest run: `m4-openai-gpt41-mini-base`
- run manifest with MLflow status and registered model name: [`reports/run_manifest.json`](reports/run_manifest.json)
- selected candidate metadata for Milestone 5 serving: [`reports/model_selection.json`](reports/model_selection.json)
- optional SFT data manifest for later fine-tuning: [`data/sft/sft_manifest.json`](data/sft/sft_manifest.json)
- OpenAI fine-tuning job metadata: [`reports/openai_finetune_job.json`](reports/openai_finetune_job.json)

### 2.4 Integration of Model Training and Offline Evaluation into the ML Pipeline / MLOps Platform

The milestone integrates the full offline workflow into both a local pipeline and a ZenML pipeline.

Local pipeline flow:

1. Build deterministic reference solutions from the shared benchmark.
2. Build the benchmark instance dataset.
3. Build grouped train/validation/test splits.
4. Build an SFT prompt/response dataset for future fine-tuning.
5. Evaluate one candidate or multiple candidates offline.
6. Aggregate split-level metrics and write report artifacts.
7. Log the run to MLflow.

The same flow is wrapped by ZenML in [`src/m4_model_dev/pipelines/zenml_pipeline.py`](src/m4_model_dev/pipelines/zenml_pipeline.py), and the verified status is recorded in [`reports/zenml_status.json`](reports/zenml_status.json).

Offline evaluation metrics are solver-grounded and aligned with Milestone 1:

- generation success rate
- execution success rate
- feasibility rate
- exact match rate against the deterministic baseline
- objective gap versus the deterministic baseline
- objective gap versus the best known OR-Library optimum when available
- runtime per split

Current verified comparison evidence:

- the deterministic baseline remains exact and reliable
- the hosted OpenAI base model is live and evaluated under the same solver-grounded contract
- the hosted OpenAI fine-tuned model is live, tracked, and evaluated under the same solver-grounded contract
- the latest OpenAI comparison used no template fallback rows
- template fallback remains available only for reproducible offline demos and is explicitly marked in outputs when used

The hosted fine-tuning implementation is present in [src/m4_model_dev/training/openai_finetune.py](src/m4_model_dev/training/openai_finetune.py) and can be run through [scripts/run_openai_finetune.py](scripts/run_openai_finetune.py). The self-hosted LoRA path remains available in [src/m4_model_dev/training/sft_training.py](src/m4_model_dev/training/sft_training.py) for later GPU-based experiments.

Alignment with previous milestones:

- Uses the same UFLP benchmark from M1.
- Keeps OR-Tools/CBC as the reference verifier from M1/M2.
- Uses M3-derived benchmark splits and symbolic SFT data.
- Produces the candidate-selection metadata consumed by M5 production serving.

### 2.5 Optional Energy Efficiency Measurement

Optional energy measurement is enabled for the single-candidate workflow through CodeCarbon in [`src/m4_model_dev/tracking/codecarbon_utils.py`](src/m4_model_dev/tracking/codecarbon_utils.py).

Current energy evidence is written to:

- [`reports/emissions.csv`](reports/emissions.csv)

This keeps energy-efficiency measurement as a real runnable option instead of a documentation-only claim.

## 3. References

- OR-Library UFLP benchmark: [`../data/raw/`](../data/raw/)
- OR-Tools linear solver: https://developers.google.com/optimization
- MLflow documentation: https://mlflow.org/
- ZenML documentation: https://docs.zenml.io/
- CodeCarbon documentation: https://mlco2.github.io/codecarbon/
- OpenAI-compatible API serving with vLLM: https://docs.vllm.ai/
