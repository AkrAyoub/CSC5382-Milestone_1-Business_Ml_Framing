# Milestone 3 - Data Acquisition, Validation, and Preparation

This milestone implements the data layer for the UFLP project defined in Milestone 1 and used by the solver PoC in Milestone 2. The refactor keeps the project on one shared benchmark source, builds reproducible derived datasets, validates them against explicit contracts, versions the pipeline with DVC, and exposes engineered features through Feast.

### Table of Contents

- [1. Setup, Usage, and Pipeline Guide](#1-setup-usage-and-pipeline-guide)
  - [1.1 Repository Structure](#11-repository-structure)
  - [1.2 Installation and Setup](#12-installation-and-setup)
  - [1.3 Running the Pipeline](#13-running-the-pipeline)
  - [1.4 Native UI and Visualization Tools](#14-native-ui-and-visualization-tools)
  - [1.5 Troubleshooting](#15-troubleshooting)
- [2. Milestone 3 - ML Pipeline Development - Data Ingestion, Validation and Preparation](#2-milestone-3---ml-pipeline-development---data-ingestion-validation-and-preparation)
  - [2.1 Schema Definition](#21-schema-definition)
  - [2.2 Data Validation and Verification](#22-data-validation-and-verification)
  - [2.3 Data Versioning](#23-data-versioning)
  - [2.4 Setting up a Feature Store](#24-setting-up-a-feature-store)
  - [2.5 Setup of Data Pipeline within the Larger ML Pipeline / MLOps Platform](#25-setup-of-data-pipeline-within-the-larger-ml-pipeline--mlops-platform)
    - [2.5.1 Ingestion of Raw Data and Storage into a Repository](#251-ingestion-of-raw-data-and-storage-into-a-repository)
    - [2.5.2 Preprocessing and Feature Engineering](#252-preprocessing-and-feature-engineering)
- [3. References](#3-references)

## Quick Links

- Shared raw dataset: [../data/raw/](../data/raw/)
- DVC pipeline: [dvc.yaml](dvc.yaml)
- Pipeline runner: [pipeline/run_data_pipeline.py](pipeline/run_data_pipeline.py)
- Full workflow runner: [run_full_workflow.py](run_full_workflow.py)
- ZenML workflow runner: [run_zenml_workflow.py](run_zenml_workflow.py)
- ZenML pipeline definition: [pipeline/zenml_pipeline.py](pipeline/zenml_pipeline.py)
- Symbolic SFT dataset builder: [pipeline/build_symbolic_sft_dataset.py](pipeline/build_symbolic_sft_dataset.py)
- Symbolic SFT manifest: [data/training/symbolic_sft_manifest.json](data/training/symbolic_sft_manifest.json)
- Validation summary: [reports/validation/validation_summary.json](reports/validation/validation_summary.json)
- ZenML status: [reports/zenml_status.json](reports/zenml_status.json)
- Pipeline evidence report: [reports/m3_pipeline_results.txt](reports/m3_pipeline_results.txt)
- Pipeline evidence JSON: [reports/m3_pipeline_results.json](reports/m3_pipeline_results.json)
- Feast demo: [feature_repo/run_feature_store_demo.py](feature_repo/run_feature_store_demo.py)
- Milestone 1 README: [../Milestone_1-Project_Inception/README.md](../Milestone_1-Project_Inception/README.md)
- Milestone 2 README: [../Milestone_2-PoC/README.md](../Milestone_2-PoC/README.md)

## 1. Setup, Usage, and Pipeline Guide

### 1.1 Repository Structure

Milestone 3 uses the shared root raw dataset and keeps all derived artifacts inside the milestone folder:

- [../data/raw/](../data/raw/): canonical OR-Library UFLP input and [../data/raw/uncapopt.txt](../data/raw/uncapopt.txt)
- [pipeline/ingest_data.py](pipeline/ingest_data.py): manifest creation from the shared raw dataset
- [pipeline/preprocess_data.py](pipeline/preprocess_data.py): normalized relational tables
- [pipeline/engineer_features.py](pipeline/engineer_features.py): instance-, facility-, and customer-level features
- [pipeline/validate_data.py](pipeline/validate_data.py): schema checks, statistics, anomaly checks, and consistency validation
- [pipeline/build_symbolic_sft_dataset.py](pipeline/build_symbolic_sft_dataset.py): symbolic prompt/response/code dataset for the self-hosted fine-tuning track
- [feature_repo/](feature_repo/): Feast entities, views, apply script, and retrieval demo
- [data/interim/](data/interim/), [data/processed/](data/processed/), [data/features/](data/features/), [data/training/](data/training/): generated M3 outputs
- [reports/validation/](reports/validation/) and [reports/stats/](reports/stats/): validation and statistics outputs

### 1.2 Installation and Setup

From the repo root:

```powershell
python -m venv .venv-m3
.venv-m3\Scripts\Activate.ps1
cd Milestone_3-Data_Prep
pip install -r requirements-m3.txt
```

[requirements-m3.txt](requirements-m3.txt) now includes the ZenML local-runtime and dashboard-server dependencies used by [run_zenml_workflow.py](run_zenml_workflow.py), including `zenml[local,server]`, `sqlmodel`, `passlib[bcrypt]`, and the supporting SQL/runtime packages needed by the local ZenML store.

For deep Windows paths, DVC works more reliably with a short local cache path:

```powershell
..\.venv-m3\Scripts\python.exe -m dvc config cache.dir D:/dvc-cache-csc5382-m3 --local
```

If you use a different short local path, substitute it in the command above.

### 1.3 Running the Pipeline

Run the data pipeline only:

```powershell
python pipeline/run_data_pipeline.py
```

Run the full workflow, including Feast apply and retrieval demo:

```powershell
python run_full_workflow.py
```

Run the ZenML-orchestrated workflow:

```powershell
python run_zenml_workflow.py
```

### 1.4 Native UI and Visualization Tools

ZenML is the main run-visualization UI for this milestone. It shows pipeline runs, steps, artifacts, and execution status. On Windows, the local ZenML server must run in blocking mode:

```powershell
$env:ZENML_CONFIG_PATH = (Resolve-Path .\.zen_m3_runtime).Path
$env:ZENML_LOCAL_STORES_PATH = "D:\zen-m3-store-csc5382"
..\.venv-m3\Scripts\zenml.exe login --local --blocking
```

On Windows, ZenML must run the local dashboard in blocking mode. Keep that terminal open while inspecting the dashboard, then press `Ctrl+C` to stop it. In a second terminal, use `zenml show` to open the dashboard URL if needed.

Feast provides a local web UI for feature-store metadata, including entities, feature views, data sources, and registry metadata:

```powershell
..\.venv-m3\Scripts\feast.exe -c feature_repo ui --host 127.0.0.1 --port 8888
```

Then open:

```text
http://localhost:8888
```

DVC provides local pipeline graph visualization in the terminal:

```powershell
..\.venv-m3\Scripts\python.exe -m dvc dag
```

DVC Studio is the optional external web UI for DVC experiments and pipeline/project tracking. It is not required for local Milestone 3 execution and will be configured later with the DVC remote/push final verification step:

```powershell
..\.venv-m3\Scripts\python.exe -m dvc studio login
```

Run DVC reproduction:

```powershell
..\.venv-m3\Scripts\python.exe -m dvc repro
```

Write the compact milestone evidence report after running the workflow:

```powershell
python pipeline/write_pipeline_report.py
```

The report records validation counts, anomaly status, DVC status, DVC DAG output, Feast historical retrieval output, and the most recent ZenML execution status.

### 1.5 Troubleshooting

- If raw data is not found, confirm that the shared dataset exists in [../data/raw/](../data/raw/).
- If DVC fails on Windows path length, use a short local cache path as shown above.
- If Feast retrieval fails, rerun the feature stage first, then apply the repo with [feature_repo/apply_repo.py](feature_repo/apply_repo.py).
- If ZenML reports missing modules such as `sqlmodel`, `passlib`, or `multipart`, reinstall [requirements-m3.txt](requirements-m3.txt) in the active environment and rerun [run_zenml_workflow.py](run_zenml_workflow.py).
- If the environment has older conflicting ZenML dependencies, recreate the virtual environment and reinstall from [requirements-m3.txt](requirements-m3.txt).

## 2. Milestone 3 - ML Pipeline Development - Data Ingestion, Validation and Preparation

This milestone converts the project from the solver-centric PoC of Milestone 2 into a reproducible data pipeline built on the same UFLP benchmark defined in Milestone 1. The refactor keeps one shared raw benchmark source, produces structured derived datasets, validates them, versions the pipeline with DVC, and exposes features through Feast.

### 2.1 Schema Definition

Milestone 3 uses explicit schemas for each layer:

- [schema/raw_schema.json](schema/raw_schema.json)
- [schema/processed_schema.json](schema/processed_schema.json)
- [schema/feature_schema.json](schema/feature_schema.json)
- [schema/training_schema.json](schema/training_schema.json)

These schemas define expected files, required columns, numeric and non-negative fields, and identifier keys. They serve as the contracts enforced during validation for the raw, processed, feature, and symbolic-SFT training layers.

The schema layer is important because this project uses OR-Library text files as raw input. The schema files document what must exist after each transformation stage, so later milestones can trust the processed data, feature tables, and symbolic fine-tuning records.

### 2.2 Data Validation and Verification

[pipeline/validate_data.py](pipeline/validate_data.py) implements a pandas-based validation layer with TFDV/Great Expectations-style checks. This project uses OR-Library text artifacts that are parsed into relational UFLP tables, so the validator focuses on project-specific contracts rather than generic tabular profiling alone. It validates:

- raw dataset completeness against the expected OR-Library benchmark files
- required columns and numeric typing
- non-negativity constraints
- duplicate identifier detection
- processed consistency for `m`, `n`, and the complete `m x n` assignment matrix
- feature consistency against processed tables
- symbolic SFT dataset schema, split coverage, and manifest consistency
- value range checks such as normalized fields remaining in `[0, 1]`

It writes:

- [reports/validation/validation_summary.json](reports/validation/validation_summary.json)
- [reports/validation/anomalies.json](reports/validation/anomalies.json)
- [reports/validation/anomalies.txt](reports/validation/anomalies.txt)
- [reports/stats/raw_statistics.json](reports/stats/raw_statistics.json)
- [reports/stats/processed_statistics.json](reports/stats/processed_statistics.json)
- [reports/stats/feature_statistics.json](reports/stats/feature_statistics.json)
- [reports/stats/training_statistics.json](reports/stats/training_statistics.json)

Current result: `status = passed`, `anomaly_count = 0`.

The validation stage verifies not only column presence, but also project-specific consistency. For example, each processed instance must produce the expected number of facility rows, customer rows, and assignment-cost rows. This prevents silent parser errors from reaching Milestone 4 training/evaluation or Milestone 5 production serving. The current anomaly report is empty, so no schema revisions were required for this dataset version.

### 2.3 Data Versioning

Milestone 3 uses DVC through [dvc.yaml](dvc.yaml) and [dvc.lock](dvc.lock). The refactor removed the duplicated milestone-local raw snapshot, switched DVC dependencies to the shared root dataset [../data/raw/](../data/raw/), and moved generated Parquet outputs out of normal Git tracking so DVC can manage the pipeline outputs correctly.

The current DVC stages are:

1. `ingest`
2. `preprocess`
3. `features`
4. `training`
5. `validate`

The symbolic SFT dataset is also generated as a first-class DVC stage in [data/training/](data/training/). It is lightweight and derived directly from the processed/interim layers, then validated together with the raw, processed, and feature layers.

This means the generated data can be reproduced from raw input and code rather than manually edited. The expected final remote step is a DVC remote/push when the final submission storage target is selected.

Current DVC evidence is saved in [reports/m3_pipeline_results.txt](reports/m3_pipeline_results.txt) and [reports/m3_pipeline_results.json](reports/m3_pipeline_results.json). The latest report records `dvc_up_to_date = true` and includes the terminal DAG produced by `python -m dvc dag`.

### 2.4 Setting up a Feature Store

Milestone 3 uses Feast through [feature_repo/](feature_repo/) with:

- entities: `instance_id`, `facility_key`, `customer_key`
- feature views over the generated Parquet files
- local registry and online store for demonstration

[feature_repo/views.py](feature_repo/views.py) now points directly at the generated M3 feature outputs through the shared path contract. Both [feature_repo/apply_repo.py](feature_repo/apply_repo.py) and [feature_repo/run_feature_store_demo.py](feature_repo/run_feature_store_demo.py) run successfully after the refactor, and the feature-store publish step is also included in the ZenML workflow.

The feature store is used to expose engineered instance, facility, and customer features as reusable entities/views. This gives the milestone a real MLOps-style feature-management layer instead of only writing CSV files.

### 2.5 Setup of Data Pipeline within the Larger ML Pipeline / MLOps Platform

Milestone 3 now exposes two orchestration layers:

- a lightweight local runner in [pipeline/run_data_pipeline.py](pipeline/run_data_pipeline.py) and [run_full_workflow.py](run_full_workflow.py)
- an MLOps-platform runner in [pipeline/zenml_pipeline.py](pipeline/zenml_pipeline.py), launched through [run_zenml_workflow.py](run_zenml_workflow.py)

The ZenML pipeline wraps the same refactored stage functions used by the local workflow and executes them in explicit order:

1. ingestion
2. preprocessing
3. feature engineering
4. symbolic SFT dataset build
5. validation
6. feature-store publication

This keeps the milestone reproducible for local development while also satisfying the requirement to integrate the data pipeline into a larger ML pipeline / MLOps platform. The most recent ZenML execution status is written to:

- [reports/zenml_status.json](reports/zenml_status.json)
- [reports/zenml_status.txt](reports/zenml_status.txt)
- [reports/m3_pipeline_results.json](reports/m3_pipeline_results.json)
- [reports/m3_pipeline_results.txt](reports/m3_pipeline_results.txt)

ZenML is the MLOps orchestration layer for this milestone. DVC handles reproducible stage dependencies and data versioning. Feast handles feature-store registration and retrieval. Together, these tools cover the data-preparation side of the project pipeline.

#### 2.5.1 Ingestion of Raw Data and Storage into a Repository

[pipeline/ingest_data.py](pipeline/ingest_data.py) scans the shared benchmark source in [../data/raw/](../data/raw/), excludes `uncapopt.txt`, reads each instance header, checks optimum availability, and writes a structured manifest to:

- [data/interim/dataset_manifest.csv](data/interim/dataset_manifest.csv)
- [data/interim/dataset_manifest.json](data/interim/dataset_manifest.json)

The manifest captures source path, file size, facility count, customer count, optimum availability, and ingestion status for all 15 benchmark instances.

The ingestion stage preserves the Milestone 1 data contract: the canonical raw dataset remains the repository-root [../data/raw/](../data/raw/) folder. M3 does not create a second raw-data source of truth.

#### 2.5.2 Preprocessing and Feature Engineering

[pipeline/preprocess_data.py](pipeline/preprocess_data.py) parses the OR-Library benchmark into normalized relational tables:

- [data/processed/instances.csv](data/processed/instances.csv)
- [data/processed/facilities.csv](data/processed/facilities.csv)
- [data/processed/customers.csv](data/processed/customers.csv)
- [data/processed/assignment_costs.csv](data/processed/assignment_costs.csv)

The refactor fixed the parser so it now handles both benchmark formats used in the dataset, including the larger `capa/capb/capc` files that contain literal `capacity` tokens in facility rows.

[pipeline/engineer_features.py](pipeline/engineer_features.py) then builds:

- instance features: counts, ratios, and cost summaries
- facility features: normalized fixed costs, z-scores, ranks, and assignment-cost summaries
- customer features: assignment-cost statistics and nearest-facility features
- symbolic SFT training assets through [pipeline/build_symbolic_sft_dataset.py](pipeline/build_symbolic_sft_dataset.py), including prompt, target solver code, split labels, and validation metadata for train/validation/test

Current pipeline output counts are:

- 15 ingested benchmark instances
- 15 processed instance rows
- 664 processed facility rows
- 3600 processed customer rows
- 318200 assignment-cost rows
- 15 instance feature rows
- 664 facility feature rows
- 3600 customer feature rows
- 7 symbolic SFT train records
- 4 symbolic SFT validation records
- 4 symbolic SFT test records

These counts are recorded in [reports/validation/validation_summary.json](reports/validation/validation_summary.json) and [data/training/symbolic_sft_manifest.json](data/training/symbolic_sft_manifest.json).

Alignment with previous milestones:

- Uses the same UFLP benchmark and raw data source as M1.
- Feeds the same baseline/LLM verification architecture demonstrated in M2.
- Builds the symbolic SFT dataset needed by the M4 model-development track.

## 3. References

- OR-Library UFLP benchmark dataset, mirrored in [../data/raw/](../data/raw/)
- DVC documentation: https://dvc.org/doc
- Feast documentation: https://docs.feast.dev/
- Google OR-Tools documentation: https://developers.google.com/optimization
