## AI-Assisted Symbolic Optimization for Strategic Facility Network Design
## Milestone 3: Data Acquisition, Validation, Preparation, and Feature Store


## Table of Contents

- [Overview](#overview)
- [Quick Links](#quick-links)
- [Project Context and Integration with Previous Milestones](#project-context-and-integration-with-previous-milestones)
- [Repository Structure (Milestone 3)](#repository-structure-milestone-3)
- [Dataset (OR-Library UFLP)](#dataset-or-library-uflp)
- [Highlights](#highlights)
- [Installation and Setup](#installation-and-setup)
- [Running the Data Pipeline](#running-the-data-pipeline)
  - [Stage A - Raw Data Ingestion](#stage-a---raw-data-ingestion)
  - [Stage B - Data Preprocessing and Transformation](#stage-b---data-preprocessing-and-transformation)
  - [Stage C - Feature Engineering](#stage-c---feature-engineering)
  - [Stage D - Validation and Anomaly Analysis](#stage-d---validation-and-anomaly-analysis)
  - [Stage E - Feature Store Setup (Feast)](#stage-e---feature-store-setup-feast)
- [DVC Pipeline and Reproducibility](#dvc-pipeline-and-reproducibility)
- [Produced Artifacts](#produced-artifacts)
- [Results Summary](#results-summary)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Troubleshooting](#troubleshooting)


## Quick Links

- **[Milestone 3 Report (PDF)](report/report.pdf)**
- **[DVC Pipeline Definition](dvc.yaml)**
- **[Full Workflow Runner](run_full_workflow.py)**
- **[Data Pipeline Runner](pipeline/run_data_pipeline.py)**
- **[Feature Store Demo](feature_repo/run_feature_store_demo.py)**
- **[Milestone 1 README](../Milestone_1-Project_Inception/README.md)**
- **[Milestone 2 README](../Milestone_2-PoC/README.md)**


## Overview

This milestone implements the **data and MLOps layer** for the **Uncapacitated Facility Location Problem (UFLP)** project.

While Milestone 1 established the research framing and Milestone 2 implemented the solver pipeline, Milestone 3 focuses on everything required to make the system operationally reliable:

- structured raw data ingestion
- reproducible preprocessing
- feature engineering for analytics and ML-oriented experimentation
- schema-based validation and anomaly analysis
- dataset versioning with DVC
- feature-store integration with Feast

The outcome is a transition from a solver-centric prototype into a **data-centric optimization workflow**, where downstream symbolic optimization components can rely on validated, versioned, and reusable data assets.


## Project Context and Integration with Previous Milestones

The full project evolves across three connected milestones:

- **Milestone 1 - Problem Framing:** defined UFLP as an AI-assisted symbolic optimization problem and introduced the LLM-as-modeler architecture.
- **Milestone 2 - Solver Layer:** implemented a deterministic OR-Tools baseline and an LLM-generated verified solver pipeline.
- **Milestone 3 - Data and MLOps Layer:** adds ingestion, preprocessing, validation, versioning, and feature management so that the solver stack operates on trustworthy data.

In practical terms, this milestone ensures that:

- raw OR-Library benchmark files are cataloged and auditable
- processed tables are structured for analysis and reuse
- engineered features are available for experimentation
- validation catches structural and statistical issues before solver use
- data lineage is reproducible through DVC
- feature retrieval is centralized through Feast


## Repository Structure (Milestone 3)

- **Pipeline**
  - [`pipeline/ingest_data.py`](pipeline/ingest_data.py) - discovers raw instances, parses header metadata, and builds the dataset manifest
  - [`pipeline/preprocess_data.py`](pipeline/preprocess_data.py) - parses OR-Library instances into structured relational tables
  - [`pipeline/engineer_features.py`](pipeline/engineer_features.py) - computes instance-, facility-, and customer-level features
  - [`pipeline/validate_data.py`](pipeline/validate_data.py) - validates raw, processed, and feature layers against schemas and consistency rules
  - [`pipeline/run_data_pipeline.py`](pipeline/run_data_pipeline.py) - runs the full Milestone 3 data pipeline end-to-end
- **Feature Store**
  - [`feature_repo/entities.py`](feature_repo/entities.py) - Feast entities: `instance_id`, `facility_key`, `customer_key`
  - [`feature_repo/views.py`](feature_repo/views.py) - Feast feature views backed by Parquet feature files
  - [`feature_repo/apply_repo.py`](feature_repo/apply_repo.py) - applies Feast definitions
  - [`feature_repo/run_feature_store_demo.py`](feature_repo/run_feature_store_demo.py) - historical feature retrieval demo
  - [`feature_repo/feature_store.yaml`](feature_repo/feature_store.yaml) - local Feast configuration
- **Schemas**
  - [`schema/raw_schema.json`](schema/raw_schema.json)
  - [`schema/processed_schema.json`](schema/processed_schema.json)
  - [`schema/feature_schema.json`](schema/feature_schema.json)
- **Workflow / Reproducibility**
  - [`dvc.yaml`](dvc.yaml) - DVC stage definition
  - [`dvc.lock`](dvc.lock) - locked dependencies and outputs
  - [`run_full_workflow.py`](run_full_workflow.py) - runs data pipeline, applies Feast repo, and executes demo retrieval
- **Data**
  - [`data/raw/`](data/raw/) - OR-Library instance files and `uncapopt.txt`
  - [`data/interim/`](data/interim/) - ingestion manifests
  - [`data/processed/`](data/processed/) - normalized tabular outputs
  - [`data/features/`](data/features/) - engineered features in CSV, JSON, and Parquet
- **Reports**
  - [`reports/validation/`](reports/validation/) - validation summaries and anomaly outputs
  - [`reports/stats/`](reports/stats/) - descriptive statistics for each pipeline layer
- **Documentation**
  - [`report/report.pdf`](report/report.pdf) - milestone report
  - [`requirements-m3.txt`](requirements-m3.txt) - Python dependencies for Milestone 3


## Dataset (OR-Library UFLP)

This milestone uses the **OR-Library UFLP benchmark collection**, the same benchmark family used throughout the project.

- **Raw data directory:** [`data/raw/`](data/raw/)
- **Known optima file:** [`data/raw/uncapopt.txt`](data/raw/uncapopt.txt)

Supported benchmark instances:

- `cap71.txt` to `cap74.txt`
- `cap101.txt` to `cap104.txt`
- `cap131.txt` to `cap134.txt`
- `capa.txt`, `capb.txt`, `capc.txt`

Each instance provides:

- number of facilities `m`
- number of customers `n`
- facility opening costs
- customer-to-facility assignment costs

For the uncapacitated setting, capacity and demand tokens from the original OR-Library format are ignored during parsing, while fixed costs and assignment costs are preserved as the effective optimization input.


## Highlights

### Structured Raw Data Ingestion

- discovers all benchmark instance files automatically
- parses key metadata (`m`, `n`, file size, source, optimum availability)
- produces auditable manifest files in CSV and JSON

### Relational Data Preparation

- converts raw text instances into structured tables
- separates the dataset into `instances`, `facilities`, `customers`, and `assignment_costs`
- preserves exact benchmark information for reproducible downstream use

### Feature Engineering for Reuse

- computes statistical, normalized, and derived features
- supports instance-level, facility-level, and customer-level analysis
- exports features to Parquet for feature-store consumption

### Validation and Data Quality Controls

- checks required columns, numeric fields, non-negativity constraints, and ID uniqueness
- validates cross-table consistency such as `m`, `n`, and full assignment-matrix completeness
- generates anomaly logs and descriptive statistics automatically

### Reproducibility and MLOps Readiness

- DVC defines a multi-stage reproducible pipeline
- Feast exposes engineered features through a local feature store
- outputs are organized for repeated experimentation and future solver integration


## Installation and Setup

### 1. Create a Virtual Environment

**Windows (PowerShell)**

```bash
python -m venv .venv
```

### 2. Activate the Environment

```bash
.venv\Scripts\Activate.ps1
```

### 3. Install Milestone 3 Dependencies

```bash
pip install -r requirements-m3.txt
```

Dependencies include:

- `pandas`
- `pyarrow`
- `feast`
- `dvc`

### 4. Work from the Milestone Directory

```bash
cd Milestone_3-Data_Prep
```


## Running the Data Pipeline

### Full Milestone 3 Workflow

This runs:

1. the full data pipeline,
2. Feast repository application,
3. a sample feature retrieval demo.

```bash
python run_full_workflow.py
```

### Data Pipeline Only

```bash
python pipeline/run_data_pipeline.py
```


### Stage A - Raw Data Ingestion

```bash
python pipeline/ingest_data.py
```

What this stage does:

- scans [`data/raw/`](data/raw/) for OR-Library instance files
- excludes non-instance companion files such as `uncapopt.txt`
- reads the leading `m n` pair from each file
- checks whether each instance has a known optimum entry
- writes:
  - [`data/interim/dataset_manifest.csv`](data/interim/dataset_manifest.csv)
  - [`data/interim/dataset_manifest.json`](data/interim/dataset_manifest.json)

Manifest fields include:

- `instance_id`
- `file_name`
- `file_path`
- `source`
- `file_size_bytes`
- `facility_count_m`
- `customer_count_n`
- `has_known_optimum`
- `ingestion_status`


### Stage B - Data Preprocessing and Transformation

```bash
python pipeline/preprocess_data.py
```

This stage parses each OR-Library instance into normalized tables suitable for analytics and downstream modeling.

Produced outputs:

- [`data/processed/instances.csv`](data/processed/instances.csv)
- [`data/processed/facilities.csv`](data/processed/facilities.csv)
- [`data/processed/customers.csv`](data/processed/customers.csv)
- [`data/processed/assignment_costs.csv`](data/processed/assignment_costs.csv)
- JSON mirrors of all four datasets in [`data/processed/`](data/processed/)

Core parsing logic:

- ignores irrelevant capacity and demand labels from the capacitated source format
- extracts facility fixed costs correctly
- reconstructs the full customer-facility assignment cost matrix
- builds relational tables with stable identifiers

Processed tables:

| Table | Description |
|---|---|
| `instances` | instance-level metadata and aggregate fixed-cost statistics |
| `facilities` | one row per facility with fixed opening cost |
| `customers` | one row per customer with assignment-cost summaries |
| `assignment_costs` | full customer-facility cost matrix |


### Stage C - Feature Engineering

```bash
python pipeline/engineer_features.py
```

This stage transforms processed tables into feature datasets for analysis, benchmarking, and future ML-assisted extensions.

Produced outputs:

- [`data/features/instance_features.csv`](data/features/instance_features.csv)
- [`data/features/facility_features.csv`](data/features/facility_features.csv)
- [`data/features/customer_features.csv`](data/features/customer_features.csv)
- Parquet mirrors for all three datasets in [`data/features/`](data/features/)
- JSON mirrors for all three datasets in [`data/features/`](data/features/)

Feature categories:

- **Instance-level**
  - `facility_count_m`
  - `customer_count_n`
  - `facility_customer_ratio`
  - `avg_fixed_cost`
  - `avg_assignment_cost`
  - `fixed_cost_range`
  - `assignment_cost_range`
- **Facility-level**
  - `facility_key`
  - `normalized_fixed_cost_minmax`
  - `fixed_cost_zscore`
  - `fixed_cost_rank_ascending`
  - `avg_assignment_cost_from_facility`
- **Customer-level**
  - `customer_key`
  - `std_assignment_cost`
  - `assignment_cost_range`
  - `nearest_facility_id`
  - `nearest_facility_cost`

Implementation notes:

- synthetic keys are generated as `instance_id__facility_id` and `instance_id__customer_id`
- a fixed event timestamp is used to keep Feast joins deterministic for this benchmark dataset
- outputs are written in CSV, JSON, and Parquet for both inspection and feature-store compatibility


### Stage D - Validation and Anomaly Analysis

```bash
python pipeline/validate_data.py
```

The validation layer enforces explicit contracts over the raw, processed, and feature datasets.

Schemas used:

- [`schema/raw_schema.json`](schema/raw_schema.json)
- [`schema/processed_schema.json`](schema/processed_schema.json)
- [`schema/feature_schema.json`](schema/feature_schema.json)

Validation logic includes:

- required-column checks
- numeric-column checks
- non-negativity checks
- duplicate ID detection
- raw dataset completeness checks
- processed row consistency (`m`, `n`, and `m x n` assignment matrix)
- feature row consistency against processed tables
- range checks such as `normalized_fixed_cost_minmax in [0,1]`

Produced outputs:

- [`reports/validation/validation_summary.json`](reports/validation/validation_summary.json)
- [`reports/validation/anomalies.json`](reports/validation/anomalies.json)
- [`reports/validation/anomalies.txt`](reports/validation/anomalies.txt)
- [`reports/stats/raw_statistics.json`](reports/stats/raw_statistics.json)
- [`reports/stats/processed_statistics.json`](reports/stats/processed_statistics.json)
- [`reports/stats/feature_statistics.json`](reports/stats/feature_statistics.json)


### Stage E - Feature Store Setup (Feast)

Apply the feature repository:

```bash
python feature_repo/apply_repo.py
```

Run the retrieval demo:

```bash
python feature_repo/run_feature_store_demo.py
```

Feast configuration:

- provider: local
- registry: SQLite-backed local registry
- online store: SQLite
- offline source: generated Parquet feature files

Defined entities:

- `instance_id`
- `facility_key`
- `customer_key`

Defined feature views:

- `instance_features_view`
- `facility_features_view`
- `customer_features_view`

The retrieval demo fetches example instance-level features such as:

- `facility_count_m`
- `customer_count_n`
- `avg_fixed_cost`
- `avg_assignment_cost`


## DVC Pipeline and Reproducibility

Milestone 3 uses **Data Version Control (DVC)** to track data artifacts and execution dependencies.

- **Pipeline file:** [`dvc.yaml`](dvc.yaml)
- **Locked outputs:** [`dvc.lock`](dvc.lock)
- **Raw data tracking:** [`data/raw.dvc`](data/raw.dvc)

Defined DVC stages:

1. `ingest`
2. `preprocess`
3. `features`
4. `validate`

Typical usage:

```bash
dvc repro
```

Tracked artifact families include:

- raw data
- interim manifests
- processed datasets
- engineered features
- validation outputs
- statistics reports

Windows note:

- due to long path handling on some Windows setups, DVC path resolution may require a drive-mapping workaround such as `subst X: <repo-path>` before running the pipeline


## Produced Artifacts

### Raw and Interim

- [`data/raw/`](data/raw/)
- [`data/raw/uncapopt.txt`](data/raw/uncapopt.txt)
- [`data/interim/dataset_manifest.csv`](data/interim/dataset_manifest.csv)
- [`data/interim/dataset_manifest.json`](data/interim/dataset_manifest.json)

### Processed Data

- [`data/processed/instances.csv`](data/processed/instances.csv)
- [`data/processed/facilities.csv`](data/processed/facilities.csv)
- [`data/processed/customers.csv`](data/processed/customers.csv)
- [`data/processed/assignment_costs.csv`](data/processed/assignment_costs.csv)

### Engineered Features

- [`data/features/instance_features.csv`](data/features/instance_features.csv)
- [`data/features/facility_features.csv`](data/features/facility_features.csv)
- [`data/features/customer_features.csv`](data/features/customer_features.csv)
- [`data/features/instance_features.parquet`](data/features/instance_features.parquet)
- [`data/features/facility_features.parquet`](data/features/facility_features.parquet)
- [`data/features/customer_features.parquet`](data/features/customer_features.parquet)

### Validation and Statistics

- [`reports/validation/validation_summary.json`](reports/validation/validation_summary.json)
- [`reports/validation/anomalies.txt`](reports/validation/anomalies.txt)
- [`reports/stats/raw_statistics.json`](reports/stats/raw_statistics.json)
- [`reports/stats/processed_statistics.json`](reports/stats/processed_statistics.json)
- [`reports/stats/feature_statistics.json`](reports/stats/feature_statistics.json)


## Results Summary

The generated pipeline outputs in this repository show the following milestone-scale results:

| Output | Result |
|---|---:|
| Raw benchmark instances cataloged | 15 |
| Facilities parsed | 664 |
| Customers parsed | 3600 |
| Assignment cost rows generated | 318200 |
| Instance feature rows | 15 |
| Facility feature rows | 664 |
| Customer feature rows | 3600 |
| Validation anomalies detected | 0 |

Validation status from [`reports/validation/validation_summary.json`](reports/validation/validation_summary.json):

- `status = passed`
- `anomaly_count = 0`

This confirms that the milestone successfully produces:

- a complete dataset manifest
- a normalized processed data layer
- reusable feature datasets
- schema-compliant and statistically summarized outputs
- a Feast feature-store configuration over generated Parquet files


## Limitations and Next Steps

### Current Limitations

- event timestamps are synthetic and fixed for deterministic joins
- the dataset is static rather than streaming
- DVC path handling may be awkward on Windows with long directory names
- the feature store is local-only in this milestone configuration

### Future Work

- integrate real-time or incremental ingestion
- support dynamic feature refresh and materialization workflows
- connect feature retrieval directly to solver-facing experimentation
- add automated retriggering of validation when raw inputs change
- extend the MLOps layer toward cloud deployment and scheduled orchestration


## Troubleshooting

- **No raw instances found**
  - ensure benchmark files exist in [`data/raw/`](data/raw/)

- **Validation reports missing**
  - rerun:
  ```bash
  python pipeline/run_data_pipeline.py
  ```

- **Feast demo fails**
  - verify dependencies are installed from [`requirements-m3.txt`](requirements-m3.txt)
  - ensure feature Parquet files were generated in [`data/features/`](data/features/)
  - apply the repo first with:
  ```bash
  python feature_repo/apply_repo.py
  ```

- **DVC path issues on Windows**
  - use a shorter working path or a mapped drive before running `dvc repro`
