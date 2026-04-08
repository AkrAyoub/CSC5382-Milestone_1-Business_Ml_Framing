## AI-Assisted Symbolic Optimization for Strategic Facility Network Design
## Milestone 4: Model Development and Offline Evaluation


## Table of Contents

- [Quick Links](#quick-links)
- [Overview](#overview)
- [Project Context and Integration with Previous Milestones](#project-context-and-integration-with-previous-milestones)
- [Project Structure and Main Files](#project-structure-and-main-files)
- [4. Milestone 4 - ML Pipeline Development - Model Training and Offline Evaluation](#4-milestone-4---ml-pipeline-development---model-training-and-offline-evaluation)
  - [4.1 Project Structure Definition and Modularity](#41-project-structure-definition-and-modularity)
  - [4.2 Code Versioning](#42-code-versioning)
  - [4.3 Experiment Tracking and Model Versioning](#43-experiment-tracking-and-model-versioning)
  - [4.4 Integration of Model Training and Offline Evaluation into the ML Pipeline / MLOps Platform](#44-integration-of-model-training-and-offline-evaluation-into-the-ml-pipeline--mlops-platform)
  - [4.5 Energy Efficiency Measurement](#45-energy-efficiency-measurement)
- [Visual Outputs](#visual-outputs)
- [Limitations and Next Steps](#limitations-and-next-steps)


## Quick Links

- **[Milestone 4 Report (PDF)](report/report.pdf)**
- **[Milestone 1 README](../Milestone_1-Project_Inception/README.md)**
- **[Milestone 2 README](../Milestone_2-PoC/README.md)**
- **[Milestone 3 README](../Milestone_3-Data_Prep/README.md)**
- **[Main Training Pipeline](src/m4_model_dev/pipelines/training_pipeline.py)**
- **[Model Comparison Pipeline](src/m4_model_dev/pipelines/comparison_pipeline.py)**
- **[ZenML Pipeline](src/m4_model_dev/pipelines/zenml_pipeline.py)**


## Overview

Milestone 4 introduces the model development and offline evaluation layer of the project. While the previous milestones established the business framing, solver logic, and data/MLOps foundation, this milestone adds a supervised machine learning workflow built on top of validated UFLP benchmark data and deterministic optimization outputs.

The implemented task is a facility-level binary classification problem. Each training example represents one `(instance_id, facility_id)` pair. The input is a vector of engineered numeric features derived from Milestone 3, and the target is a binary label `is_open`, indicating whether the facility is open in the deterministic optimal UFLP solution computed from the Milestone 2 baseline solver.

This milestone does not replace the symbolic optimization system described in Milestone 1. Instead, it adds a complementary ML pipeline for offline experimentation, facility ranking, model comparison, and future hybrid optimization support.

The figures, screenshots, and diagrams referenced in the milestone documentation are available in the full report PDF:

- **[Milestone 4 Report (PDF)](report/report.pdf)**


## Project Context and Integration with Previous Milestones

The project now consists of four connected layers:

- **Milestone 1** defined the business problem, research framing, and the LLM-as-modeler architecture.
- **Milestone 2** implemented the deterministic OR-Tools solver baseline and the LLM-generated verified solver pipeline.
- **Milestone 3** implemented the data engineering and MLOps foundation, including ingestion, preprocessing, validation, feature engineering, DVC, and Feast.
- **Milestone 4** implements supervised model training, offline evaluation, experiment tracking, model versioning, visualization, and ZenML pipeline orchestration.

Milestone 4 depends directly on the outputs of the previous milestones:

- raw OR-Library UFLP benchmark files remain the source dataset
- Milestone 3 provides engineered facility-level and instance-level features
- Milestone 2 provides deterministic optimal facility-opening decisions used as labels
- Milestone 4 merges those assets into a supervised dataset, trains and compares models, logs runs in MLflow, orchestrates the workflow with ZenML, and tracks optional energy usage through CodeCarbon


## Project Structure and Main Files

Milestone 4 is organized as a modular ML project.

Main top-level folders:

- [`configs`](configs) - YAML configuration files for training and model comparison
- [`src/m4_model_dev`](src/m4_model_dev) - reusable source code
- [`scripts`](scripts) - command-line entrypoints for training, comparison, tests, and ZenML runs
- [`tests`](tests) - regression checks
- [`data`](data) - generated labels, merged datasets, and split definitions
- [`artifacts`](artifacts) - serialized trained models
- [`reports`](reports) - summaries, metrics, comparisons, confusion reports, figure outputs, ZenML status, and energy reports
- [`mlruns`](mlruns) and [`mlflow.db`](mlflow.db) - MLflow local tracking store and artifacts
- [`.zen`](.zen) - ZenML local configuration

Explanation of the most important code modules:

- [`src/m4_model_dev/paths.py`](src/m4_model_dev/paths.py) - centralizes project paths and runtime directories
- [`src/m4_model_dev/data/build_labels.py`](src/m4_model_dev/data/build_labels.py) - generates facility-open labels from the deterministic baseline solver
- [`src/m4_model_dev/data/prepare_dataset.py`](src/m4_model_dev/data/prepare_dataset.py) - merges Milestone 3 features with Milestone 4 labels
- [`src/m4_model_dev/data/make_splits.py`](src/m4_model_dev/data/make_splits.py) - builds grouped train, validation, and test splits by instance
- [`src/m4_model_dev/models/logistic_numpy.py`](src/m4_model_dev/models/logistic_numpy.py) - custom NumPy logistic regression baseline
- [`src/m4_model_dev/models/model_registry.py`](src/m4_model_dev/models/model_registry.py) - model factory, training, scoring, serialization, and feature importance extraction
- [`src/m4_model_dev/evaluation/metrics.py`](src/m4_model_dev/evaluation/metrics.py) - computes classification metrics and confusion counts
- [`src/m4_model_dev/reporting/training_reports.py`](src/m4_model_dev/reporting/training_reports.py) - writes summaries, manifests, metrics, and training reports
- [`src/m4_model_dev/reporting/comparison_reports.py`](src/m4_model_dev/reporting/comparison_reports.py) - writes comparison summaries and selected-model outputs
- [`src/m4_model_dev/reporting/figures.py`](src/m4_model_dev/reporting/figures.py) - generates PNG visualizations for training and comparison results
- [`src/m4_model_dev/tracking/mlflow_utils.py`](src/m4_model_dev/tracking/mlflow_utils.py) - logs runs, metrics, artifacts, and registered models to MLflow
- [`src/m4_model_dev/tracking/codecarbon_utils.py`](src/m4_model_dev/tracking/codecarbon_utils.py) - activates CodeCarbon tracking when configured
- [`src/m4_model_dev/pipelines/training_pipeline.py`](src/m4_model_dev/pipelines/training_pipeline.py) - full local training and evaluation workflow
- [`src/m4_model_dev/pipelines/comparison_pipeline.py`](src/m4_model_dev/pipelines/comparison_pipeline.py) - multi-model comparison workflow
- [`src/m4_model_dev/pipelines/zenml_pipeline.py`](src/m4_model_dev/pipelines/zenml_pipeline.py) - ZenML pipeline definition with visible step orchestration

Main runnable scripts:

- [`scripts/run_train.py`](scripts/run_train.py)
- [`scripts/run_compare.py`](scripts/run_compare.py)
- [`scripts/run_tests.py`](scripts/run_tests.py)
- [`scripts/run_zenml.py`](scripts/run_zenml.py)

The repository structure figure and folder screenshots are included in:

- **[Milestone 4 Report (PDF)](report/report.pdf)**


## 4. Milestone 4 - ML Pipeline Development - Model Training and Offline Evaluation

### 4.1 Project Structure Definition and Modularity

The milestone follows a modular structure aligned with the principles of Cookiecutter-style data science projects and lifecycle-oriented ML engineering. The implementation avoids a monolithic script and instead separates responsibilities across distinct modules:

- data preparation
- model definition
- evaluation
- reporting
- tracking
- orchestration
- execution scripts
- tests

This modularity improves readability, reuse, maintainability, and experimentation. For example:

- label generation is isolated in [`build_labels.py`](src/m4_model_dev/data/build_labels.py)
- dataset construction is isolated in [`prepare_dataset.py`](src/m4_model_dev/data/prepare_dataset.py)
- split logic is isolated in [`make_splits.py`](src/m4_model_dev/data/make_splits.py)
- training orchestration is isolated in [`training_pipeline.py`](src/m4_model_dev/pipelines/training_pipeline.py)
- visual output generation is isolated in [`figures.py`](src/m4_model_dev/reporting/figures.py)
- experiment logging is isolated in [`mlflow_utils.py`](src/m4_model_dev/tracking/mlflow_utils.py)


From an execution standpoint, Milestone 4 can be understood as a pipeline of reusable stages:

1. generate facility labels from the deterministic solver
2. merge labels with engineered features
3. create grouped splits
4. train the configured model
5. evaluate offline on train, validation, and test
6. write reports and PNG visualizations
7. log experiments to MLflow
8. optionally orchestrate all stages through ZenML


### 4.2 Code Versioning

Milestone 4 adopts a Git-based workflow compatible with GitHub Flow. The versioning process is documented in [`docs/versioning_and_delivery.md`](docs/versioning_and_delivery.md) and is designed around short-lived feature branches, local validation before merge, and controlled commit scope.

The intended flow is:

1. create a short-lived feature branch from the default branch
2. implement one scoped change
3. run the verification scripts locally
4. commit only source code, configuration, and lightweight report files
5. push the branch and open a pull request
6. merge only after validation succeeds

This workflow was also reflected in practice during implementation through branch-based milestone cleanup and structural refactoring. The purpose of this versioning discipline is to keep changes auditable and reduce the risk of mixing runtime artifacts with source code changes.

Validation scripts used before merge include:

- [`scripts/run_train.py`](scripts/run_train.py)
- [`scripts/run_compare.py`](scripts/run_compare.py)
- [`scripts/run_tests.py`](scripts/run_tests.py)
- [`scripts/run_zenml.py`](scripts/run_zenml.py)



### 4.3 Experiment Tracking and Model Versioning

Milestone 4 integrates MLflow for experiment tracking and local model versioning.

The MLflow integration is implemented in [`src/m4_model_dev/tracking/mlflow_utils.py`](src/m4_model_dev/tracking/mlflow_utils.py). It provides the following capabilities:

- configuration of the local tracking URI
- creation of experiments when missing
- logging of run parameters
- logging of split metrics
- logging of supporting artifacts such as summaries and PNG figures
- logging of sklearn model artifacts
- registration of the selected model in a local model registry

The local MLflow store consists of:

- [`mlflow.db`](mlflow.db) - tracking database
- [`mlruns/`](mlruns) - artifact storage

Two main experiment types are logged:

- single-model training runs
- multi-model comparison runs

The currently supported model families are:

- `dummy_prior`
- `logreg_numpy`
- `logreg_sklearn`
- `random_forest`
- `hist_gradient_boosting`

Model selection is based on validation metrics only, which prevents leakage from the test split into selection decisions. The current selected model is `random_forest`. The local registered model name is:

- `m4-facility-opening-best`

Latest local results:

- validation F1: `0.6349`
- validation ROC-AUC: `0.8944`
- test F1: `0.3571`
- test ROC-AUC: `0.7849`

Milestone 4 also uses MLflow as a visualization layer. The local UI can be launched and inspected through the MLflow web interface at:

- `http://127.0.0.1:5004`

Within the MLflow UI, the following are visible:

- experiment names
- run parameters
- metrics per split
- artifact files
- registered model versions
- attached training and comparison figures

The MLflow screenshots used in the milestone documentation are included in:

- **[Milestone 4 Report (PDF)](report/report.pdf)**


### 4.4 Integration of Model Training and Offline Evaluation into the ML Pipeline / MLOps Platform

Milestone 4 integrates the training and evaluation workflow into a local MLOps pipeline using ZenML.

The ZenML integration is implemented in [`src/m4_model_dev/pipelines/zenml_pipeline.py`](src/m4_model_dev/pipelines/zenml_pipeline.py). Rather than wrapping the entire workflow in a single opaque step, the pipeline is split into visible stages:

- `load_config_step`
- `prepare_data_step`
- `train_model_step`
- `evaluate_model_step`
- `persist_outputs_step`

This decomposition makes the pipeline easier to inspect and debug in the ZenML dashboard and aligns better with standard MLOps practice.

The ZenML pipeline performs:

1. configuration loading
2. label generation and dataset preparation
3. split generation
4. model fitting
5. offline evaluation
6. report writing
7. MLflow logging
8. artifact persistence

ZenML status is also written locally in:

- [`reports/zenml_status.txt`](reports/zenml_status.txt)
- [`reports/zenml_status.json`](reports/zenml_status.json)

The local ZenML dashboard is available at:

- `http://127.0.0.1:8238`

This UI provides visual access to pipeline runs and step execution. It complements MLflow by focusing on orchestration rather than experiment history.

The logical execution flow of the project is therefore:

- [`run_train.py`](scripts/run_train.py) -> [`training_pipeline.py`](src/m4_model_dev/pipelines/training_pipeline.py) -> data preparation -> model training -> evaluation -> reports -> MLflow
- [`run_compare.py`](scripts/run_compare.py) -> [`comparison_pipeline.py`](src/m4_model_dev/pipelines/comparison_pipeline.py) -> shared training inputs -> train all candidate models -> rank on validation -> reports -> MLflow
- [`run_zenml.py`](scripts/run_zenml.py) -> [`zenml_pipeline.py`](src/m4_model_dev/pipelines/zenml_pipeline.py) -> same workflow orchestrated as explicit ZenML steps


The ZenML dashboard screenshots and runtime diagrams are included in:

- **[Milestone 4 Report (PDF)](report/report.pdf)**


### 4.5 Energy Efficiency Measurement

As an optional bonus feature, Milestone 4 integrates CodeCarbon for energy-efficiency measurement during model training.

The integration is activated from the training workflow through [`src/m4_model_dev/tracking/codecarbon_utils.py`](src/m4_model_dev/tracking/codecarbon_utils.py). When enabled in configuration, the pipeline starts a CodeCarbon tracker before model fitting and stops it after training.

The output is written to:

- [`reports/emissions.csv`](reports/emissions.csv)

The report contains estimated energy consumption statistics for CPU, RAM, and GPU activity during training.

This milestone also benefits from the CodeCarbon console output during execution, which helps validate that tracking is active. On the local Windows environment, CodeCarbon may fall back to CPU estimation mode when direct power measurement tools are unavailable. This is an implementation limitation of the environment rather than of the pipeline itself.

The CodeCarbon screenshots included in the documentation are available in:

- **[Milestone 4 Report (PDF)](report/report.pdf)**


## Visual Outputs

Milestone 4 generates visible PNG outputs automatically whenever the training or comparison workflows are executed.

Training visual outputs:

- [`reports/training_metrics.png`](reports/training_metrics.png)
- [`reports/training_confusion_matrices.png`](reports/training_confusion_matrices.png)
- [`reports/feature_importance.png`](reports/feature_importance.png)
- [`reports/training_dashboard.png`](reports/training_dashboard.png)

Comparison visual outputs:

- [`reports/comparison_validation_metrics.png`](reports/comparison_validation_metrics.png)
- [`reports/comparison_test_metrics.png`](reports/comparison_test_metrics.png)
- [`reports/comparison_dashboard.png`](reports/comparison_dashboard.png)

These figures are also logged as MLflow artifacts during tracked runs.


## Limitations and Next Steps

Current limitations:

- the predictive model is an auxiliary supervised layer and does not replace the symbolic solver
- the benchmark remains relatively small, with only 15 UFLP instances
- ZenML on Windows still produces environment-specific warnings, including the default pickle materializer warning for the custom scaler object
- CodeCarbon CPU measurement may use fallback estimation on the local machine
- the ML workflow currently focuses on tabular supervised models and does not include LLM fine-tuning

Possible future work:

- add more model families or tuning strategies
- integrate richer hyperparameter search
- add custom ZenML materializers for cleaner artifact handling
- extend the ML layer toward solver warm-starting or candidate facility screening
- connect the ML layer more explicitly to the LLM-generated modeling pipeline from Milestone 2
- package the milestone outputs more tightly for cloud or CI execution
