## AI-Assisted Symbolic Optimization for Strategic Facility Network Design
## Milestone 4: ML Pipeline Development - Model Training and Offline Evaluation


## Table of Contents

- [Overview](#overview)
- [Technical Scope of the Milestone](#technical-scope-of-the-milestone)
- [Project Context and Integration with Previous Milestones](#project-context-and-integration-with-previous-milestones)
- [4.1 Project Structure Definition and Modularity](#41-project-structure-definition-and-modularity)
- [4.2 Code Versioning](#42-code-versioning)
- [4.3 Experiment Tracking and Model Versioning](#43-experiment-tracking-and-model-versioning)
- [4.4 Integration of Model Training and Offline Evaluation into the ML Pipeline / MLOps Platform](#44-integration-of-model-training-and-offline-evaluation-into-the-ml-pipeline--mlops-platform)
- [4.5 Energy Efficiency Measurement](#45-energy-efficiency-measurement)
- [Technical Workflow and Execution](#technical-workflow-and-execution)
- [Offline Evaluation Setup and Results](#offline-evaluation-setup-and-results)
- [Produced Artifacts](#produced-artifacts)
- [Conclusion](#conclusion)


## Overview

This milestone implements the model development and offline evaluation layer of the project.

Milestone 4 extends the workflow established in the previous milestones by introducing supervised machine learning models trained on solver-derived outcomes. The milestone uses the deterministic optimization outputs produced earlier in the project as labels and combines them with the validated feature layer created in Milestone 3.

The predictive task implemented in this milestone is a binary classification problem:

- input: facility-level and instance-level engineered features
- output: a binary variable indicating whether a facility is open in the deterministic optimal UFLP solution

This predictive layer does not replace the symbolic optimization system. Instead, it provides an additional machine learning workflow for offline experimentation on solver-derived decisions and supports future work on facility screening, heuristic guidance, and scenario analysis.


## Technical Scope of the Milestone

From a technical standpoint, this milestone focuses on making the model-development layer operational, repeatable, and testable.

The implemented task is binary classification at the facility level:

- input: one row per `(instance_id, facility_id)` with engineered numeric features
- target: `is_open`
  - `1` when the facility is open in the deterministic optimal UFLP solution
  - `0` otherwise

The implemented technical workflow includes:

- facility-opening label generation from the deterministic baseline solver
- merged training dataset generation
- grouped train, validation, and test splitting by `instance_id`
- configurable single-model training
- multi-model comparison
- offline evaluation with confusion counts and threshold selection
- feature-importance export for tree-based models
- MLflow tracking and local model registry
- CodeCarbon energy tracking
- ZenML pipeline integration

The currently supported model families are:

- `dummy_prior`
- `logreg_numpy`
- `logreg_sklearn`
- `random_forest`
- `hist_gradient_boosting`

The current default training configuration targets `random_forest` through `configs/train_best_model.yaml`. Model selection is based on validation metrics only, while the test split is reserved for final offline evaluation.


## Project Context and Integration with Previous Milestones

The complete project now evolves across four connected milestone stages:

- Milestone 1 defined the business case, research framing, and the LLM-as-modeler architecture for AI-assisted symbolic optimization.
- Milestone 2 implemented the deterministic OR-Tools baseline and the LLM-generated verified solver workflow.
- Milestone 3 implemented the data layer, including ingestion, preprocessing, feature engineering, validation, DVC reproducibility, and Feast integration.
- Milestone 4 adds supervised model training, offline evaluation, experiment tracking, model versioning, and ML pipeline orchestration.

In practical terms, Milestone 4 is built on the outputs of the earlier milestones as follows:

- raw OR-Library benchmark instances are processed by the Milestone 3 data pipeline
- deterministic facility decisions from Milestone 2 are used to generate supervised labels
- Milestone 3 engineered features are merged with Milestone 4 labels
- machine learning models are trained and evaluated using grouped train, validation, and test splits
- MLflow and ZenML are used to track and orchestrate the model development workflow


## 4.1 Project Structure Definition and Modularity

Milestone 4 follows a modular project layout in order to separate configuration, reusable source code, generated data, reports, runnable scripts, and tests.

The milestone is organized into the following components:

- `configs`
  - configuration files for training and model comparison
- `src/m4_model_dev`
  - reusable source modules for label generation, dataset preparation, split creation, model fitting, evaluation, tracking, and pipeline orchestration
- `scripts`
  - executable entrypoints for training, comparison, testing, and ZenML runs
- `data`
  - generated labels, merged datasets, and split definitions
- `artifacts`
  - saved trained model files
- `reports`
  - offline evaluation summaries, confusion reports, model selection outputs, ZenML status, and emissions reports
- `tests`
  - regression tests for selection rules and configuration consistency

This modular layout keeps the workflow decomposed into distinct and reusable stages:

- label construction
- dataset preparation
- split generation
- model training
- offline evaluation
- experiment logging
- pipeline orchestration


## 4.2 Code Versioning

Milestone 4 adopts a repository workflow aligned with Git and GitHub Flow.

The implementation is structured so that milestone changes can be introduced through isolated code updates, validated locally, and then merged into the main project line after successful testing. The intended workflow is:

1. create a branch for a modeling, tracking, or orchestration change
2. implement the change in the milestone modules and configuration files
3. rerun the milestone scripts and regression tests
4. merge the validated change into the main branch

This versioning approach supports:

- controlled experimentation
- traceability of training and evaluation updates
- reproducible configuration changes
- safer integration of new model families or tracking logic


## 4.3 Experiment Tracking and Model Versioning

Milestone 4 integrates MLflow for experiment tracking and local model versioning.

MLflow is used to record:

- run metadata
- selected model family
- configuration values
- validation and test metrics
- selected classification threshold
- summary artifacts
- registered model versions

The implemented workflow supports both single-model training runs and multi-model comparison runs. The selected trained model is stored in the local registry under a stable registered model name:

- `m4-facility-opening-best`

The models currently implemented in the milestone are:

- dummy prior baseline
- custom NumPy logistic regression
- scikit-learn logistic regression
- random forest
- histogram-based gradient boosting

This experiment-tracking layer makes it possible to:

- compare multiple runs consistently
- retain local model history
- register updated model versions
- preserve a reproducible record of offline evaluation outputs


## 4.4 Integration of Model Training and Offline Evaluation into the ML Pipeline / MLOps Platform

Milestone 4 integrates the training and offline evaluation logic into a local MLOps workflow using ZenML.

The pipeline orchestrates the following stages:

1. generate or refresh facility-opening labels
2. build the merged supervised dataset
3. create grouped train, validation, and test splits
4. fit the configured machine learning model
5. select a classification threshold using validation results only
6. evaluate the trained model on train, validation, and test splits
7. write reports and artifacts
8. log the run to MLflow

This pipeline integration provides a single entrypoint for end-to-end model development and evaluation while preserving the modular internal structure of the milestone.

The MLOps layer therefore includes:

- configuration-driven training
- repeatable offline evaluation
- experiment logging
- local model registration
- local pipeline orchestration


## 4.5 Energy Efficiency Measurement

As an optional extension, Milestone 4 includes energy-efficiency tracking through CodeCarbon.

During training runs, CodeCarbon records estimated resource consumption and writes an emissions report. This adds an operational measurement layer alongside the standard predictive performance metrics.

On the current local Windows environment, CPU measurement may rely on fallback estimation when direct hardware power measurement is unavailable. Even with this limitation, the milestone still produces a usable record of training-time energy estimates.


## Technical Workflow and Execution

The milestone includes runnable scripts for training, comparison, testing, and pipeline execution. This technical layer is important because the milestone is not limited to reporting model results; it also implements the operational workflow required to run, debug, and validate the model-development process.

Main execution entrypoints:

- train the default selected model
- train an alternative baseline configuration
- compare all implemented model families
- run regression tests for selection and configuration consistency
- execute the ZenML pipeline

Typical commands used during milestone execution are:

```powershell
& 'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0\python3.13.exe' scripts\run_train.py
```

```powershell
& 'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0\python3.13.exe' scripts\run_train.py configs\train_logreg.yaml
```

```powershell
& 'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0\python3.13.exe' scripts\run_compare.py
```

```powershell
& 'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0\python3.13.exe' scripts\run_tests.py
```

```powershell
& 'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0\python3.13.exe' scripts\run_zenml.py
```

The technical workflow also includes explicit debugging and validation support through:

- regression tests for model ordering and configuration alignment
- validation-only threshold selection
- confusion-count reporting
- MLflow run summaries and artifact logging
- ZenML status reporting

This makes the milestone suitable not only for reporting offline model results, but also for reproducing, debugging, and maintaining the training pipeline as a complete local workflow.


## Offline Evaluation Setup and Results

The dataset is split by grouped benchmark instances rather than by random row-level shuffling. This design reduces leakage across facilities that belong to the same UFLP instance and keeps evaluation aligned with the structure of the benchmark collection.

The offline evaluation workflow uses:

- train split for model fitting
- validation split for threshold selection and model comparison
- test split for final held-out reporting

The reported offline metrics include:

- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- log loss

Model comparison ranks candidates using validation performance only. Based on the current repository outputs, the selected model is the random forest classifier.

Latest selected-model results:

- validation F1: 0.6349
- validation ROC-AUC: 0.8944
- test F1: 0.3571
- test ROC-AUC: 0.7849

These values should be interpreted as offline benchmark results on solver-derived labels rather than as a replacement for the deterministic optimization solution itself.


## Produced Artifacts

Milestone 4 produces the following main artifact families.

Data outputs:

- facility-opening labels
- merged supervised dataset
- grouped split definitions

Model outputs:

- trained model artifact for the selected model
- feature importance report when supported by the model

Evaluation outputs:

- metrics report
- training summary
- confusion report
- model comparison summary
- model selection summary

Tracking and orchestration outputs:

- MLflow run history and local model registry entries
- ZenML pipeline status report
- CodeCarbon emissions report

Additional repository outputs used during execution and debugging include:

- `mlruns`
- `mlflow.db`
- `data/labels`
- `data/merged`
- `data/splits`
- `artifacts`
- `reports`


## Conclusion

Milestone 4 extends the project from symbolic optimization and validated data preparation into supervised model development and offline evaluation.

The milestone satisfies the core implementation requirements through:

- modular project structure
- versioning workflow aligned with Git and GitHub Flow
- MLflow-based experiment tracking and model versioning
- ZenML-based pipeline orchestration
- optional energy-efficiency reporting with CodeCarbon

As a result, the project now includes four connected layers:

- problem framing and research positioning
- deterministic and LLM-assisted solver workflows
- validated and reproducible data preparation
- supervised model training, experiment tracking, and pipeline orchestration

This establishes a complete local benchmark workflow for future experimentation at the intersection of symbolic optimization, data engineering, and MLOps.
