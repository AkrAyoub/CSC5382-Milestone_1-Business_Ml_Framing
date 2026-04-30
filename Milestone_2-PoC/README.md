# Milestone 2 - Development of Proof-of-Concepts

This milestone implements a proof of concept for the Milestone 1 project direction: using an off-the-shelf LLM as a symbolic modeler for the Uncapacitated Facility Location Problem (UFLP), while keeping a deterministic CBC solver as the trusted reference and verifier.

### Table of Contents

- [1. Setup, Usage, and Demo Guide](#1-setup-usage-and-demo-guide)
  - [1.1 Repository Structure](#11-repository-structure)
  - [1.2 Installation and Setup](#12-installation-and-setup)
  - [1.3 Running the Streamlit App](#13-running-the-streamlit-app)
  - [1.4 Demo Flow](#14-demo-flow)
  - [1.5 Validation Commands](#15-validation-commands)
  - [1.6 Screenshots](#16-screenshots)
  - [1.7 Troubleshooting](#17-troubleshooting)
- [2. Milestone 2 - Development of Proof-of-Concepts](#2-milestone-2---development-of-proof-of-concepts)
  - [2.1 Model Integration](#21-model-integration)
  - [2.2 App Development](#22-app-development)
  - [2.3 End-to-End Scenario Testing](#23-end-to-end-scenario-testing)
- [3. References](#3-references)

## Quick Links

- App: [app.py](app.py)
- Baseline solver: [src/baseline_solver.py](src/baseline_solver.py)
- Unified PoC pipeline: [src/poc_pipeline.py](src/poc_pipeline.py)
- LLM backend abstraction: [src/llm_backend.py](src/llm_backend.py)
- LLM generation pipeline: [src/llm_generate.py](src/llm_generate.py)
- Sandboxed execution: [src/exec_generated.py](src/exec_generated.py)
- Shared dataset paths: [src/paths.py](src/paths.py)
- End-to-end tests: [tests/e2e.py](tests/e2e.py)
- End-to-end results evidence: [reports/e2e_results.txt](reports/e2e_results.txt)
- Structured E2E results: [reports/e2e_results.json](reports/e2e_results.json)
- Shared dataset: [../data/raw/](../data/raw/)
- Best-known objectives: [../data/raw/uncapopt.txt](../data/raw/uncapopt.txt)

## 1. Setup, Usage, and Demo Guide

### 1.1 Repository Structure

The milestone is organized around one baseline-first proof-of-concept flow:

- [app.py](app.py): Streamlit user interface
- [src/baseline_solver.py](src/baseline_solver.py): OR-Library parser and deterministic OR-Tools CBC baseline
- [src/poc_pipeline.py](src/poc_pipeline.py): unified PoC scenario runner
- [src/llm_backend.py](src/llm_backend.py): LLM backend abstraction
- [src/llm_generate.py](src/llm_generate.py): code generation and validation
- [src/exec_generated.py](src/exec_generated.py): sandboxed execution and strict output validation
- [src/pipeline_trace.py](src/pipeline_trace.py): execution trace for the UI
- [tests/e2e.py](tests/e2e.py): end-to-end milestone test
- [requirements.txt](requirements.txt): dependencies
- [assets/screenshots/](assets/screenshots/): demo screenshots
- [../data/raw/](../data/raw/): shared OR-Library UFLP instances and [../data/raw/uncapopt.txt](../data/raw/uncapopt.txt)

### 1.2 Installation and Setup

Create and activate a virtual environment, install dependencies, and optionally configure the LLM environment variables from the `Milestone_2-PoC` folder.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The deterministic baseline path does not require an API key. The optional LLM verification path uses Groq:

```powershell
$env:GROQ_API_KEY="YOUR_KEY"
$env:GROQ_MODEL="llama-3.1-8b-instant"
```

A key can be obtained from: https://console.groq.com/keys

### 1.3 Running the Streamlit App

Launch the Streamlit app:

```powershell
python -m streamlit run app.py
```

The app allows the user to:

- select an OR-Library instance from the shared dataset
- run the deterministic CBC baseline
- optionally enable the LLM-generated solver verification path
- inspect the baseline result, optional generated result, and the pipeline trace

### 1.4 Demo Flow

The demo flow is:

- choose an OR-Library instance from the shared dataset
- run the deterministic CBC baseline
- optionally enable LLM-generated solver verification for the same instance
- inspect the baseline objective, best-known gap when available, generated objective when available, and the full pipeline trace

This replaces the older split between a baseline-only mode and a separate LLM mode with one clearer scenario centered on the trusted baseline.

### 1.5 Validation Commands

Run the deterministic end-to-end scenario without an external API key:

```powershell
$env:E2E_ENABLE_LLM="0"
python tests\e2e.py
```

Run the pytest-compatible validation path:

```powershell
python -m pytest tests\e2e.py -q
```

Run optional LLM verification when credentials are available:

```powershell
$env:E2E_ENABLE_LLM="1"
$env:GROQ_API_KEY="YOUR_KEY"
python tests\e2e.py
```

### 1.6 Screenshots

Captured application screenshots are stored in [assets/screenshots/](assets/screenshots/). The folder includes baseline runs and LLM-verification runs that can be reused in the milestone report and demo package.

### 1.7 Troubleshooting

- If `streamlit` is not found, use `python -m streamlit run app.py`.
- If `No instances found` appears, verify that the dataset exists in [../data/raw/](../data/raw/).
- If the LLM path is disabled or no API key is set, the app still runs the deterministic baseline path.
- If the LLM path hits rate limits, the backend retries automatically; if the failure persists, wait briefly and rerun.
- The app shows a pipeline trace so generation, sandbox execution, and comparison failures can be inspected directly in the UI.

## 2. Milestone 2 - Development of Proof-of-Concepts

Milestone 2 proves that the Milestone 1 project idea can run as an interactive application. The PoC uses the same OR-Library UFLP benchmark data from [../data/raw/](../data/raw/), solves selected instances with the deterministic OR-Tools/CBC baseline, and optionally asks an off-the-shelf LLM to generate solver-compatible code for the same instance. The generated code is not trusted automatically; it is executed in a controlled path and compared against the deterministic baseline.

### 2.1 Model Integration

Implementation evidence:

- Baseline model: [src/baseline_solver.py](src/baseline_solver.py)
- Scenario runner: [src/poc_pipeline.py](src/poc_pipeline.py)
- Optional LLM backend: [src/llm_backend.py](src/llm_backend.py)
- LLM code generation: [src/llm_generate.py](src/llm_generate.py)
- Generated-code execution and validation: [src/exec_generated.py](src/exec_generated.py)
- Shared data path contract: [src/paths.py](src/paths.py)

The baseline parser reads OR-Library UFLP files, ignores capacities and demands as required by the project contract, builds the canonical uncapacitated facility-location MILP, solves with OR-Tools/CBC, and compares against [../data/raw/uncapopt.txt](../data/raw/uncapopt.txt). The optional LLM path uses the LLM as a symbolic modeler: it generates solver code, while OR-Tools/CBC remains the executor and verifier.

| Component | Role in the PoC | Evidence |
|---|---|---|
| Deterministic OR-Tools/CBC baseline | Trusted reference solver and verifier | [src/baseline_solver.py](src/baseline_solver.py) |
| Groq-hosted LLM | Optional symbolic model/code generator | [src/llm_backend.py](src/llm_backend.py), [src/llm_generate.py](src/llm_generate.py) |
| Sandboxed executor | Runs generated code and validates objective/assignments | [src/exec_generated.py](src/exec_generated.py) |
| Scenario runner | Connects baseline, optional LLM, execution, and comparison | [src/poc_pipeline.py](src/poc_pipeline.py) |

This satisfies the model-integration requirement because the PoC runs a real project baseline on the actual benchmark data and demonstrates the LLM-assisted symbolic modeling path without replacing the trusted solver.

### 2.2 App Development

Implementation evidence:

- Streamlit app: [app.py](app.py)
- Screenshots: [assets/screenshots/](assets/screenshots/)
- Requirements: [requirements.txt](requirements.txt)

The app gives a user-facing workflow:

1. Select a catalog benchmark instance.
2. Run the deterministic CBC baseline.
3. Optionally enable LLM-generated solver-code verification.
4. Inspect objective value, best-known gap, generated result, and execution trace.

The UI intentionally stays PoC-focused. It is not a production service yet; that role is handled later in Milestone 5. For Milestone 2, the important evidence is that the project idea can be exercised interactively and that the user can see the baseline, optional LLM result, and verification trace in one scenario.

### 2.3 End-to-End Scenario Testing

Implementation evidence:

- End-to-end test: [tests/e2e.py](tests/e2e.py)
- App smoke flow: [app.py](app.py)
- Saved validation output: [reports/e2e_results.txt](reports/e2e_results.txt)
- Structured validation output: [reports/e2e_results.json](reports/e2e_results.json)

The tested scenario is:

1. Load an OR-Library instance from the shared repository data.
2. Parse it into the UFLP problem representation.
3. Solve it with OR-Tools/CBC.
4. Compare the objective against the known optimum when available.
5. Optionally generate solver code with the LLM.
6. Execute generated code and compare it against the deterministic baseline.

The baseline E2E path verifies multiple benchmark instances. The optional LLM path is enabled with `E2E_ENABLE_LLM=1` when credentials are available. This gives a reproducible grading path without requiring an external API key, while still proving that the LLM scenario works when configured.

Alignment with Milestone 1:

- Same UFLP problem.
- Same canonical raw-data source.
- OR-Tools/CBC remains the trusted solver.
- LLM remains a symbolic code/model generator, not the optimizer.

## 3. References

- OR-Library, UFLP benchmark instances and best-known solutions, stored in [../data/raw/](../data/raw/)
- Google OR-Tools CBC solver documentation: https://developers.google.com/optimization
- Groq API documentation: https://console.groq.com/docs
