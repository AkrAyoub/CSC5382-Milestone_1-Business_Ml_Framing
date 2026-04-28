from __future__ import annotations

import sys
from pathlib import Path


M5_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = M5_ROOT.parent
M4_ROOT = REPO_ROOT / "Milestone_4-Model_Dev"
M4_SRC_DIR = M4_ROOT / "src"

if str(M4_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(M4_SRC_DIR))

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
UNCAPOPT_PATH = DATA_RAW_DIR / "uncapopt.txt"

M5_ASSETS_DIR = M5_ROOT / "assets"
M5_DEPLOYMENT_DIR = M5_ROOT / "deployment"
M5_DOCS_DIR = M5_ROOT / "docs"
M5_REPORTS_DIR = M5_ROOT / "reports"
M5_RUNTIME_DIR = M5_ROOT / ".m5_runtime"
M5_GENERATED_CODE_DIR = M5_RUNTIME_DIR / "generated_code"
M5_TEMP_INPUT_DIR = M5_RUNTIME_DIR / "tmp_inputs"
M5_MLFLOW_DIR = Path(M5_ROOT.anchor) / "csc5382_m5_mlruns"

M4_MODEL_SELECTION_PATH = M4_ROOT / "reports" / "model_selection.json"
M4_ZENML_STATUS_PATH = M4_ROOT / "reports" / "zenml_status.json"


def ensure_runtime_dirs() -> None:
    for path in [
        M5_ASSETS_DIR,
        M5_DEPLOYMENT_DIR,
        M5_DOCS_DIR,
        M5_REPORTS_DIR,
        M5_RUNTIME_DIR,
        M5_GENERATED_CODE_DIR,
        M5_TEMP_INPUT_DIR,
        M5_MLFLOW_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
