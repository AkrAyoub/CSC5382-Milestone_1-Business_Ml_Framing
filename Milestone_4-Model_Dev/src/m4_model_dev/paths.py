from __future__ import annotations

from pathlib import Path


# Resolve milestone paths relative to the installed source tree so local
# scripts and pipeline runs behave the same way.
M4_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = M4_ROOT.parent
M3_ROOT = REPO_ROOT / "Milestone_3-Data_Prep"
M2_ROOT = REPO_ROOT / "Milestone_2-PoC"

M4_DATA_DIR = M4_ROOT / "data"
M4_LABELS_DIR = M4_DATA_DIR / "labels"
M4_MERGED_DIR = M4_DATA_DIR / "merged"
M4_SPLITS_DIR = M4_DATA_DIR / "splits"
M4_ARTIFACTS_DIR = M4_ROOT / "artifacts"
M4_REPORTS_DIR = M4_ROOT / "reports"
M4_CONFIGS_DIR = M4_ROOT / "configs"
M4_MLFLOW_DIR = M4_ROOT / "mlruns"

M3_FEATURES_DIR = M3_ROOT / "data" / "features"
M3_RAW_DIR = M3_ROOT / "data" / "raw"


def ensure_runtime_dirs() -> None:
    runtime_dirs = [
        M4_DATA_DIR,
        M4_LABELS_DIR,
        M4_MERGED_DIR,
        M4_SPLITS_DIR,
        M4_ARTIFACTS_DIR,
        M4_REPORTS_DIR,
        M4_MLFLOW_DIR,
    ]
    for path in runtime_dirs:
        path.mkdir(parents=True, exist_ok=True)
