from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
M4_ROOT = SCRIPT_DIR.parent
SRC_DIR = M4_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from m4_model_dev.pipelines.comparison_pipeline import run_model_comparison_pipeline


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else (M4_ROOT / "configs" / "compare_models.yaml")
    result = run_model_comparison_pipeline(config_arg)
    print(result["summary_path"].read_text(encoding="utf-8"))
