from __future__ import annotations

import sys

from _bootstrap import ensure_src_path, resolve_config_arg


M4_ROOT = ensure_src_path()

from m4_model_dev.pipelines.comparison_pipeline import run_model_comparison_pipeline


if __name__ == "__main__":
    config_arg = resolve_config_arg(sys.argv, "compare_models.yaml")
    result = run_model_comparison_pipeline(config_arg)
    print(result["summary_path"].read_text(encoding="utf-8"))
