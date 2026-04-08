from __future__ import annotations

import sys

from _bootstrap import ensure_src_path, resolve_config_arg


M4_ROOT = ensure_src_path()

from m4_model_dev.pipelines.training_pipeline import run_training_pipeline


if __name__ == "__main__":
    config_arg = resolve_config_arg(sys.argv, "train_best_model.yaml")
    result = run_training_pipeline(config_arg)
    print(result["summary_path"].read_text(encoding="utf-8"))
