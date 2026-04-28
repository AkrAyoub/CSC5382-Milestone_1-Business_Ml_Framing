from __future__ import annotations

import os
import platform
import sys

from _bootstrap import ensure_src_path, resolve_config_arg


if platform.system() == "Windows" and not sys.flags.utf8_mode:
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

M4_ROOT = ensure_src_path()

from m4_model_dev.training.sft_training import run_self_hosted_sft_training


if __name__ == "__main__":
    config_arg = resolve_config_arg(sys.argv, "train_self_hosted_fine_tuned.yaml")
    try:
        result = run_self_hosted_sft_training(config_arg)
    except RuntimeError as exc:
        print(f"Self-hosted SFT training is not ready to run locally: {exc}", file=sys.stderr)
        raise SystemExit(2)

    print(result["summary_path"].read_text(encoding="utf-8"))
