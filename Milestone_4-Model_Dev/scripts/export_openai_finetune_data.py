from __future__ import annotations

import json

from _bootstrap import ensure_src_path


ensure_src_path()

from m4_model_dev.training.openai_finetune import export_openai_finetune_files


if __name__ == "__main__":
    result = export_openai_finetune_files()
    print(json.dumps({
        "manifest_path": str(result["manifest_path"]),
        "train_records": result["splits"]["train"]["record_count"],
        "val_records": result["splits"]["val"]["record_count"],
        "test_records": result["splits"]["test"]["record_count"],
    }, indent=2))
