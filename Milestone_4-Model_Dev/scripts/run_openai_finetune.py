from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_path


ensure_src_path()

from m4_model_dev.training.openai_finetune import create_openai_finetune_job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch OpenAI supervised fine-tuning for the M4 UFLP solver dataset.")
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14", help="OpenAI base model to fine-tune.")
    parser.add_argument("--suffix", default="uflp-symbolic-solver", help="Fine-tuned model suffix.")
    args = parser.parse_args()

    result = create_openai_finetune_job(model=args.model, suffix=args.suffix)
    print(json.dumps({
        "job_id": result["job"].get("id"),
        "status": result.get("status"),
        "fine_tuned_model": result.get("fine_tuned_model"),
        "training_file_id": result.get("training_file_id"),
        "validation_file_id": result.get("validation_file_id"),
    }, indent=2))
