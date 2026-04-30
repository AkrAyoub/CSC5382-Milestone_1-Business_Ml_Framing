from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_path


ensure_src_path()

from m4_model_dev.training.openai_finetune import refresh_openai_finetune_status, wait_for_openai_finetune


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check or wait for an OpenAI fine-tuning job.")
    parser.add_argument("--job-id", default=None, help="Fine-tuning job id. Defaults to the saved job artifact.")
    parser.add_argument("--wait", action="store_true", help="Poll until the job reaches a terminal status.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-minutes", type=int, default=120)
    args = parser.parse_args()

    if args.wait:
        result = wait_for_openai_finetune(args.job_id, args.poll_seconds, args.max_wait_minutes)
    else:
        result = refresh_openai_finetune_status(args.job_id)

    print(json.dumps({
        "job_id": result.get("job", {}).get("id"),
        "status": result.get("status"),
        "fine_tuned_model": result.get("fine_tuned_model"),
    }, indent=2))
