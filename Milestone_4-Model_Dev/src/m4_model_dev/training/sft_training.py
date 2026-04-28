from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

from m4_model_dev.data.build_benchmark_dataset import build_benchmark_dataset
from m4_model_dev.data.build_reference_solutions import build_reference_solutions
from m4_model_dev.data.build_sft_dataset import build_sft_dataset
from m4_model_dev.data.make_splits import build_grouped_splits
from m4_model_dev.paths import (
    M4_CONFIGS_DIR,
    M4_DATASETS_DIR,
    M4_MODELS_DIR,
    M4_REFERENCE_DIR,
    M4_REPORTS_DIR,
    M4_SFT_DIR,
    M4_SPLITS_DIR,
    M4_TRAINING_RUNS_DIR,
    ensure_runtime_dirs,
)
from m4_model_dev.tracking.mlflow_utils import configure_mlflow, get_or_create_experiment
from m4_model_dev.utils.config import load_yaml_config


def load_self_hosted_training_config(config_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    ensure_runtime_dirs()
    resolved_path = config_path or (M4_CONFIGS_DIR / "train_self_hosted_fine_tuned.yaml")
    return resolved_path, load_yaml_config(resolved_path)


def _assert_self_hosted_training_dependencies() -> None:
    missing: list[str] = []
    for module_name in ["torch", "datasets", "peft", "transformers", "trl"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "Self-hosted SFT dependencies are missing: "
            f"{missing_text}. Install Milestone_4-Model_Dev/requirements-selfhosted.txt "
            "only on a machine where you intend to run local/cloud fine-tuning."
        )


def _prepare_sft_assets() -> dict[str, Path]:
    reference_path = build_reference_solutions(M4_REFERENCE_DIR / "reference_solutions.csv")
    dataset_path = build_benchmark_dataset(reference_path=reference_path, output_path=M4_DATASETS_DIR / "benchmark_instances.csv")
    split_path = build_grouped_splits(dataset_path=dataset_path, output_path=M4_SPLITS_DIR / "instance_splits.csv")
    sft_paths = build_sft_dataset(
        split_path=split_path,
        dataset_path=dataset_path,
        reference_path=reference_path,
        output_dir=M4_SFT_DIR,
    )
    return {
        "reference_path": reference_path,
        "dataset_path": dataset_path,
        "split_path": split_path,
        **sft_paths,
    }


def _runtime_manifest_path(run_name: str) -> Path:
    return M4_TRAINING_RUNS_DIR / run_name / "runtime_manifest.json"


def _optional_int(block: dict[str, Any], key: str) -> int | None:
    value = block.get(key)
    if value is None:
        return None
    return int(value)


def _optional_float(block: dict[str, Any], key: str) -> float | None:
    value = block.get(key)
    if value is None:
        return None
    return float(value)


def _summary_text(payload: dict[str, Any]) -> str:
    lines = [
        "Milestone 4 self-hosted SFT training summary",
        f"run_name: {payload['run_name']}",
        f"base_model_name: {payload['base_model_name']}",
        f"adapter_output_dir: {payload['adapter_output_dir']}",
        f"train_records: {payload['train_records']}",
        f"validation_records: {payload['validation_records']}",
        f"elapsed_s: {payload['elapsed_s']:.3f}",
        f"mlflow_logged: {payload['mlflow_logged']}",
    ]
    if payload.get("eval_metrics"):
        lines.append("eval_metrics:")
        for key, value in sorted(payload["eval_metrics"].items()):
            lines.append(f"  - {key}: {value}")
    return "\n".join(lines) + "\n"


def _maybe_quantization_config(model_block: dict[str, Any]):
    use_4bit = bool(model_block.get("use_4bit", False))
    if not use_4bit:
        return None

    if platform.system() == "Windows":
        raise RuntimeError("4-bit QLoRA is disabled in this local Windows setup. Set model.use_4bit=false.")

    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("bitsandbytes/transformers are missing for 4-bit training.") from exc

    compute_dtype_name = str(model_block.get("bnb_4bit_compute_dtype", "bfloat16"))
    compute_dtype = getattr(torch, compute_dtype_name, torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(model_block.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(model_block.get("bnb_4bit_use_double_quant", True)),
    )


def run_self_hosted_sft_training(config_path: Path | None = None) -> dict[str, Any]:
    resolved_config_path, config = load_self_hosted_training_config(config_path)
    _assert_self_hosted_training_dependencies()
    runtime_assets = _prepare_sft_assets()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_block = dict(config.get("model", {}))
    peft_block = dict(config.get("peft", {}))
    training_block = dict(config.get("training", {}))
    tracking_block = dict(config.get("tracking", {}))

    run_name = str(config.get("run_name", "self_hosted_qwen25_sft"))
    train_jsonl = runtime_assets["train_jsonl"]
    eval_jsonl = runtime_assets["val_jsonl"]
    data_files = {"train": str(train_jsonl), "validation": str(eval_jsonl)}
    dataset_dict = load_dataset("json", data_files=data_files)
    if "response" in dataset_dict["train"].column_names and "completion" not in dataset_dict["train"].column_names:
        dataset_dict = dataset_dict.map(lambda row: {"completion": row["response"]})

    base_model_name = str(model_block.get("base_model_name", "Qwen/Qwen2.5-Coder-7B-Instruct"))
    trust_remote_code = bool(model_block.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = _maybe_quantization_config(model_block)
    torch_dtype_name = str(model_block.get("torch_dtype", "float16"))
    torch_dtype = getattr(torch, torch_dtype_name, torch.float16)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    model.config.use_cache = False

    target_modules = peft_block.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    peft_config = LoraConfig(
        r=int(peft_block.get("r", 16)),
        lora_alpha=int(peft_block.get("lora_alpha", 32)),
        lora_dropout=float(peft_block.get("lora_dropout", 0.05)),
        bias=str(peft_block.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )

    output_dir = M4_MODELS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(training_block.get("learning_rate", 2.0e-4)),
        "num_train_epochs": float(training_block.get("num_train_epochs", 2)),
        "per_device_train_batch_size": int(training_block.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_block.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_block.get("gradient_accumulation_steps", 4)),
        "logging_steps": int(training_block.get("logging_steps", 5)),
        "save_steps": int(training_block.get("save_steps", 25)),
        "eval_strategy": str(training_block.get("eval_strategy", "steps")),
        "eval_steps": int(training_block.get("eval_steps", 25)),
        "warmup_ratio": float(training_block.get("warmup_ratio", 0.03)),
        "max_length": int(training_block.get("max_length", 2048)),
        "dataset_text_field": str(training_block.get("dataset_text_field", "text")),
        "report_to": list(training_block.get("report_to", [])),
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(training_block.get("gradient_checkpointing", True)),
        "bf16": bool(training_block.get("bf16", False)),
        "fp16": bool(training_block.get("fp16", False)),
    }
    optional_int_keys = [
        "max_steps",
        "save_total_limit",
        "dataloader_num_workers",
    ]
    optional_float_keys = ["weight_decay", "max_grad_norm"]
    for key in optional_int_keys:
        value = _optional_int(training_block, key)
        if value is not None:
            training_args[key] = value
    for key in optional_float_keys:
        value = _optional_float(training_block, key)
        if value is not None:
            training_args[key] = value

    args = SFTConfig(**training_args)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    started = time.time()
    train_output = trainer.train()
    eval_metrics = trainer.evaluate()
    elapsed_s = time.time() - started

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    runtime_manifest = {
        "run_name": run_name,
        "base_model_name": base_model_name,
        "adapter_output_dir": str(output_dir),
        "train_records": int(len(dataset_dict["train"])),
        "validation_records": int(len(dataset_dict["validation"])),
        "elapsed_s": elapsed_s,
        "eval_metrics": {key: float(value) for key, value in eval_metrics.items() if isinstance(value, (int, float))},
        "train_metrics": {
            key: float(value)
            for key, value in train_output.metrics.items()
            if isinstance(value, (int, float))
        },
        "config_path": str(resolved_config_path),
        "sft_manifest_path": str(runtime_assets["manifest"]),
        "suggested_runtime_env": {
            "SELF_HOSTED_FINE_TUNED_MODEL_NAME": str(output_dir),
        },
    }

    run_dir = _runtime_manifest_path(run_name).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _runtime_manifest_path(run_name)
    manifest_path.write_text(json.dumps(runtime_manifest, indent=2), encoding="utf-8")

    summary_path = run_dir / "training_summary.txt"
    summary_path.write_text(_summary_text({**runtime_manifest, "mlflow_logged": False}), encoding="utf-8")

    mlflow_logged = False
    if bool(tracking_block.get("enable_mlflow", True)):
        try:
            mlflow, client = configure_mlflow()
            experiment_name = str(tracking_block.get("experiment_name", "milestone4-self-hosted-sft"))
            experiment_id = get_or_create_experiment(client, experiment_name)
            with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
                mlflow.log_params(
                    {
                        "config_path": str(resolved_config_path),
                        "base_model_name": base_model_name,
                        "train_jsonl": str(train_jsonl),
                        "eval_jsonl": str(eval_jsonl),
                        "lora_r": peft_config.r,
                        "lora_alpha": peft_config.lora_alpha,
                        "lora_dropout": peft_config.lora_dropout,
                        "target_modules": ",".join(peft_config.target_modules or []),
                    }
                )
                for key, value in runtime_manifest["train_metrics"].items():
                    mlflow.log_metric(f"train_{key}", float(value))
                for key, value in runtime_manifest["eval_metrics"].items():
                    mlflow.log_metric(f"eval_{key}", float(value))
                mlflow.log_metric("elapsed_s", float(elapsed_s))
                mlflow.log_artifact(str(manifest_path))
                mlflow.log_artifact(str(summary_path))
                mlflow_logged = True
        except Exception:
            mlflow_logged = False

    summary_payload = {**runtime_manifest, "mlflow_logged": mlflow_logged}
    summary_path.write_text(_summary_text(summary_payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "config_path": resolved_config_path,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "output_dir": output_dir,
        "sft_manifest_path": runtime_assets["manifest"],
        "mlflow_logged": mlflow_logged,
    }
