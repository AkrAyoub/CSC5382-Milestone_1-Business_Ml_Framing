from __future__ import annotations

import os
import shutil
from pathlib import Path

from m5_productionization.paths import M5_ROOT, REPO_ROOT


SPACE_TEMPLATE_DIR = M5_ROOT / "deployment" / "huggingface_space"
SPACE_BUILD_DIR = (
    Path(M5_ROOT.anchor) / "csc5382_m5_hf_space_build"
    if os.name == "nt"
    else M5_ROOT / ".m5_runtime" / "hf_space_build"
)


def _ignore_runtime_files(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name == "__pycache__"
        or name.endswith(".pyc")
        or name.endswith(".egg-info")
        or name in {".pytest_cache", ".mypy_cache", ".ruff_cache"}
    }


def build_space_bundle() -> Path:
    if SPACE_BUILD_DIR.exists():
        shutil.rmtree(SPACE_BUILD_DIR)
    SPACE_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    for item in SPACE_TEMPLATE_DIR.iterdir():
        target = SPACE_BUILD_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=_ignore_runtime_files)
        else:
            shutil.copy2(item, target)

    for relative in [
        "Milestone_4-Model_Dev/src",
        "Milestone_4-Model_Dev/reports/openai_finetune_job.json",
        "Milestone_4-Model_Dev/reports/model_selection.json",
        "Milestone_5-ML_Productionization/src",
        "data/raw",
    ]:
        source = REPO_ROOT / relative
        if not source.exists():
            continue
        destination = SPACE_BUILD_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, ignore=_ignore_runtime_files)
        else:
            shutil.copy2(source, destination)
    return SPACE_BUILD_DIR


def deploy_space() -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    repo_id = os.getenv("HF_SPACE_ID")
    if not token or not repo_id:
        raise SystemExit("Set HF_TOKEN and HF_SPACE_ID, for example HF_SPACE_ID=username/uflp-production-solver.")

    from huggingface_hub import HfApi, upload_folder

    bundle = build_space_bundle()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(bundle), token=token)
    print(f"Deployed Hugging Face Space: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    if os.getenv("M5_DEPLOY_HF", "0") == "1":
        deploy_space()
    else:
        bundle = build_space_bundle()
        print(f"Built Hugging Face Space bundle at: {bundle}")
        print("Set M5_DEPLOY_HF=1, HF_TOKEN, and HF_SPACE_ID to upload it.")
