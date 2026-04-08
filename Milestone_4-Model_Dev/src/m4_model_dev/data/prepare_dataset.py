from __future__ import annotations

from pathlib import Path

import pandas as pd

from m4_model_dev.paths import M3_FEATURES_DIR, M4_LABELS_DIR, M4_MERGED_DIR


FACILITY_FEATURES_PATH = M3_FEATURES_DIR / "facility_features.csv"
INSTANCE_FEATURES_PATH = M3_FEATURES_DIR / "instance_features.csv"


def _read_required_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Required input file was not found: {csv_path}")
    return pd.read_csv(csv_path)


def _merge_feature_tables(
    facility_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    # Instance-level aggregates are joined once per facility row so the final
    # training table contains both local facility signals and global scenario
    # context for the same supervised example.
    return facility_df.merge(
        instance_df.drop(columns=["event_timestamp"]),
        on="instance_id",
        how="left",
        suffixes=("", "_instance"),
    ).merge(
        labels_df,
        on=["instance_id", "facility_id"],
        how="inner",
    )


def build_training_dataset(
    labels_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    labels_path = labels_path or (M4_LABELS_DIR / "facility_open_labels.csv")
    output_path = output_path or (M4_MERGED_DIR / "facility_training_dataset.csv")

    facility_df = _read_required_csv(FACILITY_FEATURES_PATH)
    instance_df = _read_required_csv(INSTANCE_FEATURES_PATH)
    labels_df = _read_required_csv(labels_path)
    merged_df = _merge_feature_tables(facility_df, instance_df, labels_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    return output_path
