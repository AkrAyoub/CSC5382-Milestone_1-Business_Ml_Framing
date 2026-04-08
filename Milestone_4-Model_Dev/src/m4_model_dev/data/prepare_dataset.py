from __future__ import annotations

from pathlib import Path

import pandas as pd

from m4_model_dev.paths import M3_FEATURES_DIR, M4_LABELS_DIR, M4_MERGED_DIR


FACILITY_FEATURES_PATH = M3_FEATURES_DIR / "facility_features.csv"
INSTANCE_FEATURES_PATH = M3_FEATURES_DIR / "instance_features.csv"


def build_training_dataset(
    labels_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    labels_path = labels_path or (M4_LABELS_DIR / "facility_open_labels.csv")
    output_path = output_path or (M4_MERGED_DIR / "facility_training_dataset.csv")

    facility_df = pd.read_csv(FACILITY_FEATURES_PATH)
    instance_df = pd.read_csv(INSTANCE_FEATURES_PATH)
    labels_df = pd.read_csv(labels_path)

    # Keep one facility row per instance/facility pair and append instance-level
    # aggregates only once during the merge.
    merged_df = facility_df.merge(
        instance_df.drop(columns=["event_timestamp"]),
        on="instance_id",
        how="left",
        suffixes=("", "_instance"),
    ).merge(
        labels_df,
        on=["instance_id", "facility_id"],
        how="inner",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    return output_path
