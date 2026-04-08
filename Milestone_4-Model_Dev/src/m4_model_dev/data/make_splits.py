from __future__ import annotations

from pathlib import Path

import pandas as pd

from m4_model_dev.paths import M4_MERGED_DIR, M4_SPLITS_DIR


def assign_split_for_group(instance_ids: list[str]) -> dict[str, str]:
    instance_ids = sorted(instance_ids)
    size = len(instance_ids)
    assignments: dict[str, str] = {}

    if size == 1:
        assignments[instance_ids[0]] = "train"
        return assignments
    if size == 2:
        assignments[instance_ids[0]] = "train"
        assignments[instance_ids[1]] = "test"
        return assignments
    if size == 3:
        assignments[instance_ids[0]] = "train"
        assignments[instance_ids[1]] = "val"
        assignments[instance_ids[2]] = "test"
        return assignments

    for idx, instance_id in enumerate(instance_ids):
        if idx < 2:
            assignments[instance_id] = "train"
        elif idx == 2:
            assignments[instance_id] = "val"
        else:
            assignments[instance_id] = "test"
    return assignments


def build_grouped_splits(
    dataset_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    dataset_path = dataset_path or (M4_MERGED_DIR / "facility_training_dataset.csv")
    output_path = output_path or (M4_SPLITS_DIR / "instance_splits.csv")

    df = pd.read_csv(dataset_path)
    instances = (
        df[["instance_id", "facility_count_m", "customer_count_n"]]
        .drop_duplicates()
        .sort_values(["facility_count_m", "customer_count_n", "instance_id"])
    )

    split_rows: list[dict[str, str | int]] = []
    # Group by problem size so related variants stay together when we assign
    # train/validation/test instances.
    for (_, group_df) in instances.groupby(["facility_count_m", "customer_count_n"], sort=True):
        assignments = assign_split_for_group(group_df["instance_id"].tolist())
        for _, row in group_df.iterrows():
            split_rows.append(
                {
                    "instance_id": row["instance_id"],
                    "facility_count_m": int(row["facility_count_m"]),
                    "customer_count_n": int(row["customer_count_n"]),
                    "split": assignments[row["instance_id"]],
                }
            )

    split_df = pd.DataFrame(split_rows).sort_values(["split", "facility_count_m", "instance_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    return output_path
