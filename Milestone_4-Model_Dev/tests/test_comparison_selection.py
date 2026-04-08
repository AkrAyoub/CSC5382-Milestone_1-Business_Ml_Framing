from __future__ import annotations

import sys
import unittest

import pandas as pd

from _bootstrap import ensure_src_path


M4_ROOT = ensure_src_path()

from m4_model_dev.pipelines.comparison_pipeline import order_results_df


class ComparisonSelectionTests(unittest.TestCase):
    def test_ordering_uses_validation_metrics_not_test_f1(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "model_name": "model_a",
                    "val_f1": 0.70,
                    "val_roc_auc": 0.82,
                    "val_precision": 0.60,
                    "test_f1": 0.90,
                },
                {
                    "model_name": "model_b",
                    "val_f1": 0.70,
                    "val_roc_auc": 0.91,
                    "val_precision": 0.55,
                    "test_f1": 0.10,
                },
            ]
        )

        ordered = order_results_df(df)

        self.assertEqual(ordered.iloc[0]["model_name"], "model_b")


if __name__ == "__main__":
    unittest.main()
