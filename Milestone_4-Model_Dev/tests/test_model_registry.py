from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TESTS_DIR = Path(__file__).resolve().parent
M4_ROOT = TESTS_DIR.parent
SRC_DIR = M4_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from m4_model_dev.models.logistic_numpy import StandardScaler
from m4_model_dev.models.model_registry import fit_model_bundle


class ModelRegistryTests(unittest.TestCase):
    def test_hist_gradient_boosting_uses_configured_early_stopping(self) -> None:
        x_train = np.array(
            [
                [0.1, 1.0],
                [0.2, 0.9],
                [0.8, 0.3],
                [0.9, 0.2],
                [0.15, 0.85],
                [0.85, 0.25],
            ],
            dtype=float,
        )
        y_train = np.array([0, 0, 1, 1, 0, 1], dtype=int)
        scaler = StandardScaler().fit(x_train)
        bundle = fit_model_bundle(
            model_name="hist_gradient_boosting",
            config={
                "training": {
                    "learning_rate": 0.05,
                    "max_depth": 2,
                    "max_iter": 20,
                    "early_stopping": False,
                }
            },
            x_train_raw=x_train,
            x_train_scaled=scaler.transform(x_train),
            y_train=y_train,
            scaler=scaler,
            random_seed=42,
        )

        self.assertFalse(bundle["model"].early_stopping)


if __name__ == "__main__":
    unittest.main()
