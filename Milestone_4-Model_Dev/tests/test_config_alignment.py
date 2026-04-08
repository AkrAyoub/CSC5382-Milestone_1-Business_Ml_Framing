from __future__ import annotations

import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
M4_ROOT = TESTS_DIR.parent
SRC_DIR = M4_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from m4_model_dev.utils.config import load_yaml_config


class ConfigAlignmentTests(unittest.TestCase):
    def test_selected_model_compare_and_train_configs_match(self) -> None:
        train_config = load_yaml_config(M4_ROOT / "configs" / "train_best_model.yaml")
        compare_config = load_yaml_config(M4_ROOT / "configs" / "compare_models.yaml")

        model_family = train_config["model_family"]
        compare_model_config = compare_config["comparison"][model_family]

        for key, value in train_config["training"].items():
            if key == "threshold_grid":
                continue
            self.assertEqual(compare_model_config[key], value)


if __name__ == "__main__":
    unittest.main()
