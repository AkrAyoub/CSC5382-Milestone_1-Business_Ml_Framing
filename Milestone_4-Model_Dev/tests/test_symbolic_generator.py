from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _bootstrap import ensure_src_path


M4_ROOT = ensure_src_path()

from m4_model_dev.evaluation.generated_exec import run_generated_solver
from m4_model_dev.models.model_registry import resolve_candidate_spec
from m4_model_dev.models.symbolic_generator import generate_solver_code, _repair_generated_code_to_contract
from m4_model_dev.paths import DATA_RAW_DIR


class SymbolicGeneratorTests(unittest.TestCase):
    def test_live_draft_repair_normalizes_known_token_failure(self) -> None:
        raw_code = """
from pathlib import Path
from ortools.linear_solver import pywraplp

def solve(instance_path: str) -> dict:
    raw = Path(instance_path).read_text()
    tokens = raw.split()
    header = [float(tokens[0]), float(tokens[1])]
    m = int(header[0])
    n = int(header[1])
    solver = pywraplp.Solver.CreateSolver("CBC")
    x = []
    y = []
    assignments = []
    return {"objective": None, "open_facilities": [y[i].solution_value() for i in range(m)], "assignments": [x[j].solution_value() for j in range(n)]}
        """.strip()
        repaired = _repair_generated_code_to_contract(raw_code, "token_v0")
        result = run_generated_solver(repaired, str(DATA_RAW_DIR / "cap71.txt"))
        self.assertGreater(result.objective, 0.0)
        self.assertEqual(len(result.assignments), 50)

    def test_live_draft_repair_normalizes_known_robust_failure(self) -> None:
        raw_code = """
import pathlib
import re
from ortools.linear_solver import pywraplp

def solve(instance_path: str) -> dict:
    lines = [line.strip() for line in pathlib.Path(instance_path).read_text().splitlines() if line.strip()]
    header = [float(value) for value in lines[0].split()]
    m = int(header[0])
    n = int(header[1])
    assignments = []
    row = [0]
    assignments.append(row)
    objective = None
    return {"objective": objective, "open_facilities": [0], "assignments": assignments}
        """.strip()
        repaired = _repair_generated_code_to_contract(raw_code, "robust_v1")
        result = run_generated_solver(repaired, str(DATA_RAW_DIR / "cap71.txt"))
        self.assertGreater(result.objective, 0.0)
        self.assertEqual(len(result.assignments), 50)

    def test_robust_candidate_falls_back_to_validated_template_without_endpoint(self) -> None:
        spec = resolve_candidate_spec("llm_robust_prompt_v1")

        with patch.dict(
            "os.environ",
            {"SELF_HOSTED_OPENAI_BASE_URL": "", "M4_DISABLE_TEMPLATE_FALLBACK": "0"},
            clear=False,
        ):
            generated = generate_solver_code(spec)

        self.assertTrue(generated.raw_text.startswith("TEMPLATE_FALLBACK::"))
        result = run_generated_solver(generated.code, str(DATA_RAW_DIR / "cap71.txt"))
        self.assertGreater(result.objective, 0.0)
        self.assertEqual(len(result.assignments), 50)

    def test_live_candidate_fails_without_endpoint_when_template_fallback_is_disabled(self) -> None:
        spec = resolve_candidate_spec("llm_robust_prompt_v1")

        with patch.dict(
            "os.environ",
            {"SELF_HOSTED_OPENAI_BASE_URL": "", "M4_DISABLE_TEMPLATE_FALLBACK": "1"},
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                generate_solver_code(spec)

    def test_token_candidate_template_handles_literal_capacity_variant(self) -> None:
        spec = resolve_candidate_spec("llm_token_prompt_v0")
        sample = "\n".join(
            [
                "2 2",
                "capacity 10",
                "capacity 20",
                "1",
                "3 4",
                "1",
                "5 6",
            ]
        )

        with patch.dict(
            "os.environ",
            {"SELF_HOSTED_OPENAI_BASE_URL": "", "M4_DISABLE_TEMPLATE_FALLBACK": "0"},
            clear=False,
        ):
            generated = generate_solver_code(spec)

        with tempfile.TemporaryDirectory() as tmp_dir:
            instance_path = Path(tmp_dir) / "tiny_capacity.txt"
            instance_path.write_text(sample, encoding="utf-8")
            result = run_generated_solver(generated.code, str(instance_path))

        self.assertGreater(result.objective, 0.0)
        self.assertEqual(len(result.open_facilities), 1)
        self.assertEqual(len(result.assignments), 2)


if __name__ == "__main__":
    unittest.main()
