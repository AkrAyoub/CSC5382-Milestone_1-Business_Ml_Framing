from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from m4_model_dev.paths import M2_ROOT, M3_RAW_DIR, M4_LABELS_DIR


OPTIMA_PATH = M3_RAW_DIR / "uncapopt.txt"


@dataclass
class FacilityOpenLabelRow:
    instance_id: str
    facility_id: int
    is_open: int
    open_facility_count: int
    objective: float
    best_known: float | None
    gap_percent: float | None
    runtime_s: float | None


@dataclass
class ParsedOrlibInstance:
    instance_id: str
    m: int
    n: int
    fixed_costs: list[float]
    costs: list[list[float]]


def _load_m2_baseline_solver() -> Any:
    module_path = M2_ROOT / "src" / "baseline_solver.py"
    spec = importlib.util.spec_from_file_location("m2_baseline_solver", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load baseline solver from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_orlib_uncap(instance_path: Path) -> ParsedOrlibInstance:
    tokens = iter(instance_path.read_text(encoding="utf-8").split())
    m = int(next(tokens))
    n = int(next(tokens))

    fixed_costs: list[float] = []
    for _ in range(m):
        first_token = next(tokens)
        second_token = next(tokens)
        # OR-Library variants may include a leading capacity-like token.
        try:
            float(first_token)
        except ValueError:
            pass
        fixed_costs.append(float(second_token))

    costs: list[list[float]] = []
    for _ in range(n):
        first_token = next(tokens)
        try:
            float(first_token)
        except ValueError:
            _ = float(next(tokens))
        costs.append([float(next(tokens)) for _ in range(m)])

    return ParsedOrlibInstance(
        instance_id=instance_path.stem,
        m=m,
        n=n,
        fixed_costs=fixed_costs,
        costs=costs,
    )


def _load_best_known_optima(m2_module: Any) -> dict[str, float]:
    return m2_module.parse_uncapopt(str(OPTIMA_PATH)) if OPTIMA_PATH.exists() else {}


def _build_instance_result_rows(m2_module: Any, parsed: ParsedOrlibInstance, best_known: float | None) -> list[FacilityOpenLabelRow]:
    inst = m2_module.UFLPInstance(
        name=parsed.instance_id,
        m=parsed.m,
        n=parsed.n,
        fixed_costs=parsed.fixed_costs,
        costs=parsed.costs,
    )
    objective, runtime_s, open_facilities, _assignments = m2_module.solve_uflp_cbc(inst)
    gap_percent = None
    if best_known is not None:
        denom = max(1.0, abs(best_known))
        gap_percent = (objective - best_known) / denom * 100.0

    open_set = set(open_facilities)
    return [
        FacilityOpenLabelRow(
            instance_id=inst.name,
            facility_id=facility_id,
            is_open=1 if facility_id in open_set else 0,
            open_facility_count=len(open_facilities),
            objective=objective,
            best_known=best_known,
            gap_percent=gap_percent,
            runtime_s=runtime_s,
        )
        for facility_id in range(inst.m)
    ]


def build_facility_open_labels(output_path: Path | None = None) -> Path:
    output_path = output_path or (M4_LABELS_DIR / "facility_open_labels.csv")
    m2_module = _load_m2_baseline_solver()
    opt_map = _load_best_known_optima(m2_module)

    rows: list[FacilityOpenLabelRow] = []
    for instance_path in sorted(M3_RAW_DIR.glob("*.txt")):
        if instance_path.name == "uncapopt.txt":
            continue

        parsed = _parse_orlib_uncap(instance_path)
        rows.extend(_build_instance_result_rows(m2_module, parsed, opt_map.get(parsed.instance_id)))

    df = pd.DataFrame([asdict(row) for row in rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
