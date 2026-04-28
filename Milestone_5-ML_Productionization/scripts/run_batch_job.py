from __future__ import annotations

import json
from pathlib import Path

from m5_productionization.api.schemas import BatchSolveItem, BatchSolveRequest, InputSource, ServingMode
from m5_productionization.service import ProductionizationService


if __name__ == "__main__":
    service = ProductionizationService()
    catalog = service.list_instances()[:3]
    request = BatchSolveRequest(
        mode=ServingMode.BASELINE,
        items=[
            BatchSolveItem(source=InputSource.CATALOG, instance_id=str(item["instance_id"]))
            for item in catalog
        ],
    )
    result = service.solve_batch(request)
    output_path = Path(__file__).resolve().parents[1] / "reports" / "batch_smoke_summary.json"
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps(result.model_dump(), indent=2))
