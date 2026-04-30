from __future__ import annotations

import json

from m5_productionization.serving_pipeline import run_serving_pipeline


if __name__ == "__main__":
    result = run_serving_pipeline()
    print(json.dumps(result, indent=2))
