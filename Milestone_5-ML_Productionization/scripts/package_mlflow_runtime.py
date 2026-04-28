from __future__ import annotations

import os

from m5_productionization.mlflow_runtime import log_runtime_model


if __name__ == "__main__":
    register_model = os.getenv("M5_REGISTER_RUNTIME_MODEL", "0") == "1"
    model_uri, model_name = log_runtime_model(register_model=register_model)
    if model_name:
        print(f"Registered runtime model: {model_name}")
    else:
        print("Logged runtime model without registry registration")
    print(f"Model URI: {model_uri}")
