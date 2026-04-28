from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    host = os.getenv("M5_API_HOST", "0.0.0.0")
    port = int(os.getenv("M5_API_PORT", "8000"))
    reload_enabled = os.getenv("M5_API_RELOAD", "0") == "1"
    uvicorn.run("m5_productionization.api.main:app", host=host, port=port, reload=reload_enabled)
