from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
M4_ROOT = SCRIPT_DIR.parent
SRC_DIR = M4_ROOT / "src"


def ensure_src_path() -> Path:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    return M4_ROOT


def resolve_config_arg(argv: list[str], default_name: str) -> Path:
    root = ensure_src_path()
    return Path(argv[1]) if len(argv) > 1 else (root / "configs" / default_name)
