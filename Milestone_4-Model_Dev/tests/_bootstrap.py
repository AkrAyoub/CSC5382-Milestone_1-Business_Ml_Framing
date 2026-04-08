from __future__ import annotations

import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
M4_ROOT = TESTS_DIR.parent
SRC_DIR = M4_ROOT / "src"


def ensure_src_path() -> Path:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    return M4_ROOT
