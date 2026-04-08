from __future__ import annotations

import unittest

from _bootstrap import ensure_src_path


M4_ROOT = ensure_src_path()


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(M4_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
