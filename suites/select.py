#!/usr/bin/env python3
"""select.py — the name CONTRACTS.md §6.1 records in every seed file's CI check.

The implementation lives in suites/generate_seeds.py; this is a one-line shim so that the
documented invocation (`suites/select.py --verify <seed file>`) works verbatim.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_seeds import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
