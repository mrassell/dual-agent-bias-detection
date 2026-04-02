#!/usr/bin/env python3
"""Alias for run_deliverables.main."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_deliverables import main

if __name__ == "__main__":
    main()
