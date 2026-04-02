"""Locate BASIL `data/` directory (JSON articles)."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_basil_data_dir() -> Path:
    """BASIL_DATA_DIR or ~/basil_workspace/emnlp19-BASIL/data (etc.)."""
    raw = os.environ.get("BASIL_DATA_DIR", "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"BASIL_DATA_DIR is not a directory: {p}")
        if not list(p.glob("*.json")):
            raise FileNotFoundError(f"No *.json BASIL articles in {p}")
        return p

    candidates = [
        Path.home() / "basil_workspace" / "emnlp19-BASIL" / "data",
        Path.home() / "emnlp19-BASIL" / "data",
    ]
    for c in candidates:
        if c.is_dir() and list(c.glob("*.json")):
            return c.resolve()

    raise FileNotFoundError(
        "BASIL corpus not found. Download with BASIL_Bias_Baseline.ipynb (DATASET_DIR) "
        "or set BASIL_DATA_DIR to the folder containing BASIL *.json articles."
    )
