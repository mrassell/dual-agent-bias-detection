#!/usr/bin/env python3
"""Export the MCP audit SQLite database to CSV and a Markdown sample.

CLI wrapper around ``mcp_server.bias_surface.run_export_audit_artifacts`` (same
as MCP tool ``export_audit_artifacts``).

Outputs
-------
outputs/audit_log_export.csv
outputs/audit_log_sample.md

Usage
-----
    python scripts/export_audit_log.py

Env vars
--------
    MCP_AUDIT_DB    path to the SQLite audit database (default: bias_mcp_audit.db)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

OUTPUTS = ROOT / "outputs"

from mcp_server.bias_surface import run_export_audit_artifacts


def main() -> None:
    result = run_export_audit_artifacts()
    if result.get("error") == "audit_db_not_found":
        print(f"ERROR: {result.get('demo_readout')}", file=sys.stderr)
        sys.exit(1)

    print(result.get("demo_readout", ""))
    print(f"Total logged calls : {result.get('total_calls', 0)}")
    if result.get("tool_counts"):
        print("\nTool call breakdown:")
        for tool, count in sorted(result["tool_counts"].items(), key=lambda x: -x[1]):
            print(f"  {tool:<35} {count:>5}")
    if result.get("csv_path"):
        print(f"\nSaved {result['csv_path']}")
    if result.get("markdown_sample_path"):
        print(f"Saved {result['markdown_sample_path']}")
    if result.get("time_range_utc"):
        tr = result["time_range_utc"]
        print(f"\nLog time range : {tr['first']}  →  {tr['last']}")
    print(f"\nAll exports saved to {OUTPUTS}/")


if __name__ == "__main__":
    main()
