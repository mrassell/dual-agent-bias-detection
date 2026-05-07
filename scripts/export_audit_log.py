#!/usr/bin/env python3
"""Export the MCP audit SQLite database to CSV and a Markdown sample.

Dumps every tool call logged during any experiment run, adds a human-readable
ISO timestamp column, and writes a 20-row Markdown table for the report.

Outputs
-------
outputs/audit_log_export.csv     all rows (id, timestamp, tool, args, result)
outputs/audit_log_sample.md      20-row Markdown table for the paper

Usage
-----
    python scripts/export_audit_log.py

Env vars
--------
    MCP_AUDIT_DB    path to the SQLite audit database (default: bias_mcp_audit.db)
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

OUTPUTS = ROOT / "outputs"

# The DB path mirrors the logic in mcp_server/audit.py
DB_PATH = Path(os.environ.get("MCP_AUDIT_DB", "bias_mcp_audit.db")).expanduser().resolve()

SAMPLE_ROWS = 20


def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)

    if not DB_PATH.exists():
        print(f"ERROR: audit DB not found at {DB_PATH}", file=sys.stderr)
        print("Run any experiment script first to populate the DB.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading audit DB from {DB_PATH} …")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT id, ts, tool_name, arguments_json, result_json "
            "FROM tool_calls ORDER BY ts ASC"
        )
        all_rows = cur.fetchall()
    finally:
        conn.close()

    print(f"Total logged calls : {len(all_rows)}")

    if not all_rows:
        print("No rows found — nothing to export.")
        return

    # ------------------------------------------------------------------ #
    # 1. Full CSV export                                                   #
    # ------------------------------------------------------------------ #
    csv_path = OUTPUTS / "audit_log_export.csv"
    fieldnames = ["id", "timestamp_utc", "tool_name", "arguments", "result"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                "id":            row["id"],
                "timestamp_utc": _ts_to_iso(row["ts"]),
                "tool_name":     row["tool_name"],
                "arguments":     row["arguments_json"],
                "result":        row["result_json"],
            })
    print(f"Saved {csv_path}  ({len(all_rows)} rows)")

    # ------------------------------------------------------------------ #
    # 2. Per-tool summary                                                  #
    # ------------------------------------------------------------------ #
    tool_counts: dict[str, int] = {}
    for row in all_rows:
        tool_counts[row["tool_name"]] = tool_counts.get(row["tool_name"], 0) + 1

    print("\nTool call breakdown:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool:<35} {count:>5}")

    # ------------------------------------------------------------------ #
    # 3. Markdown sample (most recent SAMPLE_ROWS rows)                   #
    # ------------------------------------------------------------------ #
    sample = list(all_rows)[-SAMPLE_ROWS:]

    md_lines = [
        f"# MCP Audit Log — Sample ({SAMPLE_ROWS} most recent calls)",
        "",
        f"**Database**: `{DB_PATH}`  ",
        f"**Total logged calls**: {len(all_rows)}  ",
        f"**Export date**: {_ts_to_iso(__import__('time').time())}",
        "",
        "## Per-tool call counts",
        "",
        "| Tool | Calls |",
        "|------|------:|",
    ]
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        md_lines.append(f"| `{tool}` | {count} |")

    md_lines += [
        "",
        f"## Last {SAMPLE_ROWS} calls",
        "",
        "| # | Timestamp (UTC) | Tool | Key Arguments | Result (excerpt) |",
        "|---|-----------------|------|---------------|-----------------|",
    ]

    for i, row in enumerate(sample, 1):
        ts  = _ts_to_iso(row["ts"])
        try:
            args = json.loads(row["arguments_json"])
        except Exception:
            args = {}
        try:
            result = json.loads(row["result_json"])
        except Exception:
            result = {}

        # Pick the most informative argument field
        arg_display = ""
        for key in ("sentence", "query", "text", "tool_name", "outlet"):
            if key in args:
                val = str(args[key])
                arg_display = f"`{key}`: {_truncate(val, 60)}"
                break
        if not arg_display and args:
            first_key = next(iter(args))
            arg_display = f"`{first_key}`: {_truncate(str(args[first_key]), 60)}"

        # Result: show bias_score / label / claim_follows_from_premise if present
        result_display = ""
        if isinstance(result, dict):
            for rkey in ("bias_score", "label", "entailment_probability",
                         "claim_follows_from_premise", "f1_macro", "accuracy"):
                if rkey in result:
                    result_display = f"`{rkey}={result[rkey]}`"
                    break
        if not result_display:
            result_display = _truncate(str(result), 60)

        tool_name = row["tool_name"]
        md_lines.append(
            f"| {i} | {ts} | `{tool_name}` | {arg_display} | {result_display} |"
        )

    md_path = OUTPUTS / "audit_log_sample.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Saved {md_path}")

    # ------------------------------------------------------------------ #
    # 4. Quick sanity check on time range                                 #
    # ------------------------------------------------------------------ #
    first_ts = _ts_to_iso(all_rows[0]["ts"])
    last_ts  = _ts_to_iso(all_rows[-1]["ts"])
    print(f"\nLog time range : {first_ts}  →  {last_ts}")
    print(f"\nAll exports saved to {OUTPUTS}/")


if __name__ == "__main__":
    main()
