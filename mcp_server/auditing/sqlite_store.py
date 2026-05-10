"""Append-only SQLite store for replay-friendly tool / inference logs."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from mcp_server.auditing.invocation import merge_invocation


def _db_path() -> Path:
    raw = os.environ.get("MCP_AUDIT_DB", "bias_mcp_audit.db")
    return Path(raw).expanduser().resolve()


def audit_db_path() -> Path:
    """Path to the SQLite audit database (same as ``MCP_AUDIT_DB``)."""
    return _db_path()


def init_db() -> None:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(ts)"
        )
        conn.commit()
    finally:
        conn.close()


def log_call(
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    *,
    invocation: Mapping[str, Any] | None = None,
) -> str:
    call_id = str(uuid.uuid4())
    if os.environ.get("MCP_DISABLE_AUDIT", "").lower() in ("1", "true", "yes"):
        return call_id
    payload = merge_invocation(arguments, invocation)
    conn = sqlite3.connect(_db_path())
    try:
        conn.execute(
            "INSERT INTO tool_calls (id, ts, tool_name, arguments_json, result_json) VALUES (?, ?, ?, ?, ?)",
            (
                call_id,
                time.time(),
                tool_name,
                json.dumps(payload, ensure_ascii=False, default=str),
                json.dumps(result, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return call_id


def recent_events(limit: int = 50) -> list[dict[str, Any]]:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT id, ts, tool_name, arguments_json, result_json FROM tool_calls ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                {
                    "id": r["id"],
                    "ts": r["ts"],
                    "tool_name": r["tool_name"],
                    "arguments": json.loads(r["arguments_json"]),
                    "result": json.loads(r["result_json"]),
                }
            )
        return rows
    finally:
        conn.close()
