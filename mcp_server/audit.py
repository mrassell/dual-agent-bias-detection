"""Backward-compatible import path. Prefer ``mcp_server.auditing``."""

from mcp_server.auditing import audit_db_path, init_db, log_call, recent_events

__all__ = ["audit_db_path", "init_db", "log_call", "recent_events"]
