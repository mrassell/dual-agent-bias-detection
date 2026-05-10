"""Backward-compatible import path for the MCP stdio app.

Prefer: ``from mcp_server.mcp import main, mcp`` or ``python -m mcp_server``.
"""

from mcp_server.mcp.stdio_server import main, mcp

__all__ = ["main", "mcp"]
