"""MCP transport: stdio tool host. Implements the *wiring* layer, not model logic."""

from mcp_server.mcp.stdio_server import main, mcp

__all__ = ["main", "mcp"]
