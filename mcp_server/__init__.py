"""
Bias / BASIL toolkit — layered layout:

- ``mcp_server.mcp`` — MCP stdio **transport** (tool names → capability calls + MCP audit wrap).
- ``mcp_server.bias_surface`` — **Capability API** (what CLIs and notebooks should import).
- ``mcp_server.providers`` — **Model-agnostic LLM** slots (env-driven vendors/models).
- ``mcp_server.auditing`` — SQLite **replay log** (usable outside MCP).
- Domain modules (``basil_*``, ``auditor``, ``nli``, …) — data and local models.

See ``docs/REPO_LAYOUT.md``.
"""

__version__ = "0.1.0"
