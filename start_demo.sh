#!/usr/bin/env bash
# -------------------------------------------------------
# Demo day startup script
# Usage: bash start_demo.sh
#
# Starts the Flask backend and Vite frontend together.
# The frontend auto-detects the backend via window.location.hostname,
# so no .env configuration needed.
# -------------------------------------------------------

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── API keys (edit these once, keep this file off git) ──────────────
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# ── Model slots ──────────────────────────────────────────────────────
export BIAS_LLM_A_VENDOR=openai
export BIAS_LLM_A_MODEL=gpt-4o-mini
export BIAS_LLM_B_VENDOR=anthropic
export BIAS_LLM_B_MODEL=claude-haiku-4-5-20251001
export BIAS_LLM_AUDITOR_SLOT=A
export BIAS_LLM_VERIFIER_SLOT=B

# ── Thresholds ───────────────────────────────────────────────────────
export AUDITOR_THRESHOLD=0.40
export AUDITOR_MODEL_ID=mediabiasgroup/roberta-babe-ft

# ── BASIL data (only needed for eval scripts, not the demo itself) ───
export BASIL_DATA_DIR="${BASIL_DATA_DIR:-}"

# ── Validate API keys ────────────────────────────────────────────────
if [[ -z "$OPENAI_API_KEY" ]]; then
  echo "ERROR: Set OPENAI_API_KEY before running this script."
  echo "  export OPENAI_API_KEY=sk-..."
  exit 1
fi
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
  echo "ERROR: Set ANTHROPIC_API_KEY before running this script."
  echo "  export ANTHROPIC_API_KEY=sk-ant-..."
  exit 1
fi

# ── Install frontend deps if needed ──────────────────────────────────
if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  cd "$ROOT/frontend" && npm install
fi

# ── Get local IP for display ─────────────────────────────────────────
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")

echo ""
echo "======================================================"
echo "  Dual-Agent Bias Detection Demo"
echo "======================================================"
echo "  Backend  →  http://$LOCAL_IP:5001"
echo "  Frontend →  http://$LOCAL_IP:5173"
echo ""
echo "  Share http://$LOCAL_IP:5173 with teammates on the"
echo "  same WiFi — no setup needed on their end."
echo ""
echo "  For different networks: run 'ngrok http 5173' in a"
echo "  new terminal after this script starts."
echo "======================================================"
echo ""

# ── Start backend in background ───────────────────────────────────────
echo "Starting backend..."
cd "$ROOT"
python3 demo/server.py &
BACKEND_PID=$!

# Give Flask a moment to boot
sleep 2

# ── Start frontend (foreground, --host exposes to network) ────────────
echo "Starting frontend..."
cd "$ROOT/frontend"
npm run dev -- --host

# ── Cleanup on exit ───────────────────────────────────────────────────
kill $BACKEND_PID 2>/dev/null
