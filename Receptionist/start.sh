#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "$AGENT_PID" ] && kill "$AGENT_PID" 2>/dev/null
    wait
    echo "Done."
}
trap cleanup EXIT INT TERM

echo "=== Starting agent ==="
uv run python3 src/agent.py start &
AGENT_PID=$!

echo "=== Agent running (PID: $AGENT_PID) ==="
echo "Press Ctrl+C to stop."

wait
