#!/usr/bin/env bash
# Local validation gauntlet for TradeExecGym
#
# Usage:
#   ./local-validate.sh [base_url]
# Example:
#   ./local-validate.sh http://127.0.0.1:7865

set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:7865}"
PORT="${BASE_URL##*:}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif [ -x "./.venv/Scripts/python.exe" ]; then
  PYTHON_BIN="./.venv/Scripts/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="python.exe"
else
  echo "python not found"
  exit 1
fi

echo "Starting local validation..."

"$PYTHON_BIN" -m uvicorn server.app:app --host 0.0.0.0 --port "$PORT" >/tmp/tradegym_server.log 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 40); do
  if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Running full robustness validation..."
"$PYTHON_BIN" tests/validate_robustness.py --full --url "$BASE_URL"

echo "Running edge-case tests..."
"$PYTHON_BIN" -m pytest tests/test_edge_cases.py -v

echo "Local validation passed."
