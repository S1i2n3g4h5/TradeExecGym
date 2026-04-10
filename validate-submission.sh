#!/usr/bin/env bash
# OpenEnv submission validator for TradeExecGym
#
# Usage:
#   ./validate-submission.sh <ping_url> [repo_dir]
# Example:
#   ./validate-submission.sh https://singhhsa-tradeexecgym.hf.space .

set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"
DOCKER_BUILD_TIMEOUT=900

DOCKER_MODE=""
if command -v docker >/dev/null 2>&1 && docker version >/dev/null 2>&1; then
  DOCKER_MODE="native"
elif command -v powershell.exe >/dev/null 2>&1; then
  if powershell.exe -NoProfile -Command "docker version > \$null 2>&1; if (\$?) { exit 0 } else { exit 1 }" >/dev/null 2>&1; then
    DOCKER_MODE="powershell"
  fi
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="python.exe"
else
  PYTHON_BIN=""
fi

if command -v openenv >/dev/null 2>&1; then
  OPENENV_BIN="openenv"
elif [ -x "./.venv/Scripts/openenv.exe" ]; then
  OPENENV_BIN="./.venv/Scripts/openenv.exe"
elif [ -n "$PYTHON_BIN" ]; then
  OPENENV_BIN="$PYTHON_BIN -m openenv"
else
  OPENENV_BIN=""
fi

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" && pwd)"; then
  echo "Error: repo_dir not found: ${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}"

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }

TMP_ROOT="${TMPDIR:-/tmp}"
if ! mkdir -p "$TMP_ROOT" >/dev/null 2>&1; then
  TMP_ROOT="."
fi

TMP_RESET_OUT="$TMP_ROOT/validate-reset.out"
TMP_DOCKER_OUT="$TMP_ROOT/validate-docker.out"
TMP_OPENENV_OUT="$TMP_ROOT/validate-openenv.out"

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@"
  fi
}

echo
echo "${BOLD}========================================${NC}"
echo "${BOLD}  OpenEnv Submission Validator${NC}"
echo "${BOLD}========================================${NC}"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
echo

# Step 1/3: ping HF space
log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."
HTTP_CODE="$(curl -s -o "$TMP_RESET_OUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 || echo 000)"
if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space /reset responded with 200"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE"
  hint "Check Space is running and URL is correct."
  exit 1
fi

# Step 2/3: docker build
log "${BOLD}Step 2/3: Running docker build${NC} ..."
if [ -z "$DOCKER_MODE" ]; then
  fail "docker command not found"
  hint "Install/start Docker Desktop and ensure CLI access from this shell."
  exit 1
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/"
  exit 1
fi

if [ "$DOCKER_MODE" = "native" ]; then
  BUILD_CMD=(docker build "$DOCKER_CONTEXT")
  BUILD_OK=true
  run_with_timeout "$DOCKER_BUILD_TIMEOUT" "${BUILD_CMD[@]}" >"$TMP_DOCKER_OUT" 2>&1 || BUILD_OK=false
else
  PS_CONTEXT="$DOCKER_CONTEXT"
  if command -v cygpath >/dev/null 2>&1; then
    PS_CONTEXT="$(cygpath -w "$DOCKER_CONTEXT")"
  fi
  BUILD_OK=true
  run_with_timeout "$DOCKER_BUILD_TIMEOUT" powershell.exe -NoProfile -Command \
    "Set-Location -LiteralPath '$PS_CONTEXT'; docker build ." >"$TMP_DOCKER_OUT" 2>&1 || BUILD_OK=false
fi

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  tail -40 "$TMP_DOCKER_OUT" || true
  exit 1
fi

# Step 3/3: openenv validate
log "${BOLD}Step 3/3: Running openenv validate${NC} ..."
if [ -n "$OPENENV_BIN" ]; then
  if (cd "$REPO_DIR" && sh -c "$OPENENV_BIN validate" >"$TMP_OPENENV_OUT" 2>&1); then
    pass "openenv validate passed"
    cat "$TMP_OPENENV_OUT"
  else
    fail "openenv validate failed"
    cat "$TMP_OPENENV_OUT"
    exit 1
  fi
else
  fail "openenv command not found"
  hint "Install with: pip install openenv-core"
  exit 1
fi

echo
echo "${GREEN}${BOLD}All 3/3 checks passed.${NC}"
echo
