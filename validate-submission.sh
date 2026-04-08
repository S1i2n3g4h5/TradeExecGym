#!/bin/bash
# TradeExecGym — Validation Gauntlet
# ============================================================
# Use this script to verify all phases including Layer 4 API
# compliance with full robustness validation.
# ============================================================

set -e

# Default values
BASE_URL=${1:-"http://localhost:7865"}
PROJECT_DIR=${2:-"."}

echo "🚀 Starting TradeExecGym Full Validation Gauntlet..."
echo "============================================================"

# 1. Start server in background
echo "📡 Starting server in background..."
python3 -m uvicorn server.app:app --host 0.0.0.0 --port 7865 > /tmp/tradegym_server.log 2>&1 &
SERVER_PID=$!
echo "   Server started with PID: $SERVER_PID"

# Wait for server to be ready
echo "   Waiting for server to initialize..."
sleep 5

# Function to cleanup server on exit
cleanup() {
    echo ""
    echo "🛑 Stopping server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "   Server stopped."
}
trap cleanup EXIT

# 2. Run full robustness validation (all 5 layers + performance + determinism)
echo "🔍 Running full robustness validation (Layers 0-4 + Performance)..."
if python3 tests/validate_robustness.py --full --url "$BASE_URL"; then
    echo "✅ Full validation passed!"
else
    echo "❌ Validation failed - check ROBUSTNESS_REPORT.json for details"
    exit 1
fi

# 3. Run edge case test suite
echo "🔍 Running edge case test suite..."
if python3 -m pytest tests/test_edge_cases.py -v; then
    echo "✅ Edge case tests passed!"
else
    echo "❌ Edge case tests failed"
    exit 1
fi

echo "============================================================"
echo "✅ All validation checks passed!"
echo "   - Layer 0: Environment Boot ✓"
echo "   - Layer 1: Unit Tests ✓"
echo "   - Layer 2: Baseline Scores ✓"
echo "   - Layer 3: Skill Gradient ✓"
echo "   - Layer 4: OpenEnv API Compliance ✓"
echo "   - Performance Profiling ✓"
echo "   - Determinism Check ✓"
echo "   - Edge Case Suite ✓"
echo "============================================================"
