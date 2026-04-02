#!/bin/bash
# TradeExecGym — Validation Gauntlet
# ============================================================
# Use this script to verify Phase 1 (Skeleton + Deploy)
# and all subsequent phases before pushing to HF Spaces.
# ============================================================

set -e

# Default values
BASE_URL=${1:-"http://localhost:7860"}
PROJECT_DIR=${2:-"."}

echo "🚀 Starting TradeExecGym Validation Gauntlet..."
echo "------------------------------------------------------------"

# 1. Health Check
echo "🔍 Check 1/3: Server Health..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/health || echo "FAILED")

if [ "$HEALTH_STATUS" == "200" ]; then
    echo "✅ Health check passed: HTTP 200"
else
    echo "❌ Health check failed: HTTP $HEALTH_STATUS"
    echo "   Ensure server is running: py -3.10 -m uvicorn server.app:app --port 7860"
    exit 1
fi

# 2. OpenEnv Validation
echo "🔍 Check 2/3: OpenEnv Schema Validation..."
if openenv validate 2>/dev/null; then
    echo "✅ openenv validate passed"
else
    echo "❌ openenv validate failed"
    openenv validate
    exit 1
fi

# 3. Inference Baseline (Demo Mode)
echo "🔍 Check 3/3: Inference Baseline (Skeleton Check)..."
echo "   Running inference.py in demo mode to verify plumbing..."

# Run only first 2 tasks for speed in validation
MAX_STEPS=5 py -3.10 inference.py --tasks task1_twap_beater,task2_vwap_optimizer

echo "------------------------------------------------------------"
echo "✅ All 3/3 checks passed! Phase 1 skeleton is ready."
echo "   You are now cleared to proceed to Phase 2 (Physics Engine)."
