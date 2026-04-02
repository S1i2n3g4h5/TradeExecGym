# TradeExecGym — Validation Gauntlet (PowerShell)
# ============================================================
# Use this script to verify Phase 1 (Skeleton + Deploy)
# and all subsequent phases before pushing to HF Spaces.
# ============================================================

$ErrorActionPreference = "Stop"

# Default values
$baseUrl = $args[0]
if (-not $baseUrl) { $baseUrl = "http://localhost:7860" }

Write-Host "🚀 Starting TradeExecGym Validation Gauntlet..." -ForegroundColor Cyan
Write-Host "------------------------------------------------------------"

# 1. Health Check
Write-Host "🔍 Check 1/3: Server Health..."
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get -TimeoutSec 10
    if ($response.status -eq "healthy") {
        Write-Host "✅ Health check passed: healthy" -ForegroundColor Green
    } else {
        Write-Host "❌ Health check failed: unexpected status $($response.status)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "   Ensure server is running: py -3.10 -m uvicorn server.app:app --port 7860" -ForegroundColor Yellow
    exit 1
}

# 2. OpenEnv Validation
Write-Host "🔍 Check 2/3: OpenEnv Schema Validation..."
try {
    & openenv validate
    Write-Host "✅ openenv validate passed" -ForegroundColor Green
} catch {
    Write-Host "❌ openenv validate failed" -ForegroundColor Red
    exit 1
}

# 3. Inference Baseline (Demo Mode)
Write-Host "🔍 Check 3/3: Inference Baseline (Skeleton Check)..."
Write-Host "   Running inference.py in demo mode to verify plumbing..."

# Run only first 2 tasks for speed in validation
try {
    # limit steps manually to avoid long delay
    $env:MAX_STEPS = "3"
    & py -3.10 inference.py
    Write-Host "✅ inference.py completed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ inference.py failed" -ForegroundColor Red
    exit 1
}

Write-Host "------------------------------------------------------------"
Write-Host "✅ All 3/3 checks passed! Phase 1 skeleton is ready." -ForegroundColor Green
Write-Host "   You are now cleared to proceed to Phase 2 (Physics Engine)." -ForegroundColor Cyan
