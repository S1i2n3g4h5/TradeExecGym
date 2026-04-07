# TradeExecGym — Validation Gauntlet (PowerShell)
# ============================================================
# Use this script to verify all phases including Layer 4 API
# compliance with full robustness validation.
# ============================================================

$ErrorActionPreference = "Stop"

# Default values
$baseUrl = $args[0]
if (-not $baseUrl) { $baseUrl = "http://localhost:7865" }

Write-Host "🚀 Starting TradeExecGym Full Validation Gauntlet..." -ForegroundColor Cyan
Write-Host "============================================================"

# 1. Start server in background
Write-Host "📡 Starting server in background..."
$serverJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    python -m uvicorn server.app:app --host 0.0.0.0 --port 7865
}
Write-Host "   Server started with Job ID: $($serverJob.Id)" -ForegroundColor Green

# Wait for server to be ready
Write-Host "   Waiting for server to initialize..."
Start-Sleep -Seconds 5

# Function to cleanup server on exit
function Cleanup-Server {
    Write-Host ""
    Write-Host "🛑 Stopping server (Job ID: $($serverJob.Id))..." -ForegroundColor Yellow
    Stop-Job -Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job -Job $serverJob -Force -ErrorAction SilentlyContinue
    Write-Host "   Server stopped." -ForegroundColor Green
}

try {
    # 2. Run full robustness validation (all 5 layers + performance + determinism)
    Write-Host "🔍 Running full robustness validation (Layers 0-4 + Performance)..."
    $validationResult = & python tests/validate_robustness.py --full --url $baseUrl
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Full validation passed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Validation failed - check ROBUSTNESS_REPORT.json for details" -ForegroundColor Red
        throw "Validation failed"
    }

    # 3. Run edge case test suite
    Write-Host "🔍 Running edge case test suite..."
    $edgeTestResult = & python -m pytest tests/test_edge_cases.py -v
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Edge case tests passed!" -ForegroundColor Green
    } else {
        Write-Host "❌ Edge case tests failed" -ForegroundColor Red
        throw "Edge case tests failed"
    }

    Write-Host "============================================================"
    Write-Host "✅ All validation checks passed!" -ForegroundColor Green
    Write-Host "   - Layer 0: Environment Boot ✓" -ForegroundColor Cyan
    Write-Host "   - Layer 1: Unit Tests ✓" -ForegroundColor Cyan
    Write-Host "   - Layer 2: Baseline Scores ✓" -ForegroundColor Cyan
    Write-Host "   - Layer 3: Skill Gradient ✓" -ForegroundColor Cyan
    Write-Host "   - Layer 4: OpenEnv API Compliance ✓" -ForegroundColor Cyan
    Write-Host "   - Performance Profiling ✓" -ForegroundColor Cyan
    Write-Host "   - Determinism Check ✓" -ForegroundColor Cyan
    Write-Host "   - Edge Case Suite ✓" -ForegroundColor Cyan
    Write-Host "============================================================"
    
} catch {
    Write-Host "❌ Error during validation: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    Cleanup-Server
}
