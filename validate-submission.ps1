param(
    [string]$BaseUrl = "http://127.0.0.1:7865"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting TradeExecGym validation (PowerShell) ..."

if (Test-Path ".\.venv\Scripts\python.exe") {
    $PythonExe = ".\.venv\Scripts\python.exe"
} else {
    $PythonExe = "python"
}

$server = Start-Process -FilePath $PythonExe `
    -ArgumentList "-m","uvicorn","server.app:app","--host","127.0.0.1","--port","7865" `
    -PassThru

try {
    Start-Sleep -Seconds 5

    Write-Host "Running robustness validation..."
    & $PythonExe tests/validate_robustness.py --full --url $BaseUrl
    if ($LASTEXITCODE -ne 0) { throw "validate_robustness failed" }

    Write-Host "Running edge-case tests..."
    & $PythonExe -m pytest tests/test_edge_cases.py -v
    if ($LASTEXITCODE -ne 0) { throw "test_edge_cases failed" }

    Write-Host "PowerShell validation passed."
}
finally {
    if ($server -and (Get-Process -Id $server.Id -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    }
}
