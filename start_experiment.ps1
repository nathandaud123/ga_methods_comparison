# PowerShell script to start experiment with proper logging and monitoring
# Usage: .\start_experiment.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting GA Comparison Experiment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if already running
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*main.py*" -or $_.CommandLine -like "*run_full_experiment.py*"
}

if ($pythonProcs) {
    Write-Host "⚠️  WARNING: Python process already running!" -ForegroundColor Yellow
    Write-Host "   Process IDs: $($pythonProcs.Id -join ', ')" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne 'y') {
        Write-Host "Aborted." -ForegroundColor Red
        exit
    }
}

# Create log directory
$logDir = "results\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# Generate log file name
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "experiment_$timestamp.log"

Write-Host "📝 Log file: $logFile" -ForegroundColor Green
Write-Host ""

# Start the experiment
Write-Host "🚀 Starting experiment..." -ForegroundColor Green
Write-Host "   This will run in the foreground." -ForegroundColor Yellow
Write-Host "   Press Ctrl+C to stop (checkpoint will save progress)" -ForegroundColor Yellow
Write-Host ""

# Run with output to both console and file
$scriptBlock = {
    param($logFile)
    Set-Location $using:PWD
    python main.py --config config.yaml 2>&1 | Tee-Object -FilePath $logFile
}

try {
    & $scriptBlock $logFile
    Write-Host ""
    Write-Host "✅ Experiment completed!" -ForegroundColor Green
    Write-Host "   Log saved to: $logFile" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "❌ Experiment failed!" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host "   Check log: $logFile" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "To check progress:" -ForegroundColor Cyan
Write-Host "  python verify_running.py" -ForegroundColor White
Write-Host "  python check_progress.py" -ForegroundColor White

