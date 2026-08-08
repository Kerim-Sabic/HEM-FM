param(
    [string]$RunDirectory = "$PSScriptRoot\..\runtime\runs\week_training",
    [int]$IntervalSeconds = 3
)

$ErrorActionPreference = "SilentlyContinue"
$Host.UI.RawUI.WindowTitle = "HEM-FM v4 - Live Training Monitor"

while ($true) {
    Clear-Host
    Write-Host "HEM-FM v4 - LIVE LOCAL TRAINING" -ForegroundColor Cyan
    Write-Host (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    Write-Host ""

    $StatusPath = Join-Path $RunDirectory "status.json"
    if (Test-Path -LiteralPath $StatusPath) {
        $Status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
        Write-Host "Phase:       $($Status.phase)" -ForegroundColor Yellow
        if ($null -ne $Status.complete_cines) {
            Write-Host "Cines:       $($Status.complete_cines) / $($Status.total_cines)"
        }
        if ($null -ne $Status.complete_runs) {
            Write-Host "Runs:        $($Status.complete_runs) / $($Status.total_runs)"
        }
        Write-Host "Locked test: $($Status.locked_test_accessed)"
    } else {
        Write-Host "Waiting for status file: $StatusPath" -ForegroundColor DarkYellow
    }

    Write-Host ""
    Write-Host "GPU UTILIZATION" -ForegroundColor Green
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader

    Write-Host ""
    Write-Host "ACTIVE HEM-FM PROCESSES" -ForegroundColor Green
    $Processes = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*hemfm*" -and $_.ProcessId -ne $PID
    }
    if ($Processes) {
        $Processes | Select-Object ProcessId, Name, CreationDate, CommandLine | Format-Table -Wrap
    } else {
        Write-Host "No active training process; waiting for the next launched phase."
    }

    Write-Host ""
    Write-Host "Refresh: every $IntervalSeconds seconds. Close this window only to stop monitoring." -ForegroundColor DarkGray
    Start-Sleep -Seconds $IntervalSeconds
}

