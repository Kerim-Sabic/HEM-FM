param(
    [string]$Environment = "$PSScriptRoot\..\..\..\work\.venv-hemfm",
    [string]$Config = "$PSScriptRoot\..\configs\protocol.yaml",
    [int]$Workers = 12,
    [int]$Epochs = 10,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path "$PSScriptRoot\..").Path
$Python = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$Config = (Resolve-Path $Config).Path
$RunRoot = (Resolve-Path "$PSScriptRoot\..\..\..\work\hemfm-v4-runtime\runs").Path

Write-Host "HEM-FM v4 staged final scalar wave" -ForegroundColor Cyan
Write-Host "repository : $Repository"
Write-Host "interpreter: $Python"
Write-Host "mutable data: local workstation"
Write-Host "DICOM source: authorized read-only network share"
Write-Host ""

if (-not $SkipTests) {
    Push-Location $Repository
    try {
        & $Python -m pytest -q -p no:cacheprovider
        if ($LASTEXITCODE -ne 0) { throw "Tests failed; staged training was not started." }
    }
    finally { Pop-Location }
}

& $Python -m hemfm --config $Config gates assert --through G5
if ($LASTEXITCODE -ne 0) { throw "G0-G5 evidence is incomplete." }

Write-Host ""
Write-Host "Staging every required development cine onto local storage..." -ForegroundColor Yellow
& $Python -m hemfm --config $Config staged-final cache --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "Local video-cache staging failed." }

$WorkerScript = Join-Path $PSScriptRoot "run_staged_scalar_worker.ps1"
$Worker0 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $WorkerScript,
    "-Device", "0", "-TargetsCsv", "EF,LVESV,RV_BASAL_DIAMETER",
    "-Environment", $Environment, "-Config", $Config, "-Epochs", $Epochs
) -WorkingDirectory $Repository -WindowStyle Normal -PassThru
$Worker1 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-NoExit", "-ExecutionPolicy", "Bypass", "-File", $WorkerScript,
    "-Device", "1", "-TargetsCsv", "LVEDV,LVOT_DIAMETER,AV_PEAK_VELOCITY",
    "-Environment", $Environment, "-Config", $Config, "-Epochs", $Epochs
) -WorkingDirectory $Repository -WindowStyle Normal -PassThru

$Launch = @{
    schema_version = 1
    launched_utc = [DateTime]::UtcNow.ToString("o")
    phase = "staged_final_scalar"
    gpu_0_process_id = $Worker0.Id
    gpu_1_process_id = $Worker1.Id
    gpu_0_targets = @("EF", "LVESV", "RV_BASAL_DIAMETER")
    gpu_1_targets = @("LVEDV", "LVOT_DIAMETER", "AV_PEAK_VELOCITY")
    epochs = $Epochs
    locked_test_accessed = $false
}
$LaunchPath = Join-Path $RunRoot "staged_final\launch.json"
New-Item -ItemType Directory -Force -Path (Split-Path $LaunchPath) | Out-Null
$Launch | ConvertTo-Json | Set-Content -Encoding UTF8 $LaunchPath

Write-Host ""
Write-Host "Two visible GPU workers started." -ForegroundColor Green
Write-Host "GPU 0 PID: $($Worker0.Id)"
Write-Host "GPU 1 PID: $($Worker1.Id)"
Write-Host "The locked test remains sealed."

