param(
    [string]$Environment = "$PSScriptRoot\..\.venv"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Environment "Scripts\python.exe"
& $Python -m hemfm runtime smoke
if ($LASTEXITCODE -ne 0) { throw "GPU/checkpoint smoke test failed with exit code $LASTEXITCODE" }
$BaseTemp = "$PSScriptRoot\..\runtime\pytest-hemfm"
New-Item -ItemType Directory -Force -Path $BaseTemp | Out-Null
& $Python -m pytest -q -p no:cacheprovider --basetemp $BaseTemp "$PSScriptRoot\..\tests"
if ($LASTEXITCODE -ne 0) { throw "Safety tests failed with exit code $LASTEXITCODE" }
& $Python -m hemfm splits audit
if ($LASTEXITCODE -ne 0) { throw "Patient-split audit failed with exit code $LASTEXITCODE" }
Write-Host "G0 runtime and the reusable G1 split/calibration unit checks completed."

