param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1)]
    [int]$Device,
    [Parameter(Mandatory = $true)]
    [string]$TargetsCsv,
    [string]$Environment = "$PSScriptRoot\..\..\..\work\.venv-hemfm",
    [string]$Config = "$PSScriptRoot\..\configs\protocol.yaml",
    [int]$Epochs = 10
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path "$PSScriptRoot\..").Path
$Python = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$Config = (Resolve-Path $Config).Path
$Targets = $TargetsCsv.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)

Write-Host "HEM-FM staged final worker" -ForegroundColor Cyan
Write-Host "GPU     : $Device"
Write-Host "targets : $($Targets -join ', ')"
Write-Host "schedule: frozen -> DoRA PEFT -> selective unfreeze"
Write-Host ""

foreach ($Target in $Targets) {
    Write-Host "[$([DateTime]::Now.ToString('s'))] starting $Target on GPU $Device" -ForegroundColor Yellow
    & $Python -m hemfm --config $Config staged-final train `
        --target $Target --device $Device --epochs $Epochs
    if ($LASTEXITCODE -ne 0) {
        throw "$Target staged training failed with exit code $LASTEXITCODE"
    }
    Write-Host "[$([DateTime]::Now.ToString('s'))] completed $Target" -ForegroundColor Green
    Write-Host ""
}

Write-Host "GPU $Device worker completed every assigned target." -ForegroundColor Green

