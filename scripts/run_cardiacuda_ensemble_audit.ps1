param(
    [string]$Environment = "",
    [string]$Dataset = "",
    [string]$RunRoot = "",
    [string]$Output = "",
    [ValidateSet(0, 1)]
    [int]$Device = 0
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path "$PSScriptRoot\..").Path
if ([string]::IsNullOrWhiteSpace($Environment)) {
    $Environment = Join-Path $Repository "..\..\work\.venv-hemfm"
}
if ([string]::IsNullOrWhiteSpace($Dataset)) {
    $Dataset = Join-Path $Repository "..\..\work\hemfm-v4-runtime\datasets\research\cardiacuda\extracted\cardiacUDC_dataset"
}
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $Repository "..\..\work\hemfm-v4-runtime\runs"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $RunRoot "cardiacuda_segmentation_ensemble_dev_audit_v1"
}
$Python = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$Dataset = (Resolve-Path $Dataset).Path
$RunRoot = (Resolve-Path $RunRoot).Path
$env:PYTHONPATH = Join-Path $Repository "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$GpuLock = $null

try {
    $GpuLockDirectory = Join-Path $Repository "work\gpu-locks"
    New-Item -ItemType Directory -Force -Path $GpuLockDirectory | Out-Null
    $GpuLockPath = Join-Path $GpuLockDirectory "gpu_${Device}.lock"
    try {
        $GpuLock = [System.IO.File]::Open(
            $GpuLockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "GPU $Device is already reserved by another HEM-FM worker ('$GpuLockPath')."
    }

    Write-Host "[$([DateTime]::Now.ToString('s'))] starting CardiacUDA ensemble audit on GPU $Device"
    & $Python -m hemfm.cardiacuda_ensemble `
        --root $Dataset `
        --run-root $RunRoot `
        --output-dir $Output `
        --device "cuda:$Device"
    if ($LASTEXITCODE -ne 0) {
        throw "CardiacUDA ensemble audit failed with exit code $LASTEXITCODE"
    }
    $ReportPath = Join-Path $Output "report.json"
    $FailurePath = Join-Path $Output "failure_cases.json"
    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $FailurePath -PathType Leaf)) {
        throw "CardiacUDA ensemble audit returned without complete evidence."
    }
    $Report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
    if ($Report.calibration_audit_complete -isnot [bool] -or
        $Report.calibration_audit_complete -ne $true -or
        $Report.sealed_test_accessed -isnot [bool] -or
        $Report.sealed_test_accessed -ne $false -or
        $Report.promotion_eligible -isnot [bool] -or
        $Report.promotion_eligible -ne $false) {
        throw "CardiacUDA ensemble audit safety evidence is invalid."
    }
    Write-Host "CARDIACUDA ENSEMBLE AUDIT COMPLETE: GPU $Device"
}
finally {
    if ($null -ne $GpuLock) {
        $GpuLock.Dispose()
    }
}
