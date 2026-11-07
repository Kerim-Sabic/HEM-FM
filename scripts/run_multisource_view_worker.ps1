param(
    [ValidateSet(0, 1)]
    [int]$Device = 1,
    [string]$Environment = "",
    [string]$Runtime = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path "$PSScriptRoot\..").Path
if ([string]::IsNullOrWhiteSpace($Environment)) {
    $Environment = Join-Path $Repository "..\..\work\.venv-hemfm"
}
if ([string]::IsNullOrWhiteSpace($Runtime)) {
    $Runtime = Join-Path $Repository "..\..\work\hemfm-v4-runtime"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $Runtime "runs\multisource_view_dinov3_full_v1"
}

$Python = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$Runtime = (Resolve-Path $Runtime).Path
$Config = Join-Path $Repository "configs\protocol.yaml"
$Output = [System.IO.Path]::GetFullPath($Output)
$env:PYTHONPATH = Join-Path $Repository "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$GpuLock = $null

function Assert-StrictFalse($Value, [string]$Name) {
    if ($Value -isnot [bool] -or $Value -ne $false) {
        throw "$Name must be the JSON boolean false."
    }
}

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

    $Report = Join-Path $Output "report.json"
    $Manifest = Join-Path $Output "manifest.json"
    $History = Join-Path $Output "history.json"
    $Best = Join-Path $Output "checkpoint_best.pt"
    $Last = Join-Path $Output "checkpoint_last.pt"
    if (Test-Path -LiteralPath $Report -PathType Leaf) {
        throw "A completed report already exists at '$Report'; refusing to overwrite evidence."
    }
    New-Item -ItemType Directory -Force -Path $Output | Out-Null

    Write-Host "[$([DateTime]::Now.ToString('s'))] starting Unity+EV9V full training on GPU $Device"
    & $Python -m hemfm.multisource_view `
        --config $Config `
        --output-dir $Output `
        --device $Device `
        --mode full
    if ($LASTEXITCODE -ne 0) {
        throw "Unity+EV9V training failed with exit code $LASTEXITCODE."
    }

    foreach ($Artifact in @($Report, $Manifest, $History, $Best, $Last)) {
        if (-not (Test-Path -LiteralPath $Artifact -PathType Leaf) -or
            (Get-Item -LiteralPath $Artifact).Length -le 0) {
            throw "Required completion artifact is missing or empty: $Artifact"
        }
    }
    $Completed = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    if ($Completed.mode -ne "full" -or
        $Completed.scope -ne "unity_ev9v_development_bmode_view_classifier") {
        throw "Completion report has the wrong mode or scope."
    }
    Assert-StrictFalse $Completed.locked_test_accessed "locked_test_accessed"
    Assert-StrictFalse $Completed.unity_ival_pixels_accessed "unity_ival_pixels_accessed"
    Assert-StrictFalse $Completed.ev9v_test_pixels_accessed "ev9v_test_pixels_accessed"
    Assert-StrictFalse $Completed.automatic_measurement_routing_allowed "automatic_measurement_routing_allowed"
    Write-Host "MULTISOURCE VIEW WORKER COMPLETE: GPU $Device; locked TEST remained sealed"
}
finally {
    if ($null -ne $GpuLock) {
        $GpuLock.Dispose()
    }
}
