param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1)]
    [int]$Device,
    [Parameter(Mandatory = $true)]
    [string]$SeedsCsv,
    [string]$Environment = "",
    [string]$Dataset = "",
    [string]$RunRoot = ""
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
$Python = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$Dataset = (Resolve-Path $Dataset).Path
$RunRoot = (Resolve-Path $RunRoot).Path
$Seeds = $SeedsCsv.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)
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

    foreach ($SeedText in $Seeds) {
        $Seed = [int]$SeedText
        $Output = Join-Path $RunRoot "cardiacuda_segmentation_seed_${Seed}_full"
        $Report = Join-Path $Output "report.json"
        $Best = Join-Path $Output "best.pt"
        $Last = Join-Path $Output "last.pt"
        if (Test-Path -LiteralPath $Report -PathType Leaf) {
            $Existing = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
            if ($Existing.mode -ne "full" -or [int]$Existing.seed -ne $Seed -or
                $Existing.sealed_test_accessed -isnot [bool] -or
                $Existing.sealed_test_accessed -ne $false -or
                -not (Test-Path -LiteralPath $Best -PathType Leaf) -or
                -not (Test-Path -LiteralPath $Last -PathType Leaf)) {
                throw "Seed $Seed has an invalid or incomplete existing artifact set."
            }
            Write-Host "[$([DateTime]::Now.ToString('s'))] seed $Seed already complete and sealed"
            continue
        }

        $Arguments = @(
            "-m", "hemfm.cardiacuda_segmentation", "train",
            "--root", $Dataset,
            "--output-dir", $Output,
            "--mode", "full",
            "--seed", [string]$Seed,
            "--device", "cuda:$Device",
            "--validation-domain", "Site_R"
        )
        if (Test-Path -LiteralPath $Last -PathType Leaf) {
            $Arguments += @("--resume", $Last)
            Write-Host "[$([DateTime]::Now.ToString('s'))] resuming chamber seed $Seed on GPU $Device"
        }
        else {
            Write-Host "[$([DateTime]::Now.ToString('s'))] starting chamber seed $Seed on GPU $Device"
        }
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "CardiacUDA chamber seed $Seed failed with exit code $LASTEXITCODE"
        }
        if (-not (Test-Path -LiteralPath $Report -PathType Leaf) -or
            -not (Test-Path -LiteralPath $Best -PathType Leaf) -or
            -not (Test-Path -LiteralPath $Last -PathType Leaf)) {
            throw "CardiacUDA chamber seed $Seed returned without complete artifacts."
        }
        $Completed = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
        if ($Completed.mode -ne "full" -or [int]$Completed.seed -ne $Seed -or
            $Completed.sealed_test_accessed -isnot [bool] -or
            $Completed.sealed_test_accessed -ne $false) {
            throw "CardiacUDA chamber seed $Seed completion report failed the TEST-seal checks."
        }
    }
    Write-Host "CARDIACUDA WORKER COMPLETE: GPU $Device seeds $SeedsCsv"
}
finally {
    if ($null -ne $GpuLock) {
        $GpuLock.Dispose()
    }
}
