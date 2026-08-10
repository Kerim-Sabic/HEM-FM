param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1)]
    [int]$Device,
    [Parameter(Mandatory = $true)]
    [string]$SeedsCsv,
    [string]$Environment = "$PSScriptRoot\..\..\..\work\.venv-hemfm",
    [string]$Config = "$PSScriptRoot\..\configs\protocol.yaml",
    [int]$Epochs = 8,
    [string]$ArtifactRoot = ""
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path "$PSScriptRoot\..").Path
$Python = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$Config = (Resolve-Path $Config).Path
$Seeds = $SeedsCsv.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)
if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $RunRoot = (Resolve-Path "$PSScriptRoot\..\..\..\work\hemfm-v4-runtime\runs").Path
    $ArtifactRoot = Join-Path $RunRoot "echonet_dynamic_trace"
}
else {
    $ArtifactRoot = [System.IO.Path]::GetFullPath($ArtifactRoot)
}

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

foreach ($Seed in $Seeds) {
    $Seed = [int]$Seed
    $SeedRoot = Join-Path $ArtifactRoot "seed_${Seed}_full"
    $ReportPath = Join-Path $SeedRoot "report.json"
    $LastCheckpointPath = Join-Path $SeedRoot "checkpoint_last.pt"
    $LaunchProvenancePath = Join-Path $SeedRoot "launch_provenance.json"
    $ReportExisted = Test-Path -LiteralPath $ReportPath -PathType Leaf
    if ($ReportExisted) {
        & $Python -m hemfm.worker_completion verify-seed `
            --kind trace --artifact-root $ArtifactRoot --seed $Seed --mode full `
            --requested-epochs $Epochs --config $Config --repository $Repository
        if ($LASTEXITCODE -ne 0) {
            throw "EchoNet trace seed $Seed has unbound or stale reusable artifacts."
        }
    }
    else {
        New-Item -ItemType Directory -Force -Path $SeedRoot | Out-Null
        if (Test-Path -LiteralPath $LastCheckpointPath -PathType Leaf) {
            & $Python -m hemfm.worker_completion verify-provenance `
                --config $Config --repository $Repository --input $LaunchProvenancePath
            if ($LASTEXITCODE -ne 0) {
                throw "EchoNet trace seed $Seed partial checkpoint lacks matching launch provenance."
            }
        }
        else {
            $Orphaned = @(
                "checkpoint_best.pt",
                "validation_masks_private.npz"
            ) | ForEach-Object { Join-Path $SeedRoot $_ } | Where-Object {
                Test-Path -LiteralPath $_ -PathType Leaf
            }
            if ($Orphaned.Count -gt 0) {
                throw "EchoNet trace seed $Seed has orphaned artifacts without a resumable last checkpoint."
            }
            & $Python -m hemfm.worker_completion snapshot-provenance `
                --config $Config --repository $Repository --output $LaunchProvenancePath
            if ($LASTEXITCODE -ne 0) {
                throw "EchoNet trace seed $Seed launch-provenance snapshot failed."
            }
        }
    }
    $SeedStartedUtc = [DateTime]::UtcNow.ToString("o")
    Write-Host "[$([DateTime]::Now.ToString('s'))] starting EchoNet trace seed $Seed on GPU $Device"
    & $Python -m hemfm --config $Config echonet-trace seed `
        --seed $Seed --device $Device --epochs $Epochs
    if ($LASTEXITCODE -ne 0) {
        throw "EchoNet trace seed $Seed failed with exit code $LASTEXITCODE"
    }
    if ($ReportExisted) {
        & $Python -m hemfm.worker_completion verify-seed `
            --kind trace --artifact-root $ArtifactRoot --seed $Seed --mode full `
            --requested-epochs $Epochs --config $Config --repository $Repository
    }
    else {
        & $Python -m hemfm.worker_completion record-seed `
            --kind trace --artifact-root $ArtifactRoot --seed $Seed --mode full `
            --requested-epochs $Epochs --config $Config --repository $Repository `
            --not-before $SeedStartedUtc --launch-provenance $LaunchProvenancePath
    }
    if ($LASTEXITCODE -ne 0) {
        throw "EchoNet trace seed $Seed completion manifest validation failed."
    }
}

Write-Host "GPU $Device completed its EchoNet trace seed queue."
}
finally {
    if ($null -ne $GpuLock) {
        $GpuLock.Dispose()
    }
}
