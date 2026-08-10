param(
    [Parameter(Mandatory = $true)]
    [string]$Environment,
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1)]
    [int]$Device,
    [Parameter(Mandatory = $true)]
    [string]$TargetList
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Environment "Scripts\python.exe"
$Repository = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Config)) {
    throw "Protocol config not found: $Config"
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

    Push-Location $Repository
    try {
        $Targets = $TargetList.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)
        if ($Targets.Count -eq 0) {
            throw "At least one staged target is required."
        }
        foreach ($Target in $Targets) {
            & $Python -m hemfm --config $Config staged-final audit `
                --target $Target --device $Device
            if ($LASTEXITCODE -ne 0) {
                throw "Matched staged uncertainty audit failed for $Target on GPU $Device."
            }
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "GPU $Device completed its matched staged audit group."
}
finally {
    if ($null -ne $GpuLock) {
        $GpuLock.Dispose()
    }
}
