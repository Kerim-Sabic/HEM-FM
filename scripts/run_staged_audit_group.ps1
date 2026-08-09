param(
    [Parameter(Mandatory = $true)]
    [string]$Environment,
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
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
