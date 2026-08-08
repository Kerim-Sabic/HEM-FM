param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1)]
    [int]$Device,
    [Parameter(Mandatory = $true)]
    [string]$SeedsCsv,
    [string]$Environment = "$PSScriptRoot\..\..\..\work\.venv-hemfm",
    [string]$Config = "$PSScriptRoot\..\configs\protocol.yaml",
    [int]$Epochs = 10
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path "$PSScriptRoot\..").Path
$Python = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$Config = (Resolve-Path $Config).Path
$Seeds = $SeedsCsv.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)

foreach ($Seed in $Seeds) {
    Write-Host "[$([DateTime]::Now.ToString('s'))] starting EchoNet-Dynamic seed $Seed on GPU $Device"
    & $Python -m hemfm --config $Config echonet-dynamic seed `
        --seed ([int]$Seed) --device $Device --epochs $Epochs
    if ($LASTEXITCODE -ne 0) {
        throw "EchoNet-Dynamic seed $Seed failed with exit code $LASTEXITCODE"
    }
}

Write-Host "GPU $Device completed its EchoNet-Dynamic seed queue."
