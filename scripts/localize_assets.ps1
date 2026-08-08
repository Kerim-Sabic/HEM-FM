param(
    [Parameter(Mandatory = $true)]
    [string]$SourceEchoJEPA,
    [Parameter(Mandatory = $true)]
    [string]$SourceCheckpoint,
    [Parameter(Mandatory = $true)]
    [string]$SourceFeatures,
    [string]$SourceOldRun,
    [string]$RuntimeRoot = "$PSScriptRoot\..\runtime"
)

$ErrorActionPreference = "Stop"

$Vendor = Join-Path $RuntimeRoot "vendor\EchoJEPA"
$Checkpoints = Join-Path $RuntimeRoot "checkpoints"
$Features = Join-Path $RuntimeRoot "feature-cache\mimic_echojepa_full"
$Private = Join-Path $RuntimeRoot "private"
$Runs = Join-Path $RuntimeRoot "runs"

New-Item -ItemType Directory -Force -Path $Vendor, $Checkpoints, $Features, $Private, $Runs | Out-Null
Copy-Item -Path "$SourceEchoJEPA\*" -Destination $Vendor -Recurse -Force
Copy-Item -LiteralPath $SourceCheckpoint -Destination (Join-Path $Checkpoints "vitl-vmix22m-pt220-c55.pt") -Force
Copy-Item -Path "$SourceFeatures\*" -Destination $Features -Recurse -Force

if ($SourceOldRun) {
    $OldInventory = Join-Path $SourceOldRun "private\dicom_inventory.csv"
    if (Test-Path -LiteralPath $OldInventory) {
        Copy-Item -LiteralPath $OldInventory -Destination (Join-Path $Private "dicom_inventory.csv") -Force
    }
    $OldPilots = Join-Path $SourceOldRun "pilots"
    if (Test-Path -LiteralPath $OldPilots) {
        Copy-Item -LiteralPath $OldPilots -Destination $Runs -Recurse -Force
    }
}

$Checkpoint = Join-Path $Checkpoints "vitl-vmix22m-pt220-c55.pt"
$TokenCache = Join-Path $Features "tokens_fp16.npy"
$Manifest = [ordered]@{
    schema_version = 1
    storage = "local"
    runtime_root = $RuntimeRoot
    functional_checkpoint = [ordered]@{
        path = $Checkpoint
        bytes = (Get-Item -LiteralPath $Checkpoint).Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Checkpoint).Hash.ToLowerInvariant()
    }
    token_cache = [ordered]@{
        path = $TokenCache
        bytes = (Get-Item -LiteralPath $TokenCache).Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $TokenCache).Hash.ToLowerInvariant()
    }
}
$Manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $RuntimeRoot "localization-manifest.json") -Encoding UTF8
Write-Host "Local assets ready at $RuntimeRoot"

