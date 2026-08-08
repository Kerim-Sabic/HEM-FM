param(
    [string]$Environment = "$PSScriptRoot\..\.venv",
    [string]$Config = "$PSScriptRoot\..\configs\protocol.yaml",
    [string]$RunDirectory = "$PSScriptRoot\..\runtime\runs\week_training"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Environment "Scripts\python.exe"
$Config = (Resolve-Path $Config).Path
$StatusPath = Join-Path $RunDirectory "status.json"
$Stdout = Join-Path $RunDirectory "specialist-training.stdout.log"
$Stderr = Join-Path $RunDirectory "specialist-training.stderr.log"

while ($true) {
    if (Test-Path $StatusPath) {
        $Status = Get-Content -Raw $StatusPath | ConvertFrom-Json
        if ($Status.phase -eq "feature_extraction_complete") { break }
        if ($Status.phase -eq "feature_extraction_failed") {
            throw "Feature extraction failed; specialist training remains fail-closed."
        }
    }
    Start-Sleep -Seconds 60
}

& $Python -m hemfm --config $Config specialists train 1>> $Stdout 2>> $Stderr
if ($LASTEXITCODE -ne 0) {
    throw "Specialist training failed with exit code $LASTEXITCODE"
}

