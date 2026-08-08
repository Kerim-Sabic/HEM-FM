param(
    [switch]$AcknowledgePassedGates,
    [string]$Environment = "$PSScriptRoot\..\.venv",
    [string]$Config = "$PSScriptRoot\..\configs\protocol.yaml",
    [string]$RunDirectory = "$PSScriptRoot\..\runtime\runs\week_training"
)

$ErrorActionPreference = "Stop"
if (-not $AcknowledgePassedGates) {
    throw "Pass -AcknowledgePassedGates after reviewing G0-G5 evidence. This is not a bypass."
}
$Python = Join-Path $Environment "Scripts\python.exe"
$Repository = (Resolve-Path "$PSScriptRoot\..").Path
$Config = (Resolve-Path $Config).Path
& $Python -m hemfm --config $Config gates assert --through G5
if ($LASTEXITCODE -ne 0) { throw "Required gates have not passed" }
New-Item -ItemType Directory -Force -Path $RunDirectory | Out-Null
$Stdout = Join-Path $RunDirectory "feature-extraction.stdout.log"
$Stderr = Join-Path $RunDirectory "feature-extraction.stderr.log"
$Process = Start-Process -FilePath $Python `
    -ArgumentList @("-m", "hemfm", "--config", $Config, "specialists", "extract") `
    -WorkingDirectory $Repository `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru
$ContinuationScript = Join-Path $PSScriptRoot "continue_week_training.ps1"
$Continuation = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ContinuationScript,
        "-Environment", $Environment, "-Config", $Config,
        "-RunDirectory", $RunDirectory
    ) `
    -WorkingDirectory $Repository `
    -RedirectStandardOutput (Join-Path $RunDirectory "continuation.stdout.log") `
    -RedirectStandardError (Join-Path $RunDirectory "continuation.stderr.log") `
    -WindowStyle Hidden `
    -PassThru
@{
    schema_version = 1
    launched_utc = [DateTime]::UtcNow.ToString("o")
    process_id = $Process.Id
    continuation_process_id = $Continuation.Id
    phase = "full_feature_extraction"
    compute_location = "local workstation"
    mutable_storage = "local runtime"
    network_role = "read-only DICOM source"
    stdout = $Stdout
    stderr = $Stderr
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $RunDirectory "launch.json")
$Process

