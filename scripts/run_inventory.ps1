param(
    [string]$Environment = "$PSScriptRoot\..\.venv",
    [int]$Workers = 16
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Environment "Scripts\python.exe"
& $Python -m hemfm inventory labels
if ($LASTEXITCODE -ne 0) { throw "Label inventory failed with exit code $LASTEXITCODE" }
& $Python -m hemfm inventory dicom --workers $Workers --resume
if ($LASTEXITCODE -ne 0) { throw "DICOM inventory failed with exit code $LASTEXITCODE" }
& $Python -m hemfm calibration validate --max-files 512
if ($LASTEXITCODE -ne 0) { throw "Physical-calibration validation failed with exit code $LASTEXITCODE" }

