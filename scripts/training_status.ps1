param(
    [string]$RunDirectory = "$PSScriptRoot\..\runtime\runs\week_training"
)

$Launch = Join-Path $RunDirectory "launch.json"
$Status = Join-Path $RunDirectory "status.json"
if (Test-Path $Launch) { Get-Content -Raw $Launch }
if (Test-Path $Status) { Get-Content -Raw $Status }

