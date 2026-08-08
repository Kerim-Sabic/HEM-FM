param(
    [string]$Environment = "$PSScriptRoot\..\.venv",
    [int]$Epochs = 120
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $Environment "Scripts\python.exe"
$Jobs = @()
$Jobs += Start-Process -FilePath $Python -ArgumentList @("-m", "hemfm", "pilot", "functional", "--seed", "1103", "--device", "0", "--epochs", $Epochs, "--batch-size", "16") -WindowStyle Hidden -PassThru
$Jobs += Start-Process -FilePath $Python -ArgumentList @("-m", "hemfm", "pilot", "functional", "--seed", "2207", "--device", "1", "--epochs", $Epochs, "--batch-size", "16") -WindowStyle Hidden -PassThru
$Jobs | Wait-Process
foreach ($Job in $Jobs) {
    if ($Job.ExitCode -ne 0) { throw "Functional pilot process $($Job.Id) failed with exit code $($Job.ExitCode)" }
}
& $Python -m hemfm pilot functional --seed 3301 --device 0 --epochs $Epochs --batch-size 16
if ($LASTEXITCODE -ne 0) { throw "Functional pilot seed 3301 failed with exit code $LASTEXITCODE" }

