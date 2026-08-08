param(
    [string]$Environment = "$PSScriptRoot\..\.venv"
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path "$PSScriptRoot\..").Path
py -3.12 -m venv $Environment
$Python = Join-Path $Environment "Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
& $Python -m pip install -e "$Project[dev,train]"
Write-Host "Environment ready: $Environment"

