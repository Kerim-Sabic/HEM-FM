Exit code: 0
Wall time: 1.4 seconds
Output:
param(
    [string]$Environment = "$PSScriptRoot\..\..\..\work\.venv-hemfm",
    [string]$Config = "$PSScriptRoot\..\configs\protocol.yaml",
    [Parameter(Mandatory = $true)]
    [string]$Archive,
    [int]$CacheWorkers = 8,
    [int]$Epochs = 10
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path "$PSScriptRoot\..").Path
$Python = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$Config = (Resolve-Path $Config).Path
$Archive = (Resolve-Path $Archive).Path
$RunRoot = (Resolve-Path "$PSScriptRoot\..\..\..\work\hemfm-v4-runtime\runs").Path
$EvidenceRoot = Join-Path $Repository "local-evidence\G6"
$ScalarEvidence = @(
    "staged_final_ef_full.json",
    "staged_final_lvesv_full.json",
    "staged_final_rv_basal_diameter_full.json",
    "staged_final_lvedv_full.json",
    "staged_final_lvot_diameter_full.json",
    "staged_final_av_peak_velocity_full.json"
)

Write-Host "Waiting for every six-endpoint, three-seed scalar report..."
while ($true) {
    $Ready = $true
    foreach ($Name in $ScalarEvidence) {
        $Path = Join-Path $EvidenceRoot $Name
        if (-not (Test-Path -LiteralPath $Path)) {
            $Ready = $false
            break
        }
        $Report = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        if (-not $Report.passed) {
            throw "Scalar report did not pass its execution contract: $Name"
        }
    }
    if ($Ready) { break }
    $WorkerErrors = @(
        Get-ChildItem -LiteralPath (Join-Path $RunRoot "staged_final") -Filter "gpu*.stderr.log" |
            Select-String -Pattern "staged training failed|Traceback|RuntimeError"
    ).Count -gt 0
    if ($WorkerErrors) {
        throw "The staged scalar queue reported a fatal error. Inspect gpu stderr logs."
    }
    Start-Sleep -Seconds 30
}

Push-Location $Repository
try {
    & $Python -m pytest -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) { throw "Tests failed before EchoNet-Dynamic staging." }
    & $Python -m hemfm --config $Config echonet-dynamic stage `
        --archive $Archive --workers $CacheWorkers
    if ($LASTEXITCODE -ne 0) { throw "EchoNet-Dynamic full cache failed." }

    $WorkerScript = Join-Path $PSScriptRoot "run_echonet_dynamic_seed_worker.ps1"
    $TransferRoot = Join-Path $RunRoot "echonet_dynamic_transfer"
    New-Item -ItemType Directory -Force -Path $TransferRoot | Out-Null
    $Worker0 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $WorkerScript,
        "-Device", "0", "-SeedsCsv", "1103,3301", "-Environment", $Environment,
        "-Config", $Config, "-Epochs", $Epochs
    ) -WorkingDirectory $Repository -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $TransferRoot "gpu0.stdout.log") `
      -RedirectStandardError (Join-Path $TransferRoot "gpu0.stderr.log") -PassThru
    $Worker1 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $WorkerScript,
        "-Device", "1", "-SeedsCsv", "2207", "-Environment", $Environment,
        "-Config", $Config, "-Epochs", $Epochs
    ) -WorkingDirectory $Repository -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $TransferRoot "gpu1.stdout.log") `
      -RedirectStandardError (Join-Path $TransferRoot "gpu1.stderr.log") -PassThru
    Wait-Process -Id $Worker0.Id, $Worker1.Id
    $Worker0.Refresh()
    $Worker1.Refresh()
    if ($Worker0.ExitCode -ne 0 -or $Worker1.ExitCode -ne 0) {
        throw "An EchoNet-Dynamic seed worker failed."
    }
    & $Python -m hemfm --config $Config echonet-dynamic finalize
    if ($LASTEXITCODE -ne 0) { throw "EchoNet-Dynamic ensemble finalization failed." }

    & $Python -m hemfm --config $Config echonet-trace smoke --device 0
    if ($LASTEXITCODE -ne 0) { throw "EchoNet expert-trace segmentation smoke failed." }
    $TraceWorkerScript = Join-Path $PSScriptRoot "run_echonet_trace_seed_worker.ps1"
    $TraceRoot = Join-Path $RunRoot "echonet_dynamic_trace"
    New-Item -ItemType Directory -Force -Path $TraceRoot | Out-Null
    $TraceWorker0 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $TraceWorkerScript,
        "-Device", "0", "-SeedsCsv", "1103,3301", "-Environment", $Environment,
        "-Config", $Config, "-Epochs", "8"
    ) -WorkingDirectory $Repository -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $TraceRoot "gpu0.stdout.log") `
      -RedirectStandardError (Join-Path $TraceRoot "gpu0.stderr.log") -PassThru
    $TraceWorker1 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $TraceWorkerScript,
        "-Device", "1", "-SeedsCsv", "2207", "-Environment", $Environment,
        "-Config", $Config, "-Epochs", "8"
    ) -WorkingDirectory $Repository -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $TraceRoot "gpu1.stdout.log") `
      -RedirectStandardError (Join-Path $TraceRoot "gpu1.stderr.log") -PassThru
    Wait-Process -Id $TraceWorker0.Id, $TraceWorker1.Id
    $TraceWorker0.Refresh()
    $TraceWorker1.Refresh()
    if ($TraceWorker0.ExitCode -ne 0 -or $TraceWorker1.ExitCode -ne 0) {
        throw "An EchoNet expert-trace segmentation seed worker failed."
    }
    & $Python -m hemfm --config $Config echonet-trace finalize
    if ($LASTEXITCODE -ne 0) { throw "EchoNet expert-trace ensemble finalization failed." }
}
finally {
    Pop-Location
}

Write-Host "Post-scalar EchoNet-Dynamic scalar and traced-segmentation queue completed."

