param(
    [string]$Environment = "$PSScriptRoot\..\..\..\work\.venv-hemfm",
    [string]$Config = "$PSScriptRoot\..\configs\protocol.yaml",
    [Parameter(Mandatory = $true)]
    [string]$Archive,
    [int]$CacheWorkers = 8,
    [int]$Epochs = 10
)

$ErrorActionPreference = "Stop"

function Assert-NoActiveEchoNetWorkers {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [switch]$IncludeStagedScalarWorkers
    )

    $WorkerMarkers = @(
        "run_echonet_dynamic_seed_worker.ps1",
        "run_echonet_trace_seed_worker.ps1",
        "echonet-dynamic seed",
        "echonet-trace seed",
        "run_staged_audit_group.ps1",
        "staged-final audit"
    )
    if ($IncludeStagedScalarWorkers) {
        $WorkerMarkers += @(
            "run_staged_scalar_worker.ps1",
            "staged-final train"
        )
    }
    $Active = @(
        Get-CimInstance Win32_Process | Where-Object {
            if ($_.ProcessId -eq $PID -or [string]::IsNullOrWhiteSpace($_.CommandLine)) {
                return $false
            }
            $CommandLine = [string]$_.CommandLine
            $MatchesRepository = $CommandLine.IndexOf(
                $Repository, [System.StringComparison]::OrdinalIgnoreCase
            ) -ge 0
            $MatchesConfig = $CommandLine.IndexOf(
                $Config, [System.StringComparison]::OrdinalIgnoreCase
            ) -ge 0
            $MatchesWorker = $false
            foreach ($Marker in $WorkerMarkers) {
                if ($CommandLine.IndexOf($Marker, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                    $MatchesWorker = $true
                    break
                }
            }
            $MatchesRepository -and $MatchesConfig -and $MatchesWorker
        }
    )
    if ($Active.Count -gt 0) {
        $ProcessIds = ($Active | ForEach-Object { [string]$_.ProcessId }) -join ", "
        throw "Refusing to launch duplicate EchoNet workers for this repository/config. Active PIDs: $ProcessIds"
    }
}

function Open-HemFmGpuLock {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [ValidateSet(0, 1)]
        [int]$Device
    )

    $GpuLockDirectory = Join-Path $Repository "work\gpu-locks"
    New-Item -ItemType Directory -Force -Path $GpuLockDirectory | Out-Null
    $GpuLockPath = Join-Path $GpuLockDirectory "gpu_${Device}.lock"
    try {
        return [System.IO.File]::Open(
            $GpuLockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "GPU $Device is already reserved by another HEM-FM worker ('$GpuLockPath')."
    }
}

function Assert-EchoNetWorkerComplete {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [string]$WorkerName,
        [Parameter(Mandatory = $true)]
        [string]$StdoutPath,
        [Parameter(Mandatory = $true)]
        [string]$StderrPath,
        [Parameter(Mandatory = $true)]
        [string]$CompletionMarker,
        [Parameter(Mandatory = $true)]
        [int[]]$Seeds,
        [Parameter(Mandatory = $true)]
        [string]$ArtifactRoot,
        [Parameter(Mandatory = $true)]
        [ValidateSet("transfer", "trace")]
        [string]$ArtifactKind,
        [Parameter(Mandatory = $true)]
        [int]$RequestedEpochs,
        [Parameter(Mandatory = $true)]
        [datetime]$NotBeforeUtc,
        [Parameter(Mandatory = $true)]
        [string]$Python,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$Repository
    )

    if (-not $Process.HasExited) {
        throw "$WorkerName is still running after the worker wait completed."
    }

    # Windows PowerShell 5.1 can return a Process object whose ExitCode is null
    # when Start-Process was used without -Wait. Treat a known nonzero code as
    # fatal, but never treat a null code as success without fresh logs and a
    # complete, sealed artifact set below.
    $ExitCode = $null
    try {
        $ExitCode = $Process.ExitCode
    }
    catch {
        $ExitCode = $null
    }
    if ($null -ne $ExitCode -and $ExitCode -ne 0) {
        throw "$WorkerName failed with exit code $ExitCode."
    }

    if (-not (Test-Path -LiteralPath $StdoutPath -PathType Leaf)) {
        throw "$WorkerName did not create its stdout log."
    }
    if ((Get-Item -LiteralPath $StdoutPath).LastWriteTimeUtc -lt $NotBeforeUtc.AddSeconds(-2)) {
        throw "$WorkerName stdout log predates this launch."
    }
    if (-not (Select-String -LiteralPath $StdoutPath -SimpleMatch $CompletionMarker -Quiet)) {
        throw "$WorkerName did not emit its completion marker."
    }
    if (-not (Test-Path -LiteralPath $StderrPath -PathType Leaf)) {
        throw "$WorkerName did not create its stderr log."
    }
    if ((Get-Item -LiteralPath $StderrPath).LastWriteTimeUtc -lt $NotBeforeUtc.AddSeconds(-2)) {
        throw "$WorkerName stderr log predates this launch."
    }
    $FatalError = Select-String -LiteralPath $StderrPath -Pattern @(
        "Traceback \(most recent call last\):",
        "RuntimeError:",
        "OutOfMemoryError:",
        "CUDA out of memory",
        "seed .* failed",
        "failed with exit code",
        "TerminatingError"
    ) -Quiet
    if ($FatalError) {
        throw "$WorkerName stderr contains a fatal error signature."
    }

    foreach ($Seed in $Seeds) {
        & $Python -m hemfm.worker_completion verify-seed `
            --kind $ArtifactKind --artifact-root $ArtifactRoot --seed $Seed --mode full `
            --requested-epochs $RequestedEpochs --config $Config --repository $Repository
        if ($LASTEXITCODE -ne 0) {
            throw "$WorkerName seed $Seed failed hash-bound completion-manifest validation."
        }
    }

    if ($null -eq $ExitCode) {
        Write-Host "$WorkerName exit code was unavailable; fresh logs and sealed artifacts passed fail-closed validation." -ForegroundColor Yellow
    }
}

$Repository = (Resolve-Path "$PSScriptRoot\..").Path
$Python = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$Config = (Resolve-Path $Config).Path
$Archive = (Resolve-Path $Archive).Path
$RunRoot = (Resolve-Path "$PSScriptRoot\..\..\..\work\hemfm-v4-runtime\runs").Path
$PytestRoot = Join-Path $Repository "work"
New-Item -ItemType Directory -Force -Path $PytestRoot | Out-Null
$PytestTemp = Join-Path $PytestRoot "pytest-post-scalar-$PID"
$EvidenceRoot = Join-Path $Repository "local-evidence\G6"
$ScalarEvidence = @(
    "staged_final_ef_full.json",
    "staged_final_lvesv_full.json",
    "staged_final_rv_basal_diameter_full.json",
    "staged_final_lvedv_full.json",
    "staged_final_lvot_diameter_full.json",
    "staged_final_av_peak_velocity_full.json"
)

$LauncherLock = $null
try {
    $LockDirectory = Join-Path $RunRoot "post_scalar_queue"
    New-Item -ItemType Directory -Force -Path $LockDirectory | Out-Null
    $LockPath = Join-Path $LockDirectory "launcher.lock"
    try {
        $LauncherLock = [System.IO.File]::Open(
            $LockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "Another post-scalar launcher owns the exclusive lock '$LockPath'."
    }
    Assert-NoActiveEchoNetWorkers -Repository $Repository -Config $Config

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
        if (
            $Report.PSObject.Properties.Name -notcontains "passed" -or
            $Report.passed -isnot [bool] -or
            -not $Report.passed
        ) {
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
    Assert-NoActiveEchoNetWorkers -Repository $Repository -Config $Config `
        -IncludeStagedScalarWorkers
    $AuditTargets = @(
        "EF",
        "LVEDV",
        "LVESV",
        "LVOT_DIAMETER",
        "RV_BASAL_DIAMETER",
        "AV_PEAK_VELOCITY"
    )
    $AuditGpuLocks = @()
    try {
        foreach ($Device in @(0, 1)) {
            $AuditGpuLocks += Open-HemFmGpuLock -Repository $Repository -Device $Device
        }
        for ($Index = 0; $Index -lt $AuditTargets.Count; $Index++) {
            $Target = $AuditTargets[$Index]
            $Device = $Index % 2
            & $Python -m hemfm --config $Config staged-final audit `
                --target $Target --device $Device
            if ($LASTEXITCODE -ne 0) {
                throw "Matched staged uncertainty audit failed to execute for $Target."
            }
        }
    }
    finally {
        foreach ($GpuLock in $AuditGpuLocks) {
            $GpuLock.Dispose()
        }
    }
    & $Python -m pytest -q -p no:cacheprovider --basetemp $PytestTemp
    if ($LASTEXITCODE -ne 0) { throw "Tests failed before EchoNet-Dynamic staging." }
    & $Python -m hemfm --config $Config echonet-dynamic stage `
        --archive $Archive --workers $CacheWorkers
    if ($LASTEXITCODE -ne 0) { throw "EchoNet-Dynamic full cache failed." }

    Assert-NoActiveEchoNetWorkers -Repository $Repository -Config $Config `
        -IncludeStagedScalarWorkers
    $WorkerScript = Join-Path $PSScriptRoot "run_echonet_dynamic_seed_worker.ps1"
    $TransferRoot = Join-Path $RunRoot "echonet_dynamic_transfer"
    New-Item -ItemType Directory -Force -Path $TransferRoot | Out-Null
    $TransferLaunchUtc = [DateTime]::UtcNow
    $Worker0 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $WorkerScript,
        "-Device", "0", "-SeedsCsv", "1103,3301", "-Environment", $Environment,
        "-Config", $Config, "-Epochs", $Epochs, "-ArtifactRoot", $TransferRoot
    ) -WorkingDirectory $Repository -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $TransferRoot "gpu0.stdout.log") `
      -RedirectStandardError (Join-Path $TransferRoot "gpu0.stderr.log") -PassThru
    $Worker1 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $WorkerScript,
        "-Device", "1", "-SeedsCsv", "2207", "-Environment", $Environment,
        "-Config", $Config, "-Epochs", $Epochs, "-ArtifactRoot", $TransferRoot
    ) -WorkingDirectory $Repository -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $TransferRoot "gpu1.stdout.log") `
      -RedirectStandardError (Join-Path $TransferRoot "gpu1.stderr.log") -PassThru
    Wait-Process -Id $Worker0.Id, $Worker1.Id
    $Worker0.Refresh()
    $Worker1.Refresh()
    Assert-EchoNetWorkerComplete -Process $Worker0 -WorkerName "EchoNet-Dynamic GPU 0 worker" `
        -StdoutPath (Join-Path $TransferRoot "gpu0.stdout.log") `
        -StderrPath (Join-Path $TransferRoot "gpu0.stderr.log") `
        -CompletionMarker "GPU 0 completed its EchoNet-Dynamic seed queue." `
        -Seeds @(1103, 3301) -ArtifactRoot $TransferRoot -ArtifactKind "transfer" `
        -RequestedEpochs $Epochs -NotBeforeUtc $TransferLaunchUtc `
        -Python $Python -Config $Config -Repository $Repository
    Assert-EchoNetWorkerComplete -Process $Worker1 -WorkerName "EchoNet-Dynamic GPU 1 worker" `
        -StdoutPath (Join-Path $TransferRoot "gpu1.stdout.log") `
        -StderrPath (Join-Path $TransferRoot "gpu1.stderr.log") `
        -CompletionMarker "GPU 1 completed its EchoNet-Dynamic seed queue." `
        -Seeds @(2207) -ArtifactRoot $TransferRoot -ArtifactKind "transfer" `
        -RequestedEpochs $Epochs -NotBeforeUtc $TransferLaunchUtc `
        -Python $Python -Config $Config -Repository $Repository
    & $Python -m hemfm --config $Config echonet-dynamic finalize
    if ($LASTEXITCODE -ne 0) { throw "EchoNet-Dynamic ensemble finalization failed." }

    $TraceSmokeGpuLock = Open-HemFmGpuLock -Repository $Repository -Device 0
    try {
        & $Python -m hemfm --config $Config echonet-trace smoke --device 0
        if ($LASTEXITCODE -ne 0) { throw "EchoNet expert-trace segmentation smoke failed." }
    }
    finally {
        $TraceSmokeGpuLock.Dispose()
    }
    Assert-NoActiveEchoNetWorkers -Repository $Repository -Config $Config `
        -IncludeStagedScalarWorkers
    $TraceWorkerScript = Join-Path $PSScriptRoot "run_echonet_trace_seed_worker.ps1"
    $TraceRoot = Join-Path $RunRoot "echonet_dynamic_trace"
    New-Item -ItemType Directory -Force -Path $TraceRoot | Out-Null
    $TraceLaunchUtc = [DateTime]::UtcNow
    $TraceWorker0 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $TraceWorkerScript,
        "-Device", "0", "-SeedsCsv", "1103,3301", "-Environment", $Environment,
        "-Config", $Config, "-Epochs", "8", "-ArtifactRoot", $TraceRoot
    ) -WorkingDirectory $Repository -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $TraceRoot "gpu0.stdout.log") `
      -RedirectStandardError (Join-Path $TraceRoot "gpu0.stderr.log") -PassThru
    $TraceWorker1 = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $TraceWorkerScript,
        "-Device", "1", "-SeedsCsv", "2207", "-Environment", $Environment,
        "-Config", $Config, "-Epochs", "8", "-ArtifactRoot", $TraceRoot
    ) -WorkingDirectory $Repository -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $TraceRoot "gpu1.stdout.log") `
      -RedirectStandardError (Join-Path $TraceRoot "gpu1.stderr.log") -PassThru
    Wait-Process -Id $TraceWorker0.Id, $TraceWorker1.Id
    $TraceWorker0.Refresh()
    $TraceWorker1.Refresh()
    Assert-EchoNetWorkerComplete -Process $TraceWorker0 `
        -WorkerName "EchoNet expert-trace GPU 0 worker" `
        -StdoutPath (Join-Path $TraceRoot "gpu0.stdout.log") `
        -StderrPath (Join-Path $TraceRoot "gpu0.stderr.log") `
        -CompletionMarker "GPU 0 completed its EchoNet trace seed queue." `
        -Seeds @(1103, 3301) -ArtifactRoot $TraceRoot -ArtifactKind "trace" `
        -RequestedEpochs 8 -NotBeforeUtc $TraceLaunchUtc `
        -Python $Python -Config $Config -Repository $Repository
    Assert-EchoNetWorkerComplete -Process $TraceWorker1 `
        -WorkerName "EchoNet expert-trace GPU 1 worker" `
        -StdoutPath (Join-Path $TraceRoot "gpu1.stdout.log") `
        -StderrPath (Join-Path $TraceRoot "gpu1.stderr.log") `
        -CompletionMarker "GPU 1 completed its EchoNet trace seed queue." `
        -Seeds @(2207) -ArtifactRoot $TraceRoot -ArtifactKind "trace" `
        -RequestedEpochs 8 -NotBeforeUtc $TraceLaunchUtc `
        -Python $Python -Config $Config -Repository $Repository
    & $Python -m hemfm --config $Config echonet-trace finalize
    if ($LASTEXITCODE -ne 0) { throw "EchoNet expert-trace ensemble finalization failed." }
    & $Python -m hemfm --config $Config g7 audit
    if ($LASTEXITCODE -notin @(0, 2)) { throw "G7 integration audit failed to execute." }
    & $Python -m hemfm --config $Config gates status
    if ($LASTEXITCODE -ne 0) { throw "Final gate status failed to execute." }
    }
    finally {
        Pop-Location
    }

    Write-Host "Post-scalar EchoNet-Dynamic scalar and traced-segmentation queue completed."
}
finally {
    if ($null -ne $LauncherLock) {
        $LauncherLock.Dispose()
    }
}
