param(
    [Parameter(Mandatory = $true)]
    [int]$QueueProcessId,
    [Parameter(Mandatory = $true)]
    [string]$Environment,
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [string]$LogDirectory,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

function Assert-NoActiveMatchedAuditWorkers {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [switch]$IncludeAllGpuWorkloads
    )

    $Markers = @("run_staged_audit_group.ps1", "staged-final audit")
    if ($IncludeAllGpuWorkloads) {
        $Markers += @(
            "run_echonet_dynamic_seed_worker.ps1",
            "run_echonet_trace_seed_worker.ps1",
            "run_staged_scalar_worker.ps1",
            "echonet-dynamic seed",
            "echonet-trace seed",
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
            foreach ($Marker in $Markers) {
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
        throw "Refusing to launch while conflicting HEM-FM GPU workers are active for this repository/config. Active PIDs: $ProcessIds"
    }
}

function Assert-MatchedAuditWorkerComplete {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)]
        [hashtable]$Group,
        [Parameter(Mandatory = $true)]
        [datetime]$NotBeforeUtc,
        [Parameter(Mandatory = $true)]
        [string]$Python,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [string]$EvidenceRoot
    )

    $WorkerName = "Matched staged audit GPU $($Group.Device) worker"
    if (-not $Process.HasExited) {
        throw "$WorkerName is still running after the worker wait completed."
    }
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

    foreach ($LogPath in @($Group.Stdout, $Group.Stderr)) {
        if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
            throw "$WorkerName did not create required log '$LogPath'."
        }
        if ((Get-Item -LiteralPath $LogPath).LastWriteTimeUtc -lt $NotBeforeUtc.AddSeconds(-2)) {
            throw "$WorkerName log '$LogPath' predates this launch."
        }
    }
    $CompletionMarker = "GPU $($Group.Device) completed its matched staged audit group."
    if (-not (Select-String -LiteralPath $Group.Stdout -SimpleMatch $CompletionMarker -Quiet)) {
        throw "$WorkerName did not emit its completion marker."
    }
    $FatalError = Select-String -LiteralPath $Group.Stderr -Pattern @(
        "Traceback \(most recent call last\):",
        "RuntimeError:",
        "OutOfMemoryError:",
        "CUDA out of memory",
        "Matched staged uncertainty audit failed",
        "TerminatingError"
    ) -Quiet
    if ($FatalError) {
        throw "$WorkerName stderr contains a fatal error signature."
    }

    foreach ($Target in $Group.Targets.Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)) {
        & $Python -m hemfm.worker_completion verify-matched-audit `
            --evidence-root $EvidenceRoot --target $Target --mode full `
            --config $Config --repository $Repository `
            --not-before $NotBeforeUtc.ToString("o")
        if ($LASTEXITCODE -ne 0) {
            throw "$WorkerName target $Target failed fresh matched-evidence validation."
        }
    }
    if ($null -eq $ExitCode) {
        Write-Host "$WorkerName exit code was unavailable; fresh logs and matched evidence passed fail-closed validation." -ForegroundColor Yellow
    }
}

$Environment = (Resolve-Path $Environment).Path
$Config = (Resolve-Path $Config).Path
$Python = Join-Path $Environment "Scripts\python.exe"
$Repository = Split-Path -Parent $PSScriptRoot
$Worker = Join-Path $PSScriptRoot "run_staged_audit_group.ps1"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Config)) {
    throw "Protocol config not found: $Config"
}
if (-not (Test-Path -LiteralPath $Worker)) {
    throw "Audit worker not found: $Worker"
}
if ($PollSeconds -lt 5) {
    throw "PollSeconds must be at least 5."
}

$CoordinatorLock = $null
try {
    $LockDirectory = Join-Path $Repository "work"
    New-Item -ItemType Directory -Path $LockDirectory -Force | Out-Null
    $LockPath = Join-Path $LockDirectory "matched-audit-coordinator.lock"
    try {
        $CoordinatorLock = [System.IO.File]::Open(
            $LockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "Another matched-audit coordinator owns the exclusive lock '$LockPath'."
    }
    Assert-NoActiveMatchedAuditWorkers -Repository $Repository -Config $Config
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

    while (Get-Process -Id $QueueProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds $PollSeconds
    }

    Assert-NoActiveMatchedAuditWorkers -Repository $Repository -Config $Config `
        -IncludeAllGpuWorkloads
    $PowerShell = (Get-Process -Id $PID).Path
    $LaunchProvenancePath = Join-Path $LogDirectory "launch_provenance.json"
    & $Python -m hemfm.worker_completion snapshot-provenance `
        --config $Config --repository $Repository --output $LaunchProvenancePath
    if ($LASTEXITCODE -ne 0) {
        throw "Matched-audit launch-provenance snapshot failed."
    }
    $Groups = @(
    @{
        Device = 0
        Targets = "EF,LVESV,RV_BASAL_DIAMETER"
        Stdout = Join-Path $LogDirectory "gpu0.stdout.log"
        Stderr = Join-Path $LogDirectory "gpu0.stderr.log"
    },
    @{
        Device = 1
        Targets = "LVEDV,LVOT_DIAMETER,AV_PEAK_VELOCITY"
        Stdout = Join-Path $LogDirectory "gpu1.stdout.log"
        Stderr = Join-Path $LogDirectory "gpu1.stderr.log"
    }
    )

    $LaunchUtc = [DateTime]::UtcNow
    $Processes = foreach ($Group in $Groups) {
    $Arguments = @(
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-File", $Worker,
        "-Environment", $Environment,
        "-Config", $Config,
        "-Device", [string]$Group.Device,
        "-TargetList", $Group.Targets
    )
    Start-Process -FilePath $PowerShell `
        -ArgumentList $Arguments `
        -WorkingDirectory $Repository `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Group.Stdout `
        -RedirectStandardError $Group.Stderr `
        -PassThru
    }

    for ($Index = 0; $Index -lt $Processes.Count; $Index++) {
        $Process = $Processes[$Index]
        $Process.WaitForExit()
        $Process.Refresh()
    }
    & $Python -m hemfm.worker_completion verify-provenance `
        --config $Config --repository $Repository --input $LaunchProvenancePath
    if ($LASTEXITCODE -ne 0) {
        throw "Matched-audit source/config provenance changed while workers were running."
    }
    for ($Index = 0; $Index -lt $Processes.Count; $Index++) {
        $Process = $Processes[$Index]
        Assert-MatchedAuditWorkerComplete -Process $Process -Group $Groups[$Index] `
            -NotBeforeUtc $LaunchUtc -Python $Python -Config $Config `
            -Repository $Repository -EvidenceRoot (Join-Path $Repository "local-evidence\G6")
    }

    Push-Location $Repository
    try {
    & $Python -m hemfm --config $Config g7 audit
    if ($LASTEXITCODE -notin @(0, 2)) {
        throw "G7 integration audit failed to execute after matched audits."
    }
    & $Python -m hemfm --config $Config gates status
    if ($LASTEXITCODE -ne 0) {
        throw "Gate status failed after matched audits."
    }
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -ne $CoordinatorLock) {
        $CoordinatorLock.Dispose()
    }
}
