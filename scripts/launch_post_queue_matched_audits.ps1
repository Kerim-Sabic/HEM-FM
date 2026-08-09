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

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

while (Get-Process -Id $QueueProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $PollSeconds
}

$PowerShell = (Get-Process -Id $PID).Path
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

foreach ($Process in $Processes) {
    $Process.WaitForExit()
    if ($Process.ExitCode -ne 0) {
        throw "Matched staged audit worker $($Process.Id) failed with exit code $($Process.ExitCode)."
    }
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
