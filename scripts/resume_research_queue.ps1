<#
.SYNOPSIS
    Resume the HEM-FM v4 research challenger queue after an interrupted or failed job.

.DESCRIPTION
    The queue runner skips every job whose status file already reads "complete", so this
    script is safe to re-run. It fast-forwards past every completed MIMIC LV, TED temporal,
    EV9V view, Unity landmark, and external-OOD wave, then retries the first incomplete job.
    The post-queue PanEcho challenger is started alongside and waits for the main queue to
    reach "complete" before it runs; it also skips an already-completed PanEcho audit.

    Nothing here bypasses a gate. The pipeline's own fail-closed checks still apply.
#>
param(
    [string]$Environment = "$PSScriptRoot\..\..\..\work\.venv-hemfm",
    [string]$RunRoot     = "$PSScriptRoot\..\..\..\work\hemfm-v4-runtime\runs",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$Repository = (Resolve-Path "$PSScriptRoot\..").Path
$Python     = Join-Path (Resolve-Path $Environment).Path "Scripts\python.exe"
$RunRoot    = (Resolve-Path $RunRoot).Path

if (-not (Test-Path $Python)) { throw "Python interpreter not found at $Python" }

Write-Host "repository : $Repository"
Write-Host "interpreter: $Python"
Write-Host "run root   : $RunRoot"
Write-Host ""

# Two RTX 5080s are a hardware floor in protocol.yaml. Fail here rather than mid-wave.
& $Python -c @"
import sys, torch
count = torch.cuda.device_count()
print(f'torch {torch.__version__} | cuda {torch.version.cuda} | visible gpus {count}')
for index in range(count):
    print(f'  [{index}] {torch.cuda.get_device_name(index)}')
sys.exit(0 if count >= 2 else 1)
"@
if ($LASTEXITCODE -ne 0) { throw "Fewer than two CUDA devices are visible; the queue schedules device 0 and device 1." }
Write-Host ""

if (-not $SkipTests) {
    Write-Host "Running the test suite before touching data..."
    Push-Location $Repository
    try {
        & $Python -m pytest -q -p no:cacheprovider
        if ($LASTEXITCODE -ne 0) { throw "Test suite failed with exit code $LASTEXITCODE; the queue was not started." }
    }
    finally { Pop-Location }
    Write-Host ""
}

$PostQueue = Start-Process -FilePath $Python `
    -ArgumentList @("$Repository\scripts\run_post_queue_challengers.py", "--repository", $Repository, "--run-root", $RunRoot) `
    -WorkingDirectory $Repository -NoNewWindow -PassThru

Write-Host "post-queue watcher started (pid $($PostQueue.Id))"
Write-Host "starting the research queue; it resumes at the first incomplete job"
Write-Host ""

& $Python "$Repository\scripts\run_research_training_queue.py" --repository $Repository --run-root $RunRoot
$QueueExit = $LASTEXITCODE

Write-Host ""
Write-Host "research queue exited with $QueueExit"
Get-Content -Raw (Join-Path $RunRoot "week_training\research_queue_status.json")

if ($QueueExit -ne 0) {
    if (-not $PostQueue.HasExited) { $PostQueue | Stop-Process }
    throw "Research queue failed; see $RunRoot\logs for the failing job."
}

$PostQueue | Wait-Process
Get-Content -Raw (Join-Path $RunRoot "week_training\post_queue_status.json")

