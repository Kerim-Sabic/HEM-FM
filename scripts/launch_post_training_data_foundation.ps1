param(
    [int]$WaitProcessId = 64004,
    [string]$Environment = "$PSScriptRoot\..\..\..\work\.venv-hemfm",
    [string]$Config = "$PSScriptRoot\..\configs\protocol.yaml",
    [int]$Workers = 16,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path "$PSScriptRoot\..").Path
$Environment = (Resolve-Path $Environment).Path
$Python = Join-Path $Environment "Scripts\python.exe"
$Config = (Resolve-Path $Config).Path
$RuntimeRoot = (Resolve-Path "$PSScriptRoot\..\..\..\work\hemfm-v4-runtime").Path
$PrivateRoot = Join-Path $RuntimeRoot "private"
$LogRoot = Join-Path $RuntimeRoot "runs\data_foundation_v2"
$StatusPath = Join-Path $LogRoot "status.json"
$StdoutPath = Join-Path $LogRoot "pipeline.stdout.log"
$StderrPath = Join-Path $LogRoot "pipeline.stderr.log"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Write-Status {
    param(
        [Parameter(Mandatory = $true)][string]$Phase,
        [hashtable]$Extra = @{}
    )
    $Payload = [ordered]@{
        schema_version = 1
        updated_utc = [DateTime]::UtcNow.ToString("o")
        phase = $Phase
        locked_test_pixels_accessed = $false
        pixel_data_read = $false
    }
    foreach ($Key in $Extra.Keys) {
        $Payload[$Key] = $Extra[$Key]
    }
    $Temporary = "$StatusPath.$PID.tmp"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Temporary -Encoding UTF8
    Move-Item -LiteralPath $Temporary -Destination $StatusPath -Force
}

function Invoke-Hemfm {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 wraps native stderr as non-terminating error
        # records. Keep the stream in the log and decide success by exit code.
        $ErrorActionPreference = "Continue"
        & $Python -m hemfm --config $Config @Arguments 1>> $StdoutPath 2>> $StderrPath
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "HEM-FM command failed ($ExitCode): $($Arguments -join ' ')"
    }
}

function Move-GeneratedArtifactToArchive {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        return
    }
    $ResolvedSource = (Resolve-Path -LiteralPath $Source).Path
    $ResolvedPrivate = (Resolve-Path -LiteralPath $PrivateRoot).Path
    if (-not $ResolvedSource.StartsWith($ResolvedPrivate + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to move an artifact outside the private runtime: $ResolvedSource"
    }
    $Stamp = [DateTime]::UtcNow.ToString("yyyyMMdd-HHmmss")
    $Extension = [IO.Path]::GetExtension($ResolvedSource)
    $Stem = [IO.Path]::GetFileNameWithoutExtension($ResolvedSource)
    $Destination = Join-Path $ResolvedPrivate "$Stem.$Label-$Stamp$Extension"
    if (Test-Path -LiteralPath $Destination) {
        throw "Refusing to overwrite an archived artifact: $Destination"
    }
    Move-Item -LiteralPath $ResolvedSource -Destination $Destination
}

try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python environment is unavailable: $Python"
    }
    if ($Workers -lt 1 -or $PollSeconds -lt 5) {
        throw "Workers must be positive and PollSeconds must be at least 5."
    }

    Write-Status -Phase "waiting_for_gpu_and_matched_audits" -Extra @{ wait_process_id = $WaitProcessId }
    while (Get-Process -Id $WaitProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds $PollSeconds
    }

    Push-Location $Repository
    try {
        Write-Status -Phase "verifying_source_and_split"
        $ConfigObject = & $Python -c "import json,yaml; c=yaml.safe_load(open(r'$Config')); print(json.dumps(c['paths']))" | ConvertFrom-Json
        if (-not (Test-Path -LiteralPath $ConfigObject.dicom_root -PathType Container)) {
            throw "The authorized read-only DICOM source is unavailable: $($ConfigObject.dicom_root)"
        }
        if (-not (Test-Path -LiteralPath $ConfigObject.record_list -PathType Leaf)) {
            throw "The MIMIC record list is unavailable: $($ConfigObject.record_list)"
        }

        # A missing first-run split manifest is the expected signal to build it.
        # Windows PowerShell promotes native stderr to a terminating
        # NativeCommandError under the script-wide Stop policy, so scope the
        # probe to Continue long enough to inspect its real exit code.
        $PreviousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $Python -m hemfm --config $Config splits audit 1>> $StdoutPath 2>> $StderrPath
            $SplitAuditExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($SplitAuditExitCode -ne 0) {
            Invoke-Hemfm -Arguments @("splits", "build")
        }

        $InventoryCsv = Join-Path $PrivateRoot "dicom_inventory.csv"
        $Checkpoint = Join-Path $PrivateRoot ".dicom_inventory.checkpoint.json"
        $Journal = Join-Path $PrivateRoot ".dicom_inventory.journal.jsonl"
        $HasCsv = Test-Path -LiteralPath $InventoryCsv -PathType Leaf
        $HasCheckpoint = Test-Path -LiteralPath $Checkpoint -PathType Leaf
        $HasJournal = Test-Path -LiteralPath $Journal -PathType Leaf
        if ($HasCheckpoint -xor $HasJournal) {
            throw "Incomplete v2 transaction artifacts; refusing to guess whether resume is safe."
        }

        $Resume = $HasCsv -and $HasCheckpoint -and $HasJournal
        if ($HasCsv -and -not $Resume) {
            Move-GeneratedArtifactToArchive -Source $InventoryCsv -Label "legacy"
        } elseif ((-not $HasCsv) -and ($HasCheckpoint -or $HasJournal)) {
            throw "Transaction artifacts exist without their CSV; refusing unsafe resume."
        }

        if (-not $Resume) {
            foreach ($Name in @(
                "endpoint_cohorts_private.csv",
                "endpoint_cohorts_development_private.csv",
                "endpoint_cohorts_locked_test_private.csv"
            )) {
                Move-GeneratedArtifactToArchive -Source (Join-Path $PrivateRoot $Name) -Label "legacy"
            }
        }

        Write-Status -Phase "dicom_inventory_v2" -Extra @{ resume = $Resume; workers = $Workers }
        $InventoryArguments = @("inventory", "dicom", "--workers", "$Workers")
        if ($Resume) {
            $InventoryArguments += "--resume"
        }
        Invoke-Hemfm -Arguments $InventoryArguments

        Write-Status -Phase "calibration_v2"
        Invoke-Hemfm -Arguments @("calibration", "validate", "--max-files", "512")

        Write-Status -Phase "cohort_rebuild"
        foreach ($Name in @(
            "endpoint_cohorts_private.csv",
            "endpoint_cohorts_development_private.csv",
            "endpoint_cohorts_locked_test_private.csv"
        )) {
            Move-GeneratedArtifactToArchive -Source (Join-Path $PrivateRoot $Name) -Label "pre-v2-rebuild"
        }
        Invoke-Hemfm -Arguments @("cohort", "build")
        Invoke-Hemfm -Arguments @("splits", "audit")

        Write-Status -Phase "gate_refresh"
        Invoke-Hemfm -Arguments @("gates", "status")
        Write-Status -Phase "complete" -Extra @{
            inventory_csv = $InventoryCsv
            metadata_only = $true
            immutable_split_reused_or_built = $true
        }
    }
    finally {
        Pop-Location
    }
}
catch {
    Write-Status -Phase "failed" -Extra @{ error = "$($_.Exception.GetType().Name): $($_.Exception.Message)" }
    $_ | Out-String | Add-Content -LiteralPath $StderrPath
    exit 2
}
