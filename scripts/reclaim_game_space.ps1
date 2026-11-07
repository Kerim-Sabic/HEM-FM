[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Execute,
    [string]$BackupRoot = "$PSScriptRoot\..\..\..\work\metron-v8-runtime\removed-game-launcher-records"
)

$ErrorActionPreference = 'Stop'

$GameTargets = @(
    'C:\Program Files (x86)\Call of Duty',
    'C:\Program Files (x86)\Steam\steamapps\common\ABInfinite',
    'C:\Program Files (x86)\Steam\steamapps\common\Bloons TD Battles 2',
    'C:\Program Files (x86)\Steam\steamapps\common\PUBG',
    'C:\Program Files (x86)\Steam\steamapps\common\REPO',
    'C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive',
    'C:\Program Files (x86)\Steam\steamapps\common\Call of Duty HQ',
    'C:\Program Files\Epic Games\AmongUs',
    'C:\Program Files\Epic Games\Fortnite',
    'C:\Program Files\Epic Games\JackboxPartyPack1',
    'C:\Program Files\Epic Games\Operatorf2ec4',
    'C:\Program Files\Epic Games\rocketleague',
    'C:\Games\Plague Inc Evolved'
)

$AllowedGameRoots = @(
    'C:\Program Files (x86)\Call of Duty',
    'C:\Program Files (x86)\Steam\steamapps\common',
    'C:\Program Files\Epic Games',
    'C:\Games'
)

$LauncherRecords = @(
    'C:\Program Files (x86)\Steam\steamapps\appmanifest_1276390.acf',
    'C:\Program Files (x86)\Steam\steamapps\appmanifest_2073620.acf',
    'C:\Program Files (x86)\Steam\steamapps\appmanifest_3241660.acf',
    'C:\Program Files (x86)\Steam\steamapps\appmanifest_578080.acf',
    'C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests\2D47F36D42D41B04BDFE04A06A1F2B87.item',
    'C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests\4BB79C5F4710E4734499198DBC251D8E.item',
    'C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests\78EC80B14513FA18F100589CDD5B4568.item',
    'C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests\BB4B52E943EB88264D6BB9A23FDA05C5.item',
    'C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests\E43F55284EC1E0D99087688499D48716.item'
)

$AllowedRecordRoots = @(
    'C:\Program Files (x86)\Steam\steamapps',
    'C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests'
)

function Test-ChildOfAllowedRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string[]]$AllowedRoots,
        [switch]$AllowExactRoot
    )
    $FullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    foreach ($Root in $AllowedRoots) {
        $FullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
        if ($AllowExactRoot -and $FullPath.Equals($FullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
        if ($FullPath.StartsWith($FullRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Get-TreeBytes {
    param([Parameter(Mandatory = $true)][string]$Path)
    $Output = robocopy $Path NUL /L /S /BYTES /NFL /NDL /NJH /XJ /R:0 /W:0 2>$null
    $Line = ($Output | Select-String -Pattern '^\s*Bytes\s*:').Line
    if ($Line -match '^\s*Bytes\s*:\s*(\d+)') {
        return [int64]$Matches[1]
    }
    return [int64]0
}

$ResolvedTargets = @()
foreach ($Target in $GameTargets) {
    if (-not (Test-Path -LiteralPath $Target -PathType Container)) {
        continue
    }
    $Resolved = (Resolve-Path -LiteralPath $Target).Path
    if (-not (Test-ChildOfAllowedRoot -Path $Resolved -AllowedRoots $AllowedGameRoots -AllowExactRoot)) {
        throw "Refusing target outside approved game roots: $Resolved"
    }
    $ResolvedTargets += [pscustomobject]@{
        Path = $Resolved
        Bytes = Get-TreeBytes -Path $Resolved
    }
}

$ResolvedRecords = @()
foreach ($Record in $LauncherRecords) {
    if (-not (Test-Path -LiteralPath $Record -PathType Leaf)) {
        continue
    }
    $Resolved = (Resolve-Path -LiteralPath $Record).Path
    if (-not (Test-ChildOfAllowedRoot -Path $Resolved -AllowedRoots $AllowedRecordRoots)) {
        throw "Refusing launcher record outside approved roots: $Resolved"
    }
    $ResolvedRecords += $Resolved
}

$TotalBytes = ($ResolvedTargets | Measure-Object -Property Bytes -Sum).Sum
Write-Host ('Confirmed game installation targets: {0}' -f $ResolvedTargets.Count)
$ResolvedTargets | Sort-Object Bytes -Descending | Select-Object Path, @{Name = 'GiB'; Expression = {[math]::Round($_.Bytes / 1GB, 2)}} | Format-Table -AutoSize
Write-Host ('Total reclaimable game content: {0:N2} GiB' -f ($TotalBytes / 1GB))
Write-Host 'Save-data locations, Steam userdata, screenshots, Documents and AppData are not targeted.'

if (-not $Execute) {
    Write-Host 'Audit only. Re-run with -Execute from an elevated PowerShell window to remove these game installations.'
    exit 0
}

$BackupPath = [System.IO.Path]::GetFullPath($BackupRoot)
New-Item -ItemType Directory -Path $BackupPath -Force | Out-Null
foreach ($Record in $ResolvedRecords) {
    Copy-Item -LiteralPath $Record -Destination (Join-Path $BackupPath ([System.IO.Path]::GetFileName($Record))) -Force
}

foreach ($Target in $ResolvedTargets) {
    if ($PSCmdlet.ShouldProcess($Target.Path, 'Permanently remove confirmed game installation files')) {
        Remove-Item -LiteralPath $Target.Path -Recurse -Force
    }
}
foreach ($Record in $ResolvedRecords) {
    if ($PSCmdlet.ShouldProcess($Record, 'Remove backed-up launcher installation record')) {
        Remove-Item -LiteralPath $Record -Force
    }
}

$Drive = [System.IO.DriveInfo]::new('C')
Write-Host ('Cleanup complete. C: free space: {0:N2} GiB' -f ($Drive.AvailableFreeSpace / 1GB))
Write-Host ('Launcher records were backed up to: {0}' -f $BackupPath)
