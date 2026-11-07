param(
    [Parameter(Mandatory = $true)]
    [string]$StatusPath
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

[System.Windows.Forms.Application]::EnableVisualStyles()

$Bg = [System.Drawing.ColorTranslator]::FromHtml('#071019')
$Panel = [System.Drawing.ColorTranslator]::FromHtml('#0D1B26')
$PanelAlt = [System.Drawing.ColorTranslator]::FromHtml('#0A1822')
$Text = [System.Drawing.ColorTranslator]::FromHtml('#EAF2F4')
$Muted = [System.Drawing.ColorTranslator]::FromHtml('#90A7B2')
$Green = [System.Drawing.ColorTranslator]::FromHtml('#77DEB1')
$Amber = [System.Drawing.ColorTranslator]::FromHtml('#F6C574')
$Red = [System.Drawing.ColorTranslator]::FromHtml('#FFAD98')
$Border = [System.Drawing.ColorTranslator]::FromHtml('#253B4A')

function New-TextLabel {
    param(
        [string]$Value,
        [int]$Size = 10,
        [System.Drawing.Color]$Color = $Text,
        [System.Drawing.FontStyle]$Style = [System.Drawing.FontStyle]::Regular
    )
    $Label = New-Object System.Windows.Forms.Label
    $Label.Text = $Value
    $Label.ForeColor = $Color
    $Label.BackColor = [System.Drawing.Color]::Transparent
    $Label.Font = New-Object System.Drawing.Font('Segoe UI', $Size, $Style)
    $Label.AutoSize = $true
    return $Label
}

function New-Panel {
    $Result = New-Object System.Windows.Forms.Panel
    $Result.BackColor = $Panel
    $Result.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
    return $Result
}

$Form = New-Object System.Windows.Forms.Form
$Form.Text = 'METRON v8 - Developer Console'
$Form.Size = New-Object System.Drawing.Size(1400, 900)
$Form.MinimumSize = New-Object System.Drawing.Size(1100, 760)
$Form.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$Form.BackColor = $Bg
$Form.ForeColor = $Text
$Form.Font = New-Object System.Drawing.Font('Segoe UI', 10)

$Header = New-Object System.Windows.Forms.Panel
$Header.Dock = [System.Windows.Forms.DockStyle]::Top
$Header.Height = 104
$Header.Padding = New-Object System.Windows.Forms.Padding(28, 18, 28, 8)
$Header.BackColor = $Bg
$Form.Controls.Add($Header)

$Brand = New-TextLabel -Value 'HORALIX RESEARCH' -Size 9 -Color $Green -Style Bold
$Brand.Location = New-Object System.Drawing.Point(28, 17)
$Header.Controls.Add($Brand)
$Title = New-TextLabel -Value 'METRON v8 Developer Console' -Size 25 -Style Bold
$Title.Location = New-Object System.Drawing.Point(25, 36)
$Header.Controls.Add($Title)
$Subtitle = New-TextLabel -Value 'Measurement-grade echocardiography programme - evidence before training' -Color $Muted
$Subtitle.Location = New-Object System.Drawing.Point(29, 77)
$Header.Controls.Add($Subtitle)
$Live = New-TextLabel -Value 'LIVE' -Color $Green -Style Bold
$Live.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Right
$Live.Location = New-Object System.Drawing.Point(1305, 35)
$Header.Controls.Add($Live)

$Main = New-Object System.Windows.Forms.Panel
$Main.Dock = [System.Windows.Forms.DockStyle]::Fill
$Main.Padding = New-Object System.Windows.Forms.Padding(28, 0, 28, 22)
$Main.BackColor = $Bg
$Form.Controls.Add($Main)

$Layout = New-Object System.Windows.Forms.TableLayoutPanel
$Layout.Dock = [System.Windows.Forms.DockStyle]::Fill
$Layout.ColumnCount = 2
$Layout.RowCount = 4
$Layout.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent, 50)))
$Layout.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle([System.Windows.Forms.SizeType]::Percent, 50)))
$Layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute, 82)))
$Layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute, 62)))
$Layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Absolute, 242)))
$Layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle([System.Windows.Forms.SizeType]::Percent, 100)))
$Main.Controls.Add($Layout)

$Banner = New-Panel
$Banner.Dock = [System.Windows.Forms.DockStyle]::Fill
$Banner.Margin = New-Object System.Windows.Forms.Padding(0, 0, 0, 10)
$Layout.Controls.Add($Banner, 0, 0)
$Layout.SetColumnSpan($Banner, 2)
$Phase = New-TextLabel -Value 'Loading S0 evidence...' -Size 14 -Style Bold
$Phase.Location = New-Object System.Drawing.Point(15, 12)
$Banner.Controls.Add($Phase)
$Next = New-TextLabel -Value 'Waiting for the programme status' -Color $Muted
$Next.Location = New-Object System.Drawing.Point(16, 43)
$Banner.Controls.Add($Next)
$Seal = New-TextLabel -Value 'TEST seal unknown' -Color $Amber -Style Bold
$Seal.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Right
$Seal.Location = New-Object System.Drawing.Point(1135, 18)
$Banner.Controls.Add($Seal)

$SummaryPanel = New-Panel
$SummaryPanel.Dock = [System.Windows.Forms.DockStyle]::Fill
$SummaryPanel.Margin = New-Object System.Windows.Forms.Padding(0, 0, 0, 10)
$Layout.Controls.Add($SummaryPanel, 0, 1)
$Layout.SetColumnSpan($SummaryPanel, 2)
$Summary = New-TextLabel -Value '' -Size 12
$Summary.Location = New-Object System.Drawing.Point(15, 16)
$SummaryPanel.Controls.Add($Summary)

$StagesPanel = New-Panel
$StagesPanel.Dock = [System.Windows.Forms.DockStyle]::Fill
$StagesPanel.Margin = New-Object System.Windows.Forms.Padding(0, 0, 0, 10)
$Layout.Controls.Add($StagesPanel, 0, 2)
$Layout.SetColumnSpan($StagesPanel, 2)
$StagesTitle = New-TextLabel -Value 'STAGE MAP AND WHAT REMAINS' -Size 9 -Color $Muted -Style Bold
$StagesTitle.Location = New-Object System.Drawing.Point(15, 11)
$StagesPanel.Controls.Add($StagesTitle)
$StagesFlow = New-Object System.Windows.Forms.FlowLayoutPanel
$StagesFlow.Location = New-Object System.Drawing.Point(12, 36)
$StagesFlow.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right
$StagesFlow.Size = New-Object System.Drawing.Size(1310, 190)
$StagesFlow.BackColor = $Panel
$StagesFlow.WrapContents = $true
$StagesPanel.Controls.Add($StagesFlow)

$ChecksPanel = New-Panel
$ChecksPanel.Dock = [System.Windows.Forms.DockStyle]::Fill
$ChecksPanel.Margin = New-Object System.Windows.Forms.Padding(0, 0, 6, 0)
$Layout.Controls.Add($ChecksPanel, 0, 3)
$ChecksTitle = New-TextLabel -Value 'S0 EVIDENCE CHECKS' -Size 9 -Color $Muted -Style Bold
$ChecksTitle.Location = New-Object System.Drawing.Point(15, 11)
$ChecksPanel.Controls.Add($ChecksTitle)
$ChecksBox = New-Object System.Windows.Forms.RichTextBox
$ChecksBox.Location = New-Object System.Drawing.Point(12, 36)
$ChecksBox.Anchor = [System.Windows.Forms.AnchorStyles]::Top -bor [System.Windows.Forms.AnchorStyles]::Bottom -bor [System.Windows.Forms.AnchorStyles]::Left -bor [System.Windows.Forms.AnchorStyles]::Right
$ChecksBox.Size = New-Object System.Drawing.Size(630, 220)
$ChecksBox.BackColor = $PanelAlt
$ChecksBox.ForeColor = $Text
$ChecksBox.BorderStyle = [System.Windows.Forms.BorderStyle]::None
$ChecksBox.ReadOnly = $true
$ChecksBox.Font = New-Object System.Drawing.Font('Consolas', 10)
$ChecksPanel.Controls.Add($ChecksBox)

$RightPanel = New-Panel
$RightPanel.Dock = [System.Windows.Forms.DockStyle]::Fill
$RightPanel.Margin = New-Object System.Windows.Forms.Padding(6, 0, 0, 0)
$Layout.Controls.Add($RightPanel, 1, 3)
$BlockTitle = New-TextLabel -Value 'CURRENT BLOCKERS' -Size 9 -Color $Muted -Style Bold
$BlockTitle.Location = New-Object System.Drawing.Point(15, 11)
$RightPanel.Controls.Add($BlockTitle)
$Blockers = New-TextLabel -Value 'Waiting for S0' -Color $Amber
$Blockers.Location = New-Object System.Drawing.Point(15, 36)
$RightPanel.Controls.Add($Blockers)
$GpuTitle = New-TextLabel -Value 'GPU AND SYSTEM HEALTH' -Size 9 -Color $Muted -Style Bold
$GpuTitle.Location = New-Object System.Drawing.Point(15, 112)
$RightPanel.Controls.Add($GpuTitle)
$GpuText = New-TextLabel -Value '' -Size 11
$GpuText.Location = New-Object System.Drawing.Point(15, 138)
$RightPanel.Controls.Add($GpuText)

function Update-Dashboard {
    if (-not (Test-Path -LiteralPath $StatusPath)) {
        $Phase.Text = 'status file unavailable'
        $Phase.ForeColor = $Red
        return
    }
    try {
        $Status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
    } catch {
        $Phase.Text = 'status file is not valid JSON'
        $Phase.ForeColor = $Red
        return
    }

    $Ready = $Status.ready_to_launch -eq $true
    $Phase.Text = ([string]$Status.phase).Replace('_', ' ')
    $Phase.ForeColor = if ($Ready) { $Green } else { $Amber }
    $Next.Text = 'Next: ' + ([string]$Status.next_action).Replace('_', ' ')
    $Sealed = $Status.locked_test_pixels_accessed -eq $false
    $Seal.Text = if ($Sealed) { 'LOCKED TEST PIXELS SEALED' } else { 'TEST SEAL NOT PROVEN' }
    $Seal.ForeColor = if ($Sealed) { $Green } else { $Red }

    $Summary.Text = '{0:N0} GPU-hours  |  {1} measurement playbooks  |  {2} GiB RAM  |  {3} / {4} GiB disk free  |  S0 {5}' -f `
        $Status.programme.gpu_hours, $Status.programme.measurement_playbooks, `
        $Status.hardware.ram_total_gib, $Status.hardware.disk_free_gib, `
        $Status.hardware.disk_total_gib, $(if ($Ready) { 'READY' } else { 'BLOCKED' })
    $Summary.ForeColor = if ($Ready) { $Green } else { $Amber }

    $StagesFlow.SuspendLayout()
    $StagesFlow.Controls.Clear()
    foreach ($Stage in $Status.stages) {
        $Card = New-Object System.Windows.Forms.Panel
        $Card.Size = New-Object System.Drawing.Size(205, 82)
        $Card.Margin = New-Object System.Windows.Forms.Padding(4)
        $Card.BackColor = $PanelAlt
        $Card.BorderStyle = [System.Windows.Forms.BorderStyle]::FixedSingle
        $State = [string]$Stage.state
        $Head = New-TextLabel -Value ('{0}   {1}' -f $Stage.stage_id, $State.ToUpper()) -Size 9 -Style Bold -Color $(if ($State -eq 'ready') { $Green } elseif ($State -eq 'blocked') { $Red } else { $Muted })
        $Head.Location = New-Object System.Drawing.Point(8, 7)
        $Card.Controls.Add($Head)
        $Name = New-TextLabel -Value ([string]$Stage.name) -Size 7 -Color $Muted
        $Name.MaximumSize = New-Object System.Drawing.Size(188, 34)
        $Name.Location = New-Object System.Drawing.Point(8, 28)
        $Card.Controls.Add($Name)
        $Budget = New-TextLabel -Value ('{0:N0} GPU-h' -f $Stage.gpu_hours) -Size 7
        $Budget.Location = New-Object System.Drawing.Point(8, 62)
        $Card.Controls.Add($Budget)
        $StagesFlow.Controls.Add($Card)
    }
    $StagesFlow.ResumeLayout()

    $ChecksBox.Clear()
    foreach ($Property in $Status.checks.PSObject.Properties) {
        $Passed = $Property.Value -eq $true
        $ChecksBox.SelectionColor = if ($Passed) { $Green } else { $Red }
        $ChecksBox.AppendText($(if ($Passed) { 'PASS  ' } else { 'BLOCK ' }))
        $ChecksBox.SelectionColor = $Text
        $ChecksBox.AppendText($Property.Name.Replace('_', ' ') + [Environment]::NewLine)
    }

    if (@($Status.blockers).Count -gt 0) {
        $Blockers.Text = (@($Status.blockers) | ForEach-Object { '- ' + ([string]$_).Replace('_', ' ') }) -join [Environment]::NewLine
        $Blockers.ForeColor = $Amber
    } else {
        $Blockers.Text = 'No S0 blocker'
        $Blockers.ForeColor = $Green
    }

    $GpuLines = @()
    try {
        $Rows = & nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
        foreach ($Row in $Rows) {
            $Parts = @($Row -split ',' | ForEach-Object { $_.Trim() })
            if ($Parts.Count -eq 7) {
                $GpuLines += 'GPU {0} - {1}' -f $Parts[0], $Parts[1]
                $GpuLines += '  {0}% utilization | {1:N1}/{2:N1} GiB VRAM | {3} C | {4} W' -f $Parts[2], ([double]$Parts[3] / 1024), ([double]$Parts[4] / 1024), $Parts[5], $Parts[6]
                $GpuLines += ''
            }
        }
    } catch {
        $GpuLines += 'GPU telemetry unavailable'
    }
    $GpuText.Text = $GpuLines -join [Environment]::NewLine
    $Live.Text = 'LIVE ' + (Get-Date -Format 'HH:mm:ss')
}

$Timer = New-Object System.Windows.Forms.Timer
$Timer.Interval = 2000
$Timer.Add_Tick({ Update-Dashboard })
$Timer.Start()
Update-Dashboard
[void]$Form.ShowDialog()
