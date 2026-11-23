<#
.SYNOPSIS
IP-Adapter (reference-image style transfer) generation launcher.

.DESCRIPTION
Runs scripts\generate_images.py against workflows\sdxl_ipadapter_api.json.
If ComfyUI is not responding, it is started via start-comfyui.ps1 and the
script waits until the API answers before submitting any prompt.

Defaults follow the recommended range for this style:
  weight 0.6 / style transfer / CFG 5.5 / steps 28 / dpmpp_2m + karras

.EXAMPLE
.\generate-ipadapter.ps1 -DryRun -Limit 2

.EXAMPLE
.\generate-ipadapter.ps1 -Limit 4 -Count 2

.EXAMPLE
.\generate-ipadapter.ps1 -Weight 0.7 -WeightType 'style and composition' -Limit 4

.EXAMPLE
.\generate-ipadapter.ps1 -Seed 12345 -Limit 1 -Prefix ipadapter/cmp_ipa
#>

[CmdletBinding()]
param(
    # Images per prompt (maps to --batch).
    [ValidateRange(1, 64)]
    [int]$Count = 1,

    # Number of prompt rows to process (0 = all).
    [ValidateRange(0, 100000)]
    [int]$Limit = 0,

    # Skip the first N prompt rows.
    [ValidateRange(0, 100000)]
    [int]$Start = 0,

    # Fixed seed (-1 = random per image).
    [long]$Seed = -1,

    # Reference image for IP-Adapter. Relative paths resolve from the project root.
    [ValidateNotNullOrEmpty()]
    [string]$RefImage = 'input\HWAC6ZqasAAkFhT.jpg',

    # IP-Adapter weight. 0.5-0.7 is the recommended band for this style.
    [ValidateRange(0.0, 1.0)]
    [double]$Weight = 0.6,

    # How the reference is applied. 'style transfer' copies look only, not composition.
    [ValidateSet(
        'linear',
        'ease in',
        'ease out',
        'ease in-out',
        'reverse in-out',
        'weak input',
        'weak output',
        'weak middle',
        'strong middle',
        'style transfer',
        'composition',
        'strong style transfer',
        'style transfer precise',
        'composition precise'
    )]
    [string]$WeightType = 'style transfer',

    # Fraction of sampling steps where IP-Adapter starts applying.
    [ValidateRange(0.0, 1.0)]
    [double]$StartAt = 0.0,

    # Fraction of sampling steps where IP-Adapter stops applying.
    [ValidateRange(0.0, 1.0)]
    [double]$EndAt = 1.0,

    # Prompt CSV files. Relative paths resolve from the project root.
    [ValidateNotNullOrEmpty()]
    [string[]]$Csv = 'prompts\prompts.csv',

    # SaveImage filename prefix.
    [ValidateNotNullOrEmpty()]
    [string]$Prefix = 'ipadapter/new',

    # Override steps for every row (0 = use CSV value).
    [ValidateRange(0, 200)]
    [int]$Steps = 0,

    # Override CFG for every row (0 = use CSV value).
    [ValidateRange(0.0, 30.0)]
    [double]$Cfg = 0.0,

    # Build and print the workflow without submitting it.
    [switch]$DryRun,

    # Do not start ComfyUI automatically; fail if it is not already running.
    [switch]$NoAutoStart,

    # Seconds to wait for ComfyUI to become ready after auto-start.
    [ValidateRange(10, 3600)]
    [int]$StartupTimeout = 180
)

$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$py = Join-Path $root 'venv\Scripts\python.exe'
$generator = Join-Path $root 'scripts\generate_images.py'
$workflowPath = Join-Path $root 'workflows\sdxl_ipadapter_api.json'
$server = 'http://127.0.0.1:8188'

# --- Path resolution -------------------------------------------------------

function Resolve-ProjectPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $root $Path)
}

$refPath = Resolve-ProjectPath $RefImage
$csvPaths = @(foreach ($path in $Csv) { Resolve-ProjectPath $path })

# --- Preconditions ---------------------------------------------------------

if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw ('venv python not found: ' + $py)
}
if (-not (Test-Path -LiteralPath $generator -PathType Leaf)) {
    throw ('generator not found: ' + $generator)
}
if (-not (Test-Path -LiteralPath $workflowPath -PathType Leaf)) {
    throw ('IP-Adapter workflow not found: ' + $workflowPath)
}
if (-not (Test-Path -LiteralPath $refPath -PathType Leaf)) {
    throw ('reference image not found: ' + $refPath)
}
foreach ($path in $csvPaths) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw ('prompt CSV not found: ' + $path)
    }
}
if ($StartAt -ge $EndAt) {
    throw ('-StartAt must be smaller than -EndAt (got ' + $StartAt + ' >= ' + $EndAt + ').')
}

# Guard against pointing at a workflow that has no IP-Adapter nodes.
$wf = Get-Content -LiteralPath $workflowPath -Raw | ConvertFrom-Json
$applyNode = $wf.PSObject.Properties['22']
if ($null -eq $applyNode -or $applyNode.Value.class_type -ne 'IPAdapterAdvanced') {
    throw ('workflow has no IPAdapterAdvanced node 22: ' + $workflowPath)
}

# --- ComfyUI readiness -----------------------------------------------------

function Test-ComfyUI {
    try {
        $null = Invoke-RestMethod -Uri ($server + '/system_stats') -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-ComfyUI)) {
    if ($NoAutoStart) {
        throw ('ComfyUI is not responding at ' + $server + '. Start it first or drop -NoAutoStart.')
    }
    $starter = Join-Path $root 'start-comfyui.ps1'
    if (-not (Test-Path -LiteralPath $starter -PathType Leaf)) {
        throw ('start-comfyui.ps1 not found: ' + $starter)
    }
    Write-Output ('ComfyUI is not responding. Starting: ' + $starter)
    $proc = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $starter) `
        -WorkingDirectory $root -WindowStyle Minimized -PassThru

    $deadline = (Get-Date).AddSeconds($StartupTimeout)
    while ((Get-Date) -lt $deadline) {
        if (Test-ComfyUI) { break }
        Start-Sleep -Seconds 2
    }
    if (-not (Test-ComfyUI)) {
        throw ('ComfyUI did not become ready within ' + $StartupTimeout + ' seconds (PID ' + $proc.Id + ').')
    }
    Write-Output ('ComfyUI is ready (PID ' + $proc.Id + ').')
} else {
    Write-Output ('ComfyUI is already running at ' + $server + '.')
}

# --- Build generator arguments --------------------------------------------

$inv = [System.Globalization.CultureInfo]::InvariantCulture

$genArgs = @(
    $generator,
    '--workflow', $workflowPath,
    '--prompts'
) + $csvPaths + @(
    '--ref-image', $refPath,
    '--ipadapter-weight', $Weight.ToString($inv),
    '--ipadapter-weight-type', $WeightType,
    '--ipadapter-start', $StartAt.ToString($inv),
    '--ipadapter-end', $EndAt.ToString($inv),
    '--prefix', $Prefix,
    '--batch', $Count.ToString($inv)
)

if ($Limit -gt 0) { $genArgs += @('--limit', $Limit.ToString($inv)) }
if ($Start -gt 0) { $genArgs += @('--start', $Start.ToString($inv)) }
if ($Seed -ge 0) { $genArgs += @('--seed', $Seed.ToString($inv)) }
if ($Steps -gt 0) { $genArgs += @('--steps', $Steps.ToString($inv)) }
if ($Cfg -gt 0) { $genArgs += @('--cfg', $Cfg.ToString($inv)) }
if ($DryRun) { $genArgs += '--dry-run' }

Write-Output ''
Write-Output 'IP-Adapter settings:'
Write-Output ('  reference : ' + $refPath)
Write-Output ('  weight    : ' + $Weight.ToString($inv) + ' (' + $WeightType + ')')
Write-Output ('  range     : ' + $StartAt.ToString($inv) + ' - ' + $EndAt.ToString($inv))
Write-Output ('  prompts   : ' + ($csvPaths -join ', '))
Write-Output ('  rows      : start=' + $Start + ' limit=' + $(if ($Limit -gt 0) { $Limit } else { 'all' }) + ' batch=' + $Count)
Write-Output ('  seed      : ' + $(if ($Seed -ge 0) { $Seed } else { 'random' }))
Write-Output ('  prefix    : ' + $Prefix)
if ($Weight -lt 0.5 -or $Weight -gt 0.7) {
    Write-Output '  note      : weight is outside the recommended 0.5-0.7 band.'
}
Write-Output ''

# --- Run -------------------------------------------------------------------

Write-Output ('running: ' + $py + ' ' + ($genArgs -join ' '))
& $py @genArgs
$code = $LASTEXITCODE
Write-Output ('LASTEXITCODE = ' + $code)
exit $code
