param(
    [int]$Count = 1,
    [long]$Seed = 12345,
    [switch]$Hires,
    [switch]$Lora,
    [ValidateNotNullOrEmpty()]
    [string[]]$Csv = (Join-Path $PSScriptRoot 'prompts\quality.csv'),
    [ValidateSet('wai', 'animagine')]
    [string]$Model = 'wai'
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if ($Model -eq 'animagine' -and $Lora) {
    throw 'The -Lora preset uses WAI-trained rei_style_v3. Omit -Lora with -Model animagine.'
}
if ($Model -eq 'animagine' -and -not $PSBoundParameters.ContainsKey('Csv')) {
    $Csv = Join-Path $root 'prompts\animagine.csv'
}
$csvPaths = @(foreach ($path in $Csv) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw ('CSV file not found: ' + $path)
    }
    (Resolve-Path -LiteralPath $path).Path
})
$workflow = if ($Hires -and $Lora) { 'sdxl_lora_hires_api.json' } elseif ($Hires) { 'sdxl_hires_api.json' } elseif ($Lora) { 'sdxl_lora_api.json' } else { 'sdxl_base_api.json' }
if ($Model -eq 'animagine') {
    $workflow = if ($Hires) { 'animagine_hires_api.json' } else { 'animagine_base_api.json' }
}
$prefix = if ($Model -eq 'animagine') { 'animagine/new' } else { 'quality/new' }
$extra = if ($Lora) { @('--prepend','rei_style') } else { @() }
& (Join-Path $root 'venv\Scripts\python.exe') (Join-Path $root 'scripts\generate_images.py') `
    --workflow (Join-Path $root ('workflows\' + $workflow)) `
    --prompts @csvPaths --batch $Count --seed $Seed --prefix $prefix @extra
exit $LASTEXITCODE
