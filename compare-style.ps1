[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string[]]$RefImage = @('input\HWAC6ZqasAAkFhT.jpg'),
    [string]$Csv = 'prompts\prompts.csv',
    [ValidateRange(0, 100000)]
    [int]$Start = 0,
    [ValidateRange(1, 20)]
    [int]$Limit = 3,
    [ValidateRange(0, 9007199254740991)]
    [long]$Seed = 12345,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
function Resolve-InputFile([string]$Path) {
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $root $Path }
    $file = Get-Item -LiteralPath $Path
    if ($file.PSIsContainer) { throw "Expected a file: $Path" }
    return $file.FullName
}

$csvPath = Resolve-InputFile $Csv
$py = Join-Path $root 'venv\Scripts\python.exe'
$names = @{}
$references = @(foreach ($path in $RefImage) {
    $refPath = Resolve-InputFile $path
    $name = [IO.Path]::GetFileName($refPath)
    if ($names.ContainsKey($name)) { throw "Duplicate reference filename: $name" }
    $names[$name] = $true
    $inputCopy = Join-Path (Join-Path $root 'input') $name
    if ($inputCopy -ne $refPath -and (Test-Path -LiteralPath $inputCopy)) {
        if ((Get-FileHash -LiteralPath $inputCopy).Hash -ne (Get-FileHash -LiteralPath $refPath).Hash) {
            throw "A different reference already exists at $inputCopy. Rename your reference file before comparing."
        }
        $refPath = $inputCopy
    }
    $refPath
})

$run = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$common = @(
    (Join-Path $root 'scripts\generate_images.py'), '--prompts', $csvPath,
    '--start', "$Start", '--limit', "$Limit", '--batch', '1', '--seed', "$Seed",
    '--steps', '28', '--cfg', '5.5', '--sampler', 'dpmpp_2m', '--scheduler', 'karras',
    '--stop-on-error'
)
if ($DryRun) { $common += '--dry-run' }

function Invoke-Variant([string]$Variant, [string]$Template, [string[]]$Options = @()) {
    $prefix = "style_compare/$run/$Variant"
    Write-Output "Comparing: $prefix"
    & $py @common --workflow (Join-Path $root "workflows\$Template") --prefix $prefix @Options
    if ($LASTEXITCODE -ne 0) { throw "Generation failed for $Variant (exit code $LASTEXITCODE)" }
}

Invoke-Variant 'base' 'sdxl_base_api.json'
for ($i = 0; $i -lt $references.Count; $i++) {
    $label = if ($references.Count -eq 1) { 'ref' } else { 'ref_{0:D2}' -f ($i + 1) }
    Write-Output "$label = $($references[$i])"
    foreach ($setting in @('035', '060')) {
        $weight = if ($setting -eq '035') { '0.35' } else { '0.60' }
        $options = @('--ref-image', $references[$i], '--ipadapter-weight', $weight,
                     '--ipadapter-weight-type', 'style transfer')
        Invoke-Variant "${label}_$setting" 'sdxl_ipadapter_api.json' $options
    }
}
