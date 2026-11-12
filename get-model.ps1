<#
.SYNOPSIS
モデルファイルを F:\ai-image\models 配下の適切な場所へダウンロードする。

.DESCRIPTION
curl.exe でストリーミングダウンロードする。Invoke-WebRequest は
数 GB のファイルをメモリに溜め込むため、大きなモデルでは使わない。

SHA256 を指定すると、ダウンロード後に検証し、不一致ならファイルを破棄する。
既に同名ファイルがある場合はダウンロードをスキップする。

.EXAMPLE
.\get-model.ps1 -Url 'https://civitai.com/api/download/models/2883731' -Kind checkpoints -FileName 'waiIllustriousSDXL_v170.safetensors'

.EXAMPLE
.\get-model.ps1 -Url 'https://...' -Kind loras -FileName 'char.safetensors' -Sha256 'ABCDEF...'
#>

param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string] $Url,

    [Parameter(Mandatory)]
    [ValidateSet('checkpoints','loras','vae','controlnet','upscale_models','embeddings')]
    [string] $Kind,

    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string] $FileName,

    [string] $Sha256,

    [string] $Token
)

$ErrorActionPreference = 'Stop'

$root = 'F:\ai-image'
$destDir = Join-Path $root ('models\' + $Kind)

if (-not (Test-Path -LiteralPath $destDir)) {
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
}

$dest = Join-Path $destDir $FileName

if (Test-Path -LiteralPath $dest) {
    Write-Output ('already exists: ' + $dest)
    Write-Output ('size = ' + [math]::Round((Get-Item -LiteralPath $dest).Length / 1MB, 1) + ' MB')
    return
}

$tmp = $dest + '.part'

$curlArgs = @(
    '--location'
    '--fail'
    '--retry', '3'
    '--retry-delay', '5'
    '--user-agent', 'Mozilla/5.0'
    '--output', $tmp
)

if (-not [string]::IsNullOrWhiteSpace($Token)) {
    $curlArgs += @('--header', ('Authorization: Bearer ' + $Token))
}

$curlArgs += $Url

Write-Output ('downloading: ' + $Url)
Write-Output ('to         : ' + $tmp)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& curl.exe @curlArgs
$code = $LASTEXITCODE
$sw.Stop()

if ($code -ne 0) {
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force }
    throw ('curl failed with exit code ' + $code)
}

$sizeMB = [math]::Round((Get-Item -LiteralPath $tmp).Length / 1MB, 1)
Write-Output ('downloaded: ' + $sizeMB + ' MB in ' + [math]::Round($sw.Elapsed.TotalSeconds, 1) + ' s')

if (-not [string]::IsNullOrWhiteSpace($Sha256)) {
    Write-Output 'verifying sha256...'
    $actual = (Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash
    if ($actual -ne $Sha256.ToUpperInvariant()) {
        Remove-Item -LiteralPath $tmp -Force
        throw ('sha256 mismatch. expected=' + $Sha256 + ' actual=' + $actual)
    }
    Write-Output 'sha256 OK'
} else {
    Write-Output 'sha256 not provided; skipped verification'
}

Move-Item -LiteralPath $tmp -Destination $dest -Force
Write-Output ('saved: ' + $dest)
