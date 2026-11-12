[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$pythonPath = Join-Path $root 'venv\Scripts\python.exe'
$mainPath = Join-Path $root 'ComfyUI\main.py'
$mainPattern = '(?i)(?:^|\s|\x22)' + [regex]::Escape($mainPath) + '(?:\x22|\s|$)'

# Match both the virtual environment and the full entry-point path so other
# Python applications (or another ComfyUI installation) are left running.
$targets = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object {
        $_.ExecutablePath -ieq $pythonPath -and
        $_.CommandLine -match $mainPattern
    })

if ($targets.Count -eq 0) {
    Write-Output 'No matching ComfyUI process is running.'
    return
}

foreach ($target in $targets) {
    $processId = [int]$target.ProcessId
    if ($PSCmdlet.ShouldProcess("ComfyUI (PID $processId)", 'Terminate process (active generation will be interrupted)')) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            Write-Output "ComfyUI already exited (PID $processId)."
            continue
        }
        # The Process object retains the handle while we stop and wait.
        Stop-Process -InputObject $process -Force
        if (-not $process.WaitForExit(10000)) {
            throw "ComfyUI did not exit within 10 seconds (PID $processId)."
        }
        Write-Output "Stopped ComfyUI (PID $processId)."
    }
}
