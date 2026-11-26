# Unattended training with automatic resume (Windows; ported from v1).
#   powershell -ExecutionPolicy Bypass -File training\train_loop.ps1 --run base --maps maps\generated\train\index_combat.json --envs 16
# Every start passes --resume auto (runs\<run>\latest.pt). A crash restarts after 15 s;
# Ctrl+C makes train.py save latest.pt and exit 0, which ends the loop.
# Keeps Windows awake while it runs (SetThreadExecutionState).
#
# Before every (re)start, leftover ViZDoom engines and Python workers from THIS repository are
# stopped. On Windows an abrupt stop (Ctrl+C, or a crash during start-up) can leave the 16
# vizdoom.exe engines running; each failed restart then added 16 more until RAM ran out
# (seen on the 4060: MemoryError / OpenBLAS allocation failures in a restart loop).
# Only processes whose executable lives under this repository are touched.
Set-Location (Join-Path $PSScriptRoot "..")
$repo = (Get-Location).Path
$py = Join-Path $repo ".venv\Scripts\python.exe"
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

function Clear-Leftovers {
    $left = @(Get-Process vizdoom, python -ErrorAction SilentlyContinue |
              Where-Object { $_.Path -and $_.Path.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase) })
    if ($left.Count -gt 0) {
        Write-Host "stopping $($left.Count) leftover engine/worker process(es) from a previous run"
        $left | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    Write-Host ("free RAM: {0:N1} GB" -f $free)
}

try {
    while ($true) {
        Clear-Leftovers
        & $py -m training.train @args --resume auto
        if ($LASTEXITCODE -eq 0) { Write-Host "training finished normally"; break }
        Write-Host "$(Get-Date -Format s) train.py exited with $LASTEXITCODE - restarting in 15 s (Ctrl+C to stop)"
        Start-Sleep -Seconds 15
    }
} finally {
    Clear-Leftovers
    [Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null  # back to normal
}
