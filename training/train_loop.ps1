# Unattended training with automatic resume (Windows; ported from v1).
#   powershell -ExecutionPolicy Bypass -File training\train_loop.ps1 --run base --maps maps\generated\train\index_combat.json --envs 16
# Every start passes --resume auto (runs\<run>\latest.pt). A crash restarts after 15 s;
# Ctrl+C makes train.py save latest.pt and exit 0, which ends the loop.
# Keeps Windows awake while it runs (SetThreadExecutionState).
Set-Location (Join-Path $PSScriptRoot "..")
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
try {
    while ($true) {
        & $py -m training.train @args --resume auto
        if ($LASTEXITCODE -eq 0) { Write-Host "training finished normally"; break }
        Write-Host "$(Get-Date -Format s) train.py exited with $LASTEXITCODE - restarting in 15 s (Ctrl+C to stop)"
        Start-Sleep -Seconds 15
    }
} finally {
    [Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null  # back to normal
}
