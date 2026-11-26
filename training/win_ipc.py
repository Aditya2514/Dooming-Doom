"""Windows fix for ViZDoom's controller <-> engine latency (training only).

On Windows 11, power throttling makes the hidden vizdoom.exe child ignore
high-resolution timer requests, so every IPC wait rounds up to the scheduler
tick. Measured with the official wrapper on the dev laptop (1 env, test map,
render off): 347 -> 1,099 decisions/s (2.88 -> 0.91 ms per step).

Fix: opt this process and every vizdoom.exe child out of power throttling
(execution speed + IGNORE_TIMER_RESOLUTION) and request a 1 ms timer.
Call it once per worker, after the env is created (make_env starts the engine).
No-op on other platforms. Ported from the v1 repo (env/engine.py).
"""
from __future__ import annotations

import sys


def windows_fast_ipc() -> dict:
    if sys.platform != "win32":
        return {"applied": False}
    import ctypes
    import ctypes.wintypes as wt

    import psutil

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _PPT(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]

    def unthrottle(pid: int) -> bool:
        h = k32.OpenProcess(0x0200 | 0x1000, False, pid)  # SET_INFORMATION | QUERY_LIMITED
        if not h:
            return False
        s = _PPT(1, 0x1 | 0x4, 0)  # control EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, state off
        ok = bool(k32.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s)))  # ProcessPowerThrottling
        k32.CloseHandle(h)
        return ok

    me = psutil.Process()
    kids = [p for p in me.children(recursive=True) if p.name().lower().startswith("vizdoom")]
    res = {"applied": True, "self": unthrottle(me.pid), "engines": [unthrottle(p.pid) for p in kids]}
    ctypes.windll.winmm.timeBeginPeriod(1)
    return res
