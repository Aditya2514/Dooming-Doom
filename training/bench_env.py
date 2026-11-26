"""Env-only throughput of the official wrapper, plus an episode-ending check (Phase 0, T+0).

Run from the repo root, in the foreground (background jobs are throttled on Windows):

    .venv\\Scripts\\python.exe -m training.bench_env                      # 8, 14, 20 envs on the test map
    .venv\\Scripts\\python.exe -m training.bench_env --map freedoom2:MAP01 --envs 14

Each env runs in its own process (as training will), with render_buffers=False, random
actions, privileged_state() called every step (the reward wrapper will), and the Windows
latency fix applied. Workers import only the wrapper and numpy, never torch.

The ending check verifies what the PPO ending rule (spec F1) relies on: the tick cap
comes back as truncated (not terminated) on the exact step, the returned observation
is the previous one, and the env needs reset() afterwards.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def resolve_map(spec: str) -> tuple[str, str]:
    """'maps/x.wad:MAP03' | 'freedoom2:MAP01' | 'maps/x.wad' -> (wad path, map name)."""
    wad, _, name = spec.partition(":")
    if wad.lower() == "freedoom2":
        from doom_env.wrapper import IWAD
        wad = IWAD
    elif not os.path.isabs(wad):
        wad = str(REPO / wad)
    return wad, (name or "MAP01")


def make(wad: str, map_name: str, ticks: int, fix: bool):
    from doom_env import make_env

    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=False)
    if fix:
        from training.win_ipc import windows_fast_ipc
        windows_fast_ipc()
    return env


# --------------------------------------------------------------------------- ending check
def check_endings(wad: str, map_name: str, fix: bool) -> dict:
    """Tick cap of 12 tics = 3 decisions at frame skip 4."""
    env = make(wad, map_name, ticks=12, fix=fix)
    noop = np.zeros(9, np.int64)
    res = {}
    try:
        obs0, _ = env.reset(seed=0)
        o1, *_ = env.step(noop)
        o2, _, term2, trunc2, _ = env.step(noop)
        o3, r3, term3, trunc3, info3 = env.step(noop)
        ps = env.privileged_state()
        res["ends_on_step_3"] = (not term2 and not trunc2) and (trunc3 and not term3)
        res["final_obs_is_previous"] = all(
            np.array_equal(o3["rays"][k], o2["rays"][k]) for k in o2["rays"]
        ) and np.array_equal(o3["vitals"], o2["vitals"])
        res["privileged_timeout_flag"] = bool(ps["timeout"] and ps["finished"] and not ps["dead"])
        try:
            env.step(noop)
            res["step_after_end_raises"] = False
        except RuntimeError:
            res["step_after_end_raises"] = True
        obs_r, info_r = env.reset(seed=1)
        o, _, te, tr, info = env.step(noop)
        res["reset_starts_fresh"] = (not te and not tr) and info["tic"] <= 8
    finally:
        env.close()
    res["passed"] = all(res.values())
    return res


# --------------------------------------------------------------------------- throughput
def _worker(idx, wad, map_name, ticks, fix, warmup, seconds, ready, start, out):
    try:
        env = make(wad, map_name, ticks, fix)
        rng = np.random.default_rng(idx)
        env.reset(seed=idx)
        ready.wait()
        start.wait()

        def run(duration):
            n = eps = 0
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < duration:
                _, _, te, tr, _ = env.step(rng.integers(0, 2, 9))
                env.privileged_state()
                n += 1
                if te or tr:
                    env.reset()
                    eps += 1
            return n, eps, time.perf_counter() - t0

        run(warmup)
        n, eps, dt = run(seconds)
        env.close()
        out.put((idx, n, eps, dt, None))
    except Exception as e:  # report, don't hang the parent
        out.put((idx, 0, 0, 0.0, repr(e)))


def bench(n_envs: int, wad: str, map_name: str, ticks: int, fix: bool,
          warmup: float, seconds: float) -> dict:
    ctx = mp.get_context("spawn")
    ready = ctx.Barrier(n_envs + 1)
    start = ctx.Event()
    out = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(i, wad, map_name, ticks, fix, warmup, seconds,
                                               ready, start, out), daemon=True)
             for i in range(n_envs)]
    for p in procs:
        p.start()
    ready.wait()                       # every engine is up before the clock starts
    start.set()
    rows = [out.get(timeout=warmup + seconds + 300) for _ in procs]
    for p in procs:
        p.join(timeout=30)
    errors = [r[4] for r in rows if r[4]]
    total = sum(r[1] / r[3] for r in rows if r[3] > 0)
    return {"envs": n_envs, "decisions_per_s": round(total), "tics_per_s": round(4 * total),
            "per_env": round(total / n_envs), "episodes": sum(r[2] for r in rows),
            "failed_workers": len(errors), "errors": errors[:3]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", type=int, nargs="+", default=[8, 14, 20])
    ap.add_argument("--map", default="maps/two_rooms.wad:MAP01")
    ap.add_argument("--ticks", type=int, default=2100, help="episode tick cap")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--no-fix", action="store_true", help="skip the Windows latency fix (for comparison)")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    a = ap.parse_args()

    wad, map_name = resolve_map(a.map)
    fix = not a.no_fix
    info = {"host": platform.node(), "os": platform.platform(), "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(), "map": f"{Path(wad).name}:{map_name}",
            "windows_fix": fix and os.name == "nt"}
    print(json.dumps(info))

    endings = check_endings(wad, map_name, fix)
    print(f"ending check: {'PASS' if endings['passed'] else 'FAIL'}  {endings}")

    results = []
    for n in a.envs:
        r = bench(n, wad, map_name, a.ticks, fix, a.warmup, a.seconds)
        results.append(r)
        print(f"{n:3d} envs: {r['decisions_per_s']:6d} decisions/s  ({r['tics_per_s']} tics/s, "
              f"{r['per_env']}/env, {r['episodes']} episodes, failed workers {r['failed_workers']})"
              + (f"  errors: {r['errors']}" if r["errors"] else ""))
    if a.out:
        Path(a.out).write_text(json.dumps({"info": info, "endings": endings, "results": results}, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
