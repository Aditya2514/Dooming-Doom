"""Evaluation runner + local proxy scorer (Spec v2.1, Evaluation protocol / Local proxy scorer).

    .venv\\Scripts\\python.exe -m training.evaluate runs/base/latest.pt --family combat --pool validation
    .venv\\Scripts\\python.exe -m training.evaluate runs/base/latest.pt --family combat rooms --seeds 2 --out eval.json

* Every episode runs through the SUBMITTED inference path: submission/agent.py DoomAgent on
  1 CPU thread, with the checkpoint's actor weights. What we measure is what we ship.
* Matched episodes: the episode list is (map, seed) pairs from the index files, identical
  for every model, so comparisons are paired.
* Worker processes each hold one DoomAgent + one engine; episodes are distributed over them.
* Per-group results: counts, rates with Wilson 95% intervals, kill fraction, proxy score
  (mean and 95% CI), median time to exit (successful episodes only).

Proxy score: our reimplementation of the published scoring categories with OUR weights.
Its job is ranking checkpoints and catching regressions, not predicting the official score.
    exit        100 + 10 x health fraction + 10 x time-left fraction
    progression +10 per keycard, +3 per door opened, +10 per locked door opened
    exploration +0.5 per first visit to a 128-unit cell (cap 30)
    combat      +5 per kill, -0.05 per point of damage taken, -20 x (1 - time fraction) if dead
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

PROXY = {"exit": 100.0, "exit_health": 10.0, "exit_time": 10.0, "key": 10.0, "door": 3.0, "locked_door": 10.0,
         "cell": 0.5, "cell_cap": 30.0, "kill": 5.0, "damage": -0.05, "death": -20.0}


def proxy_score(ep: dict) -> float:
    p = PROXY
    frac = min(1.0, ep["tic"] / max(1, ep["ticks"]))
    s = 0.0
    if ep["exit"]:
        s += p["exit"] + p["exit_health"] * min(max(ep["health"], 0), 100) / 100 + p["exit_time"] * (1 - frac)
    s += p["key"] * ep["keys"] + p["door"] * ep["doors"] + p["locked_door"] * ep["locked_doors"]
    s += min(p["cell"] * ep["cells"], p["cell_cap"])
    s += p["kill"] * ep["kills"] + p["damage"] * ep["damage"]
    if ep["death"]:
        s += p["death"] * (1 - frac)
    return s


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    m = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


# ------------------------------------------------------------------- worker side
_W: dict = {}


def _init(weights: str, device: str, ablation: str | None = None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("submission_agent_eval", REPO / "submission" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_agent_eval"] = mod
    spec.loader.exec_module(mod)
    mod.WEIGHTS_FILE = Path(weights)
    from doom_env import observation_space, action_space
    from training.vec_env import MapRunner

    mod.DoomAgent.ablation = ablation
    _W["agent"] = mod.DoomAgent(observation_space(), action_space(), device)
    _W["runner"] = MapRunner()


def _episode(task: dict) -> dict:
    from training.rewards import RewardShaper

    entry, seed = task["entry"], task["seed"]
    agent, runner = _W["agent"], _W["runner"]
    env = runner.load(entry)
    obs, _ = env.reset(seed=seed)
    agent.reset()
    ps = env.privileged_state()
    sh = RewardShaper()
    sh.set_exit(entry.get("exit_xy"))
    sh.reset(ps)
    monsters, n, t_act, fist, wsw = int(entry.get("monsters", 0)), 0, 0.0, 0, 0
    while True:
        t0 = time.perf_counter()
        a = agent.act(obs)
        t_act += time.perf_counter() - t0
        fist += int(obs["vitals"][2] <= 1)
        wsw += int(a[8])
        obs, _, te, tr, _ = env.step(a)
        ps = env.privileged_state()
        sh.step(ps, a)
        n += 1
        cleared = entry.get("end_on_clear") and monsters and ps["kills"] >= monsters
        if te or tr or cleared:
            break
    return {"map": entry.get("label", entry["map"]), "family": entry.get("family", "?"),
            "variant": entry.get("variant"), "seed": seed, "ticks": int(entry["ticks"]), "tic": ps["tic"],
            "steps": n, "exit": bool(ps["exit_reached"]), "death": bool(ps["dead"]), "tick_cap": bool(ps["timeout"]),
            "cleared": bool(cleared), "health": ps["health"], "kills": ps["kills"], "monsters": monsters,
            "kill_frac": ps["kills"] / monsters if monsters else 0.0, "damage": ps["damage_taken"],
            "keys": sh.counts["keys"], "doors": sh.counts["doors"], "locked_doors": sh.counts["locked_doors"],
            "cells": len(sh.cells), "act_ms": 1000 * t_act / max(1, n), "fist_frac": fist / max(1, n),
            "weapon_presses": wsw / max(1, n), "ammo_end": float(obs["vitals"][4])}


# --------------------------------------------------------------------- summary
def summarize(eps: list[dict]) -> dict:
    n = len(eps)
    out = {"n": n, "maps": len({e["map"] for e in eps})}
    for k in ("exit", "death", "tick_cap", "cleared"):
        c = sum(e[k] for e in eps)
        lo, hi = wilson(c, n)
        out[k] = {"rate": c / n if n else 0.0, "ci95": [round(lo, 3), round(hi, 3)], "count": c}
    scores = [e["proxy"] for e in eps]
    sd = statistics.stdev(scores) if n > 1 else 0.0
    out["proxy"] = {"mean": statistics.fmean(scores) if n else 0.0, "ci95_half": 1.96 * sd / math.sqrt(n) if n else 0.0}
    with_mon = [e for e in eps if e["monsters"]]
    out["kill_frac"] = statistics.fmean(e["kill_frac"] for e in with_mon) if with_mon else None
    out["survival"] = 1 - out["death"]["rate"]
    wins = [e["tic"] / 35.0 for e in eps if e["exit"]]
    out["median_time_to_exit_s"] = statistics.median(wins) if wins else None
    for k in ("doors", "locked_doors", "keys", "cells", "act_ms", "fist_frac", "weapon_presses", "ammo_end"):
        out[k] = statistics.fmean(e.get(k, 0.0) for e in eps) if n else 0.0
    return out


def actor_weights(ckpt: str, greedy: bool) -> str:
    """Accept a training checkpoint (actor extracted to a temp file) or an exported actor.pt."""
    import torch

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in d:
        d.setdefault("meta", {})["greedy"] = greedy
        sd, meta = d["state_dict"], d["meta"]
    else:
        sd = {k[len("actor."):]: v for k, v in d["model"].items() if k.startswith("actor.")}
        meta = {"greedy": greedy, "source": ckpt, "global_step": d.get("global_step"),
                "arch": "memory" if "mem_query.weight" in sd else "baseline"}
    f = Path(tempfile.gettempdir()) / f"eval_actor_{os.getpid()}.pt"
    torch.save({"state_dict": sd, "meta": meta}, f)
    return str(f)


def episode_list(families: list[str], pool: str, seeds: int, limit: int | None) -> list[dict]:
    from training.maps.gen import load_index

    tasks = []
    for fam in families:
        entries = load_index(pool, fam)[:limit] if limit else load_index(pool, fam)
        for s in range(seeds):
            tasks += [{"entry": e, "seed": 10_000 + s} for e in entries]
    return tasks


def evaluate(ckpt: str, families: list[str], pool: str = "validation", seeds: int = 2, workers: int = 6,
             greedy: bool = True, limit: int | None = None, device: str = "cpu", ablation: str | None = None) -> dict:
    tasks = episode_list(families, pool, seeds, limit)
    weights = actor_weights(ckpt, greedy)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(weights, device, ablation)) as pool_:
        eps = list(pool_.imap_unordered(_episode, tasks))
    for e in eps:
        e["proxy"] = proxy_score(e)
    eps.sort(key=lambda e: (e["family"], e["map"], e["seed"]))
    groups = {"all": summarize(eps)}
    for fam in families:
        groups[fam] = summarize([e for e in eps if e["family"] == fam])
        for v in sorted({e["variant"] for e in eps if e["family"] == fam and e["variant"]}):
            groups[f"{fam}/{v}"] = summarize([e for e in eps if e["family"] == fam and e["variant"] == v])
    return {"ckpt": ckpt, "pool": pool, "mode": "threshold" if greedy else "sample", "seeds": seeds,
            "ablation": ablation,
            "seconds": round(time.time() - t0, 1), "groups": groups, "episodes": eps}


def print_report(r: dict) -> None:
    print(f"\n{r['ckpt']}  pool={r['pool']}  mode={r['mode']}  ablation={r.get('ablation')}  ({r['seconds']} s)")
    print(f"{'group':18s} {'n':>4s} {'exit':>16s} {'survival':>9s} {'killfrac':>9s} {'proxy':>14s} "
          f"{'doors':>6s} {'cells':>6s} {'t_exit':>7s} {'act ms':>7s}")
    for g, s in r["groups"].items():
        ex = s["exit"]
        kf = "-" if s["kill_frac"] is None else f"{s['kill_frac']:.2f}"
        te = "-" if s["median_time_to_exit_s"] is None else f"{s['median_time_to_exit_s']:.1f}s"
        print(f"{g:18s} {s['n']:4d} {ex['rate']:5.2f} [{ex['ci95'][0]:.2f},{ex['ci95'][1]:.2f}] "
              f"{s['survival']:9.2f} {kf:>9s} {s['proxy']['mean']:7.1f}±{s['proxy']['ci95_half']:<5.1f} "
              f"{s['doors']:6.2f} {s['cells']:6.1f} {te:>7s} {s['act_ms']:7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="training checkpoint (runs/<run>/latest.pt) or exported actor.pt")
    ap.add_argument("--family", nargs="+", default=["combat"])
    ap.add_argument("--pool", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--seeds", type=int, default=2, help="episodes per map (matched seeds)")
    ap.add_argument("--limit", type=int, default=None, help="first N maps per family")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--sample", action="store_true", help="sampling mode (default: threshold at 0.5)")
    ap.add_argument("--ablation", default=None, choices=["memory_off", "clear_at_key", "foreign", "no_inv_query"],
                    help="memory interventions (memory checkpoints only; Spec v2.1 Experiments)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.pool == "test":
        print("NOTE: the test pool is opened ONCE, after the candidate is frozen (Spec v2.1, Phase 5).")
    r = evaluate(a.ckpt, a.family, a.pool, a.seeds, a.workers, greedy=not a.sample, limit=a.limit,
                 ablation=a.ablation)
    print_report(r)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1, default=float))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
