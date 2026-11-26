"""Phase 5: selection, freeze, test pool, latency (Spec v2.1, Evaluation protocol + Phase 5).

    # 1. rank candidates in BOTH action modes on matched validation episodes (all families)
    .venv\\Scripts\\python.exe -m training.select rank runs/mem/best_stage*.pt runs/mem/latest.pt runs/base/latest.pt
    # 2. freeze the winner: export to submission/weights/actor.pt, record hashes (no tuning afterwards)
    .venv\\Scripts\\python.exe -m training.select freeze
    # 3. open the test pool ONCE (refuses a second run)
    .venv\\Scripts\\python.exe -m training.select test
    # 4. latency: full episodes through DoomAgent on 1 CPU thread vs the 40 ms / 5 s limits
    .venv\\Scripts\\python.exe -m training.select latency

Frozen selection rule (plan, decided before seeing results): the candidate x mode with the best
TIER-WEIGHTED validation proxy score wins; ties (within 0.5) go to the higher key-door variant B
exit rate. Tier weights [guessing; the official weights are private]: combat 1, rooms 2,
keydoor 3, mixed 4, freedoom 4 - harder tiers weigh more, as in the official scoring.
Results are preliminary (Wilson intervals, counts reported); the test pool is touched once.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "selection"
FAMILIES = ["combat", "rooms", "keydoor", "mixed", "freedoom"]
TIER_WEIGHT = {"combat": 1.0, "rooms": 2.0, "keydoor": 3.0, "mixed": 4.0, "freedoom": 4.0}
TIE = 0.5


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def weighted_score(groups: dict) -> float:
    num = sum(TIER_WEIGHT[f] * groups[f]["proxy"]["mean"] for f in FAMILIES if f in groups)
    den = sum(TIER_WEIGHT[f] for f in FAMILIES if f in groups)
    return num / den if den else float("-inf")


def variant_b(groups: dict) -> float:
    g = groups.get("keydoor/B")
    return g["exit"]["rate"] if g else 0.0


def cmd_rank(a):
    from training.evaluate import evaluate

    cands = []
    for pat in a.ckpts:
        cands += sorted(glob.glob(pat)) or [pat]
    cands = [c for c in dict.fromkeys(cands) if Path(c).exists()]
    if not cands:
        raise SystemExit("no candidate checkpoints found")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cands:
        for greedy in (True, False):
            mode = "threshold" if greedy else "sample"
            r = evaluate(c, FAMILIES, "validation", a.seeds, a.workers, greedy=greedy, limit=a.limit)
            g = r["groups"]
            row = {"ckpt": c, "sha256": sha256(Path(c)), "mode": mode, "score": weighted_score(g),
                   "variant_b": variant_b(g), "seconds": r["seconds"],
                   "families": {f: {"n": g[f]["n"], "exit": g[f]["exit"]["rate"], "exit_ci95": g[f]["exit"]["ci95"],
                                    "survival": g[f]["survival"], "kill_frac": g[f]["kill_frac"],
                                    "proxy": g[f]["proxy"]["mean"], "proxy_ci95_half": g[f]["proxy"]["ci95_half"]}
                                for f in FAMILIES if f in g},
                   "variants": {k: g[k]["exit"]["rate"] for k in g if "/" in k}}
            rows.append(row)
            print(f"{c:42s} {mode:9s} score {row['score']:7.2f}  "
                  + "  ".join(f"{f[:4]} x{row['families'][f]['exit']:.2f}/p{row['families'][f]['proxy']:.0f}"
                              for f in FAMILIES if f in row["families"]) + f"  B {row['variant_b']:.2f}", flush=True)
    best = max(rows, key=lambda r: r["score"])
    tied = [r for r in rows if best["score"] - r["score"] <= TIE]
    choice = max(tied, key=lambda r: (r["variant_b"], r["score"]))
    sel = {"time": time.time(), "rule": "max tier-weighted validation proxy; ties (<=0.5) -> keydoor B exit",
           "tier_weights": TIER_WEIGHT, "seeds": a.seeds, "choice": choice, "all": rows}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1, default=float))
    print(f"\nCHOSEN: {choice['ckpt']} ({choice['mode']}), score {choice['score']:.2f}, "
          f"variant B exit {choice['variant_b']:.2f} -> {OUT / 'selection.json'}")


def cmd_freeze(a):
    sel = json.loads((OUT / "selection.json").read_text())
    ch = sel["choice"]
    if sha256(Path(ch["ckpt"])) != ch["sha256"]:
        raise SystemExit(f"{ch['ckpt']} changed since ranking: re-run 'rank' (a live run may have overwritten it)")
    frozen_ckpt = OUT / f"frozen_{ch['sha256'][:12]}.pt"
    shutil.copy2(ch["ckpt"], frozen_ckpt)                 # immune to the training run overwriting latest.pt
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "training.export", str(frozen_ckpt)] + (["--sample"] if ch["mode"] == "sample" else [])
    subprocess.run(cmd, check=True, cwd=REPO)
    w = REPO / "submission" / "weights" / "actor.pt"
    frozen = {"source_ckpt": ch["ckpt"], "source_sha256": ch["sha256"], "frozen_copy": str(frozen_ckpt),
              "mode": ch["mode"], "validation_score": ch["score"], "variant_b": ch["variant_b"],
              "weights_sha256": sha256(w), "time": time.time()}
    (OUT / "frozen.json").write_text(json.dumps(frozen, indent=1))
    print(f"FROZEN: {w} (sha256 {frozen['weights_sha256'][:16]}), mode {ch['mode']} -> {OUT / 'frozen.json'}")


def cmd_test(a):
    from training.evaluate import evaluate

    frozen = json.loads((OUT / "frozen.json").read_text())
    w = REPO / "submission" / "weights" / "actor.pt"
    if sha256(w) != frozen["weights_sha256"]:
        raise SystemExit("submission/weights/actor.pt differs from the frozen weights")
    marker = OUT / f"test_{frozen['weights_sha256'][:12]}.json"
    if marker.exists() and not a.force:
        raise SystemExit(f"test pool already opened for these weights: {marker} (the test pool is touched ONCE)")
    r = evaluate(str(w), FAMILIES, "test", a.seeds, a.workers, greedy=frozen["mode"] == "threshold")
    from training.evaluate import print_report
    print_report(r)
    r.pop("episodes", None) if not a.keep_episodes else None
    marker.write_text(json.dumps({"frozen": frozen, "report": r}, indent=1, default=float))
    print(f"test report (preliminary: counts + Wilson 95% intervals) -> {marker}")


def cmd_latency(a):
    """Full episodes through DoomAgent on 1 CPU thread (eval sandbox: 1 thread, 40 ms/decision budget)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from doom_env import action_space, make_env, observation_space

    from training.evaluate import _init, _W
    from training.maps.gen import load_index

    w = REPO / "submission" / "weights" / "actor.pt"
    _init(str(w), "cpu")
    agent = _W["agent"]
    _W["runner"].close()
    entries = [load_index("validation", f)[0] for f in ("rooms", "keydoor", "mixed")] + \
              [load_index("validation", "freedoom")[0]]
    times, first = [], []
    t_init = time.perf_counter()
    for e in entries[: a.episodes]:
        env = make_env(e["wad"], e["map"], episode_ticks=int(e["ticks"]), render_buffers=False)
        try:
            obs, _ = env.reset(seed=0)
            t0 = time.perf_counter()
            agent.reset()
            reset_s = time.perf_counter() - t0
            k = 0
            while True:
                t0 = time.perf_counter()
                act = agent.act(obs)
                dt = time.perf_counter() - t0
                (first if k == 0 else times).append(dt)
                obs, _, te, tr, _ = env.step(act)
                k += 1
                if te or tr:
                    break
        finally:
            env.close()
        print(f"  {e['label']}: {k} decisions, reset {1000 * reset_s:.1f} ms")
    t = np.array(times) * 1000
    res = {"episodes": len(first), "decisions": len(times) + len(first), "mean_ms": float(t.mean()),
           "p99_ms": float(np.percentile(t, 99)), "max_ms": float(t.max()),
           "first_act_max_ms": float(max(first) * 1000), "limit_avg_ms": 40.0, "limit_any_s": 5.0,
           "ok": bool(t.mean() < 40 and t.max() < 5000 and max(first) < 10)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(res, indent=1))
    print(f"latency on 1 CPU thread: mean {res['mean_ms']:.2f} ms, p99 {res['p99_ms']:.2f} ms, max {res['max_ms']:.1f} ms, "
          f"first act {res['first_act_max_ms']:.1f} ms -> {'OK' if res['ok'] else 'OVER LIMIT'} (limits: 40 ms avg, 5 s any, 10 s first)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank")
    r.add_argument("ckpts", nargs="+")
    r.add_argument("--seeds", type=int, default=1)
    r.add_argument("--limit", type=int, default=None, help="first N maps per family (quick dry runs)")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    sub.add_parser("freeze")
    t = sub.add_parser("test")
    t.add_argument("--seeds", type=int, default=2)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    t.add_argument("--force", action="store_true", help="re-open the test pool (dry runs only)")
    t.add_argument("--keep-episodes", action="store_true")
    la = sub.add_parser("latency")
    la.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args()
    {"rank": cmd_rank, "freeze": cmd_freeze, "test": cmd_test, "latency": cmd_latency}[a.cmd](a)


if __name__ == "__main__":
    main()
