"""Training entry point (Phase 1: baseline PPO on a fixed map schedule; the curriculum
controller plugs in here in Phase 4).

    .venv\\Scripts\\python.exe -m training.train --run base_s1 --maps maps/two_rooms.wad:MAP01:2100
    .venv\\Scripts\\python.exe -m training.train --run base_s1 --resume auto          # continue
    .venv\\Scripts\\python.exe -m training.train --bench --envs 8 12 16               # full-loop benchmark

Writes runs/<run>/: config.json, train_log.jsonl (one row per update + events),
latest.pt (every --save-min minutes and on stop), ckpt_<steps>.pt snapshots.
torch is imported inside main() so spawned env workers never load it (Windows spawn).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_maps(specs: list[str]) -> list[dict]:
    """'wad:MAP01:2100' entries, 'freedoom2' for the IWAD, or a JSON file with a list of
    {"wad", "map", "ticks"[, "label"]}."""
    out = []
    for s in specs:
        if s.endswith(".json"):
            for e in json.loads(Path(s).read_text()):
                if not os.path.isabs(e["wad"]):
                    e["wad"] = str(REPO / e["wad"])
                out.append(e)
            continue
        parts = s.rsplit(":", 2)
        wad, name, ticks = parts if len(parts) == 3 else (parts[0], parts[1] if len(parts) > 1 else "MAP01", 2100)
        if wad.lower() == "freedoom2":
            from doom_env.wrapper import IWAD
            wad = IWAD
        elif not os.path.isabs(wad):
            wad = str(REPO / wad)
        out.append({"wad": wad, "map": name, "ticks": int(ticks)})
    return out


def episode_summary(eps: list[dict]) -> dict:
    if not eps:
        return {}
    m = lambda k: float(sum(float(e[k]) for e in eps) / len(eps))  # noqa: E731
    return {"ep_n": len(eps), "ep_return": m("return"), "ep_length": m("length"), "exit_rate": m("exit"),
            "death_rate": m("death"), "kills": m("kills"), "keys": m("keys"), "doors": m("doors"),
            "locked_doors": m("locked_doors"), "explore_cells": m("explore_cells"),
            "survival": 1.0 - m("death"), "kill_frac": m("kill_frac"), "cleared": m("cleared")}


def bench(args):
    import torch
    from training.model import ActorCritic
    from training.ppo import PPO, PPOConfig
    from training.vec_env import DoomVecEnv

    maps = parse_maps(args.maps)
    rows = []
    for n in args.envs:
        torch.manual_seed(0)
        env = DoomVecEnv(maps, n_envs=n, seed=args.seed)
        try:
            ppo = PPO(ActorCritic(), env, PPOConfig(n_envs=n, n_steps=256, device=args.device, total_steps=10**9),
                      DoomVecEnv.obs_spec)
            ts = []
            for i in range(args.bench_updates):
                t0 = time.time()
                c = ppo.collect()
                u = ppo.update()
                ts.append((time.time() - t0, c["collect_s"], c["env_s"], u["update_s"]))
            warm = ts[2:] or ts
            tot = sum(t[0] for t in warm)
            r = {"envs": n, "dps": round(256 * n * len(warm) / tot),
                 "collect_s": round(sum(t[1] for t in warm) / len(warm), 2),
                 "env_s": round(sum(t[2] for t in warm) / len(warm), 2),
                 "update_s": round(sum(t[3] for t in warm) / len(warm), 2),
                 "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0}
            rows.append(r)
            print(f"{n:3d} envs: {r['dps']:5d} decisions/s end to end | per update: collect {r['collect_s']} s "
                  f"(env {r['env_s']} s), update {r['update_s']} s | peak VRAM {r['peak_vram_mb']} MB", flush=True)
        finally:
            env.close()
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


def warm_start(ppo, path: str) -> None:
    """Memory model from a baseline checkpoint (plan: single run, switch at T+6).
    Actor: shared weights copied, fusion memory columns zero (identical outputs at the switch).
    Critic, aux head, reward statistics and the decision count carry over; the optimizer
    starts fresh (new parameters)."""
    import torch

    d = torch.load(path, map_location=ppo.device, weights_only=False)
    actor_sd = {k[len("actor."):]: v for k, v in d["model"].items() if k.startswith("actor.")}
    if "mem_query.weight" in actor_sd:
        raise SystemExit("--warm-start expects a BASELINE checkpoint")
    ppo.model.actor.load_baseline(actor_sd)
    rest = {k: v for k, v in d["model"].items() if not k.startswith("actor.")}
    ppo.model.load_state_dict({**ppo.model.state_dict(), **rest})
    ppo.scaler.rms.load_state_dict(d["reward_rms"])
    ppo.global_step, ppo.n_updates = d["global_step"], d["n_updates"]
    ppo.meta.update({"warm_start_from": path, "warm_start_step": d["global_step"]})
    print(f"warm start from {path} at {ppo.global_step:,} decisions (memory columns zero)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="dev")
    ap.add_argument("--maps", nargs="+", default=["maps/two_rooms.wad:MAP01:2100"])
    ap.add_argument("--envs", type=int, nargs="+", default=[14])
    ap.add_argument("--budget", type=int, default=40_000_000, help="whole-run decisions (entropy schedule + stop)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--cutoff", type=int, default=None, help="our own episode cutoff in decisions (bootstrapped)")
    ap.add_argument("--no-doors", action="store_true",
                    help="disable door rewards (enabled: tests/test_door_reward.py passes on the real engine)")
    ap.add_argument("--arch", default="baseline", choices=["baseline", "memory"])
    ap.add_argument("--warm-start", default=None,
                    help="memory run, used only while runs/<run>/latest.pt doesn't exist: start from this "
                         "baseline checkpoint (shared weights copied, memory columns zero -> same outputs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default=None, help="'auto' (runs/<run>/latest.pt if present) or a path")
    ap.add_argument("--hours", type=float, default=None, help="stop after this many hours")
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--save-min", type=float, default=30.0)
    ap.add_argument("--snapshot-every", type=int, default=1_000_000, help="decisions between ckpt_*.pt")
    ap.add_argument("--bench", action="store_true", help="full-loop throughput benchmark, then exit")
    ap.add_argument("--bench-updates", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.set_num_threads(2)
    if args.bench:
        return bench(args)

    from training.model import ActorCritic, MemoryActor
    from training.ppo import PPO, PPOConfig
    from training.vec_env import DoomVecEnv

    run = REPO / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    maps = parse_maps(args.maps)
    n_envs = args.envs[0]
    torch.manual_seed(args.seed)
    rewards = {"doors_enabled": not args.no_doors}
    cfg = PPOConfig(n_envs=n_envs, gamma=args.gamma, total_steps=args.budget, seed=args.seed, device=args.device)
    (run / "config.json").write_text(json.dumps({"args": vars(args), "maps": maps, "ppo": cfg.__dict__,
                                                  "rewards": rewards}, indent=1, default=str))
    log_f = open(run / "train_log.jsonl", "a", buffering=1)

    def log(row):
        log_f.write(json.dumps(row, default=float) + "\n")

    env = DoomVecEnv(maps, n_envs=n_envs, seed=args.seed, rewards=rewards, cutoff=args.cutoff)
    model = ActorCritic(MemoryActor()) if args.arch == "memory" else ActorCritic()
    ppo = PPO(model, env, cfg, DoomVecEnv.obs_spec)
    ppo.meta["arch"] = args.arch
    if env.torch_in_workers:
        print("WARNING: torch was imported in env workers (check the entry script's top-level imports)")
    resume = run / "latest.pt" if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    if resume is not None and resume.exists():
        ppo.load(resume)
        ppo.set_gamma(args.gamma)
        print(f"resumed {resume} at {ppo.global_step:,} decisions (update {ppo.n_updates})")
        log({"event": "resume", "path": str(resume), "step": ppo.global_step, "t": time.time()})
    elif args.warm_start:
        warm_start(ppo, args.warm_start)
        log({"event": "warm_start", "from": args.warm_start, "step": ppo.global_step, "t": time.time()})
    elif args.resume == "auto":
        print("no checkpoint to resume: starting fresh")
    log({"event": "start", "step": ppo.global_step, "n_envs": n_envs, "device": args.device, "t": time.time()})

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    t_start, t_save = time.time(), time.time()
    next_snap = (ppo.global_step // args.snapshot_every + 1) * args.snapshot_every
    first = True
    print(f"[{args.run}] {n_envs} envs on {args.device}, {len(maps)} map(s), budget {args.budget:,}")
    try:
        while not stop["flag"] and ppo.global_step < args.budget:
            t0 = time.time()
            c = ppo.collect()
            if first:
                rc = ppo.recompute_check()
                log({"event": "recompute_check", **rc})
                print(f"recompute check: {'OK' if rc['ok'] else 'FAILED'} (max err logp {rc['max_err_logp']:.2e}, "
                      f"value {rc['max_err_value']:.2e})")
                if not rc["ok"]:
                    raise SystemExit("recompute check failed: stopping before any update")
                first = False
            u = ppo.update()
            eps = list(ppo.ep_stats)
            row = {"step": ppo.global_step, "update": ppo.n_updates, "t": time.time(),
                   "dps": 256 * n_envs / (time.time() - t0), **c, **u, **episode_summary(eps)}
            log(row)
            if ppo.n_updates % 5 == 0 or ppo.n_updates < 3:
                print(f"[{args.run}] {ppo.global_step:>10,}  {row['dps']:5.0f} dps  ret {row.get('ep_return', 0):7.2f}  "
                      f"exit {row.get('exit_rate', 0):.2f}  death {row.get('death_rate', 0):.2f}  "
                      f"killfrac {row.get('kill_frac', 0):.2f}  cells {row.get('explore_cells', 0):5.1f}  "
                      f"len {row.get('ep_length', 0):5.0f}  kl {row.get('kl', 0):.4f}  ent {row.get('loss/entropy', 0):.2f}",
                      flush=True)
            if time.time() - t_save > args.save_min * 60:
                ppo.save(run / "latest.pt")
                t_save = time.time()
            if ppo.global_step >= next_snap:
                ppo.save(run / f"ckpt_{ppo.global_step // 1000}k.pt")
                next_snap += args.snapshot_every
            if args.hours and time.time() - t_start > args.hours * 3600:
                break
            if args.max_updates and ppo.n_updates >= args.max_updates:
                break
    finally:
        ppo.save(run / "latest.pt")
        log({"event": "stop", "step": ppo.global_step, "t": time.time()})
        print(f"saved {run / 'latest.pt'} at {ppo.global_step:,} decisions")
        env.close()
        log_f.close()


if __name__ == "__main__":
    main()
