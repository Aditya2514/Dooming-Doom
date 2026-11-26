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
            "survival": 1.0 - m("death"), "kill_frac": m("kill_frac"), "cleared": m("cleared"),
            "fist_frac": m("fist_frac"), "ammo_end": m("ammo_end"), "weapon_presses": m("weapon_presses"),
            "r_ammo": m("r_ammo"), "r_kill": m("r_kill"), "r_exit": m("r_exit"), "r_explore": m("r_explore"),
            "r_stuck": m("r_stuck"), "stuck_frac": sum(float(e["stuck"]) / max(1, e["length"]) for e in eps) / len(eps)}


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
    ap.add_argument("--curriculum", action="store_true",
                    help="stages 1-4 with gates and deadlines (training/curriculum.py); ignores --maps/--gamma")
    ap.add_argument("--t0", default=None, help="T for the stage deadlines, local time 'YYYY-MM-DDTHH:MM' "
                                               "(stored in runs/<run>/curriculum.json on first use)")
    ap.add_argument("--start-stage", default="1", choices=["1", "2", "3", "4"])
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--reapply-weapon-prior", action="store_true",
                    help="apply --weapon-prior once more (recorded separately: never repeats either)")
    ap.add_argument("--weapon-prior", type=float, default=None,
                    help="ONCE per run: set the weapon-switch logit bias of a resumed/warm-started model to this "
                         "value (e.g. -4: pressed ~2%%); recorded in the checkpoint so it is never re-applied")
    ap.add_argument("--no-gate-video", dest="gate_video", action="store_false",
                    help="don't record a validation video at each gate (runs/<run>/videos/)")
    ap.add_argument("--gate-mode", default="sample", choices=["sample", "threshold"],
                    help="action mode of gate evals: 'sample' matches training (the frozen submission mode "
                         "is chosen in Phase 5 from both)")
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
    cur = None
    if args.curriculum:
        from datetime import datetime

        from training.curriculum import STAGES, Curriculum, build_worker_schedules
        t0 = datetime.fromisoformat(args.t0).timestamp() if args.t0 else None
        cur = Curriculum(run / "curriculum.json", t0=t0, start_stage=args.start_stage)
        maps = build_worker_schedules(cur.stage, args.envs[0], args.seed)
        args.gamma = STAGES[cur.stage]["gamma"]
        fam_of = [w[0]["family"] for w in maps]
        mix = {f: fam_of.count(f) for f in dict.fromkeys(fam_of)}
        print(f"curriculum: stage {cur.stage} ({STAGES[cur.stage]['family']}), T+{cur.hours():.1f} h, "
              f"workers per family {mix}, gamma {args.gamma}")
    else:
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
    if args.weapon_prior is not None and args.reapply_weapon_prior and not ppo.meta.get("weapon_prior_reapplied"):
        ppo.meta["weapon_prior_reapplied"] = True
        ppo.meta.pop("weapon_prior_applied", None)          # let the block below apply it once more
    if args.weapon_prior is not None and not ppo.meta.get("weapon_prior_applied"):
        import torch as _t
        from submission.doomnet.model import WEAPON_BUTTON
        with _t.no_grad():
            b = ppo.model.actor.policy[-1].bias
            old = float(b[WEAPON_BUTTON])
            b[WEAPON_BUTTON] = args.weapon_prior
            W = ppo.model.actor.policy[-1].weight
            w_norm = float(W[WEAPON_BUTTON].norm())
            W[WEAPON_BUTTON].zero_()        # state-independent again: training had raised the logit via
                                            # the weights (bias stayed ~-3.9 while presses reached 0.42)
        ppo.meta["weapon_prior_applied"] = {"step": ppo.global_step, "from": old, "to": args.weapon_prior,
                                            "weight_norm_reset": w_norm}
        print(f"weapon-switch prior: bias {old:.2f} -> {args.weapon_prior}, weights |w| {w_norm:.2f} -> 0 "
              f"(once, at {ppo.global_step:,} decisions)")
        log({"event": "weapon_prior", "step": ppo.global_step, "from": old, "to": args.weapon_prior, "t": time.time()})
    if cur is not None:
        cur.begin(ppo.global_step)
        ppo.set_gamma(STAGES[cur.stage]["gamma"])

    def gate_eval():
        from training.evaluate import evaluate
        ppo.save(run / "gate.pt")
        st = STAGES[cur.stage]
        t_e = time.time()
        fams = [st["family"]] + list(st.get("monitor", []))      # monitors: logged, never gate
        r = evaluate(str(run / "gate.pt"), fams, "validation", st["seeds"], args.eval_workers,
                     greedy=args.gate_mode == "threshold")
        g = r["groups"]
        log({"event": "gate_eval", "stage": cur.stage, "step": ppo.global_step, "seconds": time.time() - t_e,
             "mode": args.gate_mode,
             "groups": {k: {"n": v["n"], "exit": v["exit"]["rate"], "survival": v["survival"],
                            "kill_frac": v["kill_frac"], "proxy": v["proxy"]["mean"]} for k, v in g.items()}})
        for mfam in st.get("monitor", []):
            if mfam in g:
                print(f"[{args.run}] monitor {mfam}: exit {g[mfam]['exit']['rate']:.2f} on validation", flush=True)
        if args.gate_video:
            gate_video(st["family"])
        return g

    def gate_video(family: str):
        """One validation episode per gate, recorded in a background process (training doesn't wait).
        Rotates through the validation maps; snapshot of the gate weights so later gates can't race it."""
        import shutil
        import subprocess
        import sys
        vids = run / "videos"
        vids.mkdir(exist_ok=True)
        k = len(list(vids.glob("*.mp4")))
        snap = vids / "_gate_snapshot.pt"
        shutil.copy2(run / "gate.pt", snap)
        out = vids / f"{ppo.global_step // 1000:07d}k_{family}_v{k % 20:02d}.mp4"
        cmd = [sys.executable, "-m", "training.record", str(snap), "--family", family, "--pool", "validation",
               "--index", str(k % 20), "--seed", str(10_000 + k), "--out", str(out)]
        cmd += ["--sample"] if args.gate_mode == "sample" else []
        with open(vids / "record.log", "a") as lf:
            subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
        print(f"[{args.run}] recording {out.name} in the background", flush=True)

    def after_decision(d):
        if d.get("improved"):
            ppo.save(run / f"best_stage{d.get('from', cur.stage)}.pt")
        print(f"[{args.run}] curriculum @ {ppo.global_step:,}: {d['action']} - {d['why']}", flush=True)
        if d["action"] == "advance":
            ppo.save(run / f"stage{d['from']}_end.pt")
            env.configure(schedule=build_worker_schedules(d["to"], n_envs, args.seed))   # at next episode
            ppo.set_gamma(STAGES[d["to"]]["gamma"])
            print(f"[{args.run}] now stage {d['to']} ({STAGES[d['to']]['family']}), gamma {STAGES[d['to']]['gamma']}",
                  flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    t_start, t_save = time.time(), time.time()
    next_snap = (ppo.global_step // args.snapshot_every + 1) * args.snapshot_every
    first = True
    flat = [e for w in maps for e in w] if maps and isinstance(maps[0], list) else maps
    n_maps = len({(e["wad"], e["map"]) for e in flat})
    print(f"[{args.run}] {n_envs} envs on {args.device}, {n_maps} map(s), budget {args.budget:,}")
    try:
        while not stop["flag"] and ppo.global_step < args.budget:
            t0 = time.time()
            prev_step = ppo.global_step
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
                      f"killfrac {row.get('kill_frac', 0):.2f}  fist {row.get('fist_frac', 0):.2f}  "
                      f"stuck {row.get('stuck_frac', 0):.2f}  "
                      f"cells {row.get('explore_cells', 0):5.1f}  "
                      f"len {row.get('ep_length', 0):5.0f}  kl {row.get('kl', 0):.4f}  ent {row.get('loss/entropy', 0):.2f}",
                      flush=True)
            if cur is not None and cur.gate_due(prev_step, ppo.global_step):
                stage_before = cur.stage
                d = cur.decide(ppo.global_step, gate_eval())
                d.setdefault("from", stage_before)
                after_decision(d)
            elif cur is not None and cur.deadline_passed():
                after_decision(cur.decide(ppo.global_step, None))
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
