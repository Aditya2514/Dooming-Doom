"""Collection workers for DAgger (training only). One process per engine; no torch here (RAM).

A worker owns one official DoomEnv (via MapRunner: in-engine map switching + the Windows IPC fix),
the SAME observation adapter the submitted agent uses, and the privileged expert. Per step it
returns the network inputs for the current state and the expert's label for that state; the
parent decides which action is actually executed (expert or student: that's DAgger).

Protocol (multiprocessing Pipe), parent -> worker:
    ("reset", None)        start the first episode
    ("step", action_idx)   execute a macro-action
    ("close", None)
worker -> parent, after every command:
    (x, label, ended, r)   x: adapter output (numpy dict) of the CURRENT state, label: expert action
                           for it, ended: None, or the finished episode's stats when the step ended
                           an episode (x is then the first state of the next episode), r: reward of
                           the step just taken (training/ppo_rewards.py; 0.0 unless cfg["rewards"]).
"""
from __future__ import annotations

import math
import os
import random
import traceback

import numpy as np

STUCK_LIMIT = 250          # decisions without moving 64 units: end early (stalled episodes teach nothing)


def _schedule(cfg: dict):
    from training.maps.gen import load_index

    fams = {f: w for f, w in cfg["families"].items() if w > 0}
    pools = cfg.get("pool", "train")
    pools = [pools] if isinstance(pools, str) else list(pools)
    idx = {f: [e for pool in pools for e in load_index(pool, f)] for f in fams}
    return fams, idx


def worker(wid: int, conn, cfg: dict):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    runner = None
    try:
        from submission.doomnet import obs_adapter as A
        from submission.doomnet.actions import to_buttons
        from training.expert import Expert
        from training.maprunner import MapRunner

        rng = random.Random(cfg.get("seed", 0) * 1000 + wid)
        fams, idx = _schedule(cfg)
        names, weights = list(fams), [fams[f] for f in fams]
        runner = MapRunner()
        adapter = A.ObsAdapter()
        experts: dict = {}
        rewarder = None
        if cfg.get("rewards") is not None:
            from training.ppo_rewards import RewardConfig, Rewarder
            rewarder = Rewarder(RewardConfig(**cfg["rewards"]))
        st = {"left": 0, "fam": None}

        def new_episode():
            if st["left"] <= 0:                               # stay in one WAD for a few episodes
                st["fam"] = rng.choices(names, weights)[0]
                st["left"] = cfg.get("episodes_per_wad", 4)
            st["left"] -= 1
            entry = rng.choice(idx[st["fam"]])
            env = runner.load(entry)
            obs, _ = env.reset(seed=rng.randrange(2**31))
            adapter.reset()
            ex = experts.get(entry["label"])
            if ex is None:
                if len(experts) > 64:
                    experts.clear()
                ex = experts[entry["label"]] = Expert(entry["route"])
            ex.reset()
            ps = env.privileged_state()
            ep = {"entry": entry, "env": env, "expert": ex, "steps": 0, "cells": set(),
                  "anchor": (ps["x"], ps["y"]), "anchor_t": 0}
            return ep, obs, ps

        def observe(ep, obs, ps):
            x = adapter.observe(obs)
            x["map"] = np.packbits(x["map"] > 0.5)          # binary (4, 24, 24) -> 288 bytes
            label = ep["expert"].act(obs, ps)
            ep["cells"].add((int(ps["x"] // 128), int(ps["y"] // 128)))
            return x, label

        ep = obs = ps = None
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            if cmd == "reset":
                ep, obs, ps = new_episode()
                x, label = observe(ep, obs, ps)
                if rewarder is not None:
                    rewarder.reset(ps, x["pose"])
                conn.send((x, label, None, 0.0))
                continue
            # step
            a = int(data)
            ep["expert"].executed(a)                      # the teacher tracks what really happened
            buttons = to_buttons(a)
            adapter.record_action(buttons)
            obs, _, te, tr, _ = ep["env"].step(buttons)
            ps = ep["env"].privileged_state()
            ep["steps"] += 1
            entry = ep["entry"]
            # killing every enemy does NOT end an episode in the real engine: play on (explore)
            cleared = bool(entry.get("monsters", 0)) and ps["kills"] >= entry["monsters"]
            if math.dist(ep["anchor"], (ps["x"], ps["y"])) > 64:
                ep["anchor"], ep["anchor_t"] = (ps["x"], ps["y"]), ep["steps"]
            stalled = ep["steps"] - ep["anchor_t"] > STUCK_LIMIT
            if te or tr or stalled:
                stats = {"map": entry["label"], "family": entry.get("family", "?"), "variant": entry.get("variant"),
                         "steps": ep["steps"], "exit": bool(ps["exit_reached"]), "dead": bool(ps["dead"]),
                         "kills": int(ps["kills"]), "monsters": int(entry.get("monsters", 0)),
                         "keys": float(np.sum(ps["keys"])), "n_keys": len(entry["route"].get("keys", [])),
                         "health": float(ps["health"]), "cells": len(ep["cells"]), "stalled": bool(stalled),
                         "cleared": bool(cleared)}
                r = 0.0
                if rewarder is not None:
                    r = rewarder.step(ps, None)
                    stats["ret"] = sum(rewarder.terms.values())
                    stats.update({f"r_{k}": v for k, v in rewarder.terms.items()})
                    stats["doors"] = rewarder.met.counts["doors"] + rewarder.met.counts["locked_doors"]
                ep, obs, ps = new_episode()
                x, label = observe(ep, obs, ps)
                if rewarder is not None:
                    rewarder.reset(ps, x["pose"])
                conn.send((x, label, stats, r))
            else:
                x, label = observe(ep, obs, ps)
                r = rewarder.step(ps, x["pose"]) if rewarder is not None else 0.0
                conn.send((x, label, None, r))
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception:
        traceback.print_exc()
        try:
            conn.send(("error", traceback.format_exc(), None))
        except Exception:
            pass
    finally:
        if runner is not None:
            runner.close()
