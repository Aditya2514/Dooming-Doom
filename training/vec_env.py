"""DoomVecEnv: one worker process per env, official wrapper inside, shared memory, synchronous.

Each worker runs:  official DoomEnv -> shared ObsAdapter (actor inputs, identical to
DoomAgent) + RewardShaper (training-only reward, aux event, 16 critic inputs).

Implements the training/ppo.py vector-env contract with SAME-STEP auto-reset (Spec v2.1,
Vector env): on the step an episode ends, reward/term/trunc belong to the finished
episode, the returned observation is the next episode's first one, and the episode's
summary goes to info["episodes"]. Every stored transition is a real one.

Endings (F1): term = exit, death or the map's tick cap (no bootstrap); trunc = our own
cutoff (cutoff_steps), with the genuine next observation in info["final_obs"].

Map schedule: a fixed list of {"wad", "map", "ticks"}. Worker i plays entries
i, i+N, i+2N, ... in order. Or PER-WORKER lists (a list of N lists): worker i cycles
through its own list. Keep each worker's list inside ONE WAD: maps switch inside a
running engine only within the same WAD; a WAD change restarts the engine. Maps of the same WAD are switched inside the running engine
(~0.07 s vs ~0.54 s for a new engine).

Workers import only the wrapper, numpy and our numpy-only modules, never torch.
NOTE (Windows spawn): the parent's __main__ is re-imported in every worker, so entry
scripts import torch inside main().
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from submission.doomnet import obs_adapter as A

PRIV_DIM = 16
OBS_SPEC = {
    "rays": ((A.RAY_CHANNELS, A.NUM_RAYS), np.dtype(np.float32)),
    "ent_type": ((A.MAX_ENTITIES,), np.dtype(np.int64)),
    "ent_feat": ((A.MAX_ENTITIES, A.ENT_FEATURES), np.dtype(np.float32)),
    "ent_mask": ((A.MAX_ENTITIES,), np.dtype(np.bool_)),
    "scalars": ((A.NUM_SCALARS,), np.dtype(np.float32)),
    "priv": ((PRIV_DIM,), np.dtype(np.float32)),
}


# ------------------------------------------------------------------ map runner
class MapRunner:
    """Owns one official DoomEnv and moves it between maps (training only)."""

    def __init__(self, fix_ipc: bool = True):
        self.env = None
        self.fix_ipc = fix_ipc

    def load(self, entry: dict):
        from doom_env import make_env
        from doom_env.mapinfo import load_mapinfo

        wad = str(Path(entry["wad"]).resolve())          # DoomEnv stores the resolved path
        name, ticks = entry["map"].upper(), int(entry["ticks"])
        env = self.env
        if env is not None and env.wad == wad:
            if (env.map_name, env.episode_ticks) != (name, ticks):
                # same WAD: switch inside the running engine. Mirrors DoomEnv.__init__ for the
                # per-map state; geometry/sensors are rebuilt by the next reset().
                g = env._game
                if env.map_name != name:
                    g.set_doom_map(name)
                    env.map_name = name
                    env.mapinfo = load_mapinfo(env.wad, name)
                    env._geometry = None
                    env._sensors = None
                g.set_episode_timeout(ticks)
                env.episode_ticks = ticks
            return env
        if env is not None:
            env.close()
        self.env = make_env(wad, name, episode_ticks=ticks, render_buffers=False)
        if self.fix_ipc:
            from training.win_ipc import windows_fast_ipc
            windows_fast_ipc()
        return self.env

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


# ------------------------------------------------------------- shared memory
class _Shm:
    def __init__(self, n: int, create: bool, names: dict | None = None):
        self.blocks, self.arrays = {}, {}
        fields = {}
        for k, (s, d) in OBS_SPEC.items():
            fields[f"obs.{k}"] = ((n,) + s, d)
            fields[f"final.{k}"] = ((n,) + s, d)
        fields.update({"reward": ((n,), np.dtype(np.float32)), "term": ((n,), np.dtype(np.bool_)),
                       "trunc": ((n,), np.dtype(np.bool_)), "aux": ((n, 1), np.dtype(np.bool_))})
        for k, (shape, dt) in fields.items():
            size = max(1, int(np.prod(shape)) * dt.itemsize)
            b = shared_memory.SharedMemory(create=True, size=size) if create else shared_memory.SharedMemory(name=names[k])
            self.blocks[k] = b
            self.arrays[k] = np.ndarray(shape, dt, buffer=b.buf)

    @property
    def names(self):
        return {k: b.name for k, b in self.blocks.items()}

    def close(self, unlink=False):
        self.arrays.clear()
        for b in self.blocks.values():
            b.close()
            if unlink:
                b.unlink()


# ------------------------------------------------------------------- worker
def _worker(idx: int, n: int, shm_names: dict, cfg: dict, seed: int, pipe):
    shm = runner = None
    try:
        from training.rewards import RewardConfig, RewardShaper

        shm = _Shm(n, create=False, names=shm_names)
        S = shm.arrays
        runner = MapRunner(fix_ipc=cfg.get("fix_ipc", True))
        adapter = A.ObsAdapter()
        shaper = RewardShaper(RewardConfig(**cfg.get("rewards", {})))
        state = {"schedule": list(cfg["schedule"]), "own": cfg.get("own", False), "cutoff": cfg.get("cutoff"),
                 "k": 0}
        pending: dict = {}
        ep = {}

        def start_episode():
            if pending:
                if "schedule" in pending:
                    state["schedule"], state["k"] = list(pending["schedule"]), 0
                    state["own"] = pending.get("own", False)
                if "rewards" in pending:
                    shaper.cfg = RewardConfig(**pending["rewards"])
                if "cutoff" in pending:
                    state["cutoff"] = pending["cutoff"]
                pending.clear()
            sched = state["schedule"]
            if state["own"]:
                entry = sched[(idx * 13 + state["k"]) % len(sched)]
            else:
                entry = sched[(idx + state["k"] * n) % len(sched)]
            state["k"] += 1
            env = runner.load(entry)
            obs, _ = env.reset(seed=(seed + state["k"]) % (2**31))
            adapter.reset()
            ps = env.privileged_state()
            shaper.set_exit(entry.get("exit_xy"))
            shaper.reset(ps)
            ep.clear()
            ep.update(map=entry.get("label", f"{os.path.basename(entry['wad'])}:{entry['map']}"),
                      ticks=int(entry["ticks"]), length=0, ret=0.0, monsters=int(entry.get("monsters", 0)),
                      end_on_clear=bool(entry.get("end_on_clear", False)))
            return env, obs, ps

        def put(prefix, obs, ps):
            x = adapter.observe(obs)
            for k in A.INPUT_KEYS:
                S[f"{prefix}.{k}"][idx] = x[k]
            S[f"{prefix}.priv"][idx] = shaper.features(ps)

        env = None
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                env, obs, ps = start_episode()
                put("obs", obs, ps)
                pipe.send(("ok", {"torch": "torch" in sys.modules}))
            elif cmd == "step":
                a = np.asarray(data, np.int64).reshape(A.NUM_BUTTONS)
                adapter.record_action(a)
                obs, _, te, tr, info = env.step(a)
                ps = env.privileged_state()
                r, aux = shaper.step(ps, a)
                ep["length"] += 1
                ep["ret"] += r
                end = bool(te or tr)
                cleared = ep["end_on_clear"] and ep["monsters"] > 0 and ps["kills"] >= ep["monsters"]
                end = end or cleared            # combat maps: nothing left to score once every enemy is dead
                cutoff = (not end) and bool(state["cutoff"]) and ep["length"] >= state["cutoff"]
                S["reward"][idx], S["term"][idx], S["trunc"][idx], S["aux"][idx, 0] = r, end, cutoff, aux
                msg = {}
                if cutoff:
                    put("final", obs, ps)       # genuine next observation: bootstrap from it
                if end or cutoff:
                    msg["episode"] = {
                        "worker": idx, "map": ep["map"], "return": ep["ret"], "length": ep["length"],
                        "exit": bool(ps["exit_reached"]), "death": bool(ps["dead"]),
                        "tick_cap": bool(ps["timeout"]), "cutoff": cutoff, "tic": ps["tic"],
                        "health": ps["health"], "monsters": ep["monsters"], "cleared": bool(cleared),
                        "kill_frac": (ps["kills"] / ep["monsters"]) if ep["monsters"] else 0.0,
                        **shaper.episode_summary()}
                    env, obs, ps = start_episode()
                put("obs", obs, ps)
                pipe.send(("ok", msg))
            elif cmd == "configure":
                pending.update(data)            # applied at this worker's next episode start
                pipe.send(("ok", None))
            elif cmd == "close":
                pipe.send(("ok", None))
                break
    except Exception:
        pipe.send(("error", traceback.format_exc()))
    finally:
        if runner is not None:
            runner.close()
        if shm is not None:
            shm.close()


# ------------------------------------------------------------------ vector env
class DoomVecEnv:
    obs_spec = OBS_SPEC
    n_buttons = A.NUM_BUTTONS

    def __init__(self, schedule: list, n_envs: int, seed: int = 0, rewards: dict | None = None,
                 cutoff: int | None = None, fix_ipc: bool = True):
        self.n = n_envs
        self.shm = _Shm(n_envs, create=True)
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        per = self._per_worker(schedule)
        for i in range(n_envs):
            cfg = {"rewards": rewards or {}, "cutoff": cutoff, "fix_ipc": fix_ipc, **per[i]}
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(i, n_envs, self.shm.names, cfg, seed * 100_003 + 1000 * i, b),
                            daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)
        self.torch_in_workers = None

    def _gather(self):
        out, err = [], None
        for p in self.pipes:
            status, msg = p.recv()
            if status == "error":
                err = msg
            out.append(msg)
        if err:
            self.close()
            raise RuntimeError(f"env worker failed:\n{err}")
        return out

    def _obs(self, prefix):
        return {k: self.shm.arrays[f"{prefix}.{k}"].copy() for k in OBS_SPEC}

    def reset(self):
        for p in self.pipes:
            p.send(("reset", None))
        msgs = self._gather()
        self.torch_in_workers = any(m["torch"] for m in msgs)
        return self._obs("obs")

    def step(self, actions: np.ndarray):
        for p, a in zip(self.pipes, actions):
            p.send(("step", np.asarray(a, np.int8)))
        msgs = self._gather()
        S = self.shm.arrays
        info = {"final_obs": self._obs("final"), "aux_events": S["aux"].copy(),
                "episodes": [m["episode"] for m in msgs if "episode" in m]}
        return self._obs("obs"), S["reward"].copy(), S["term"].copy(), S["trunc"].copy(), info

    def _per_worker(self, schedule) -> list[dict]:
        """Shared list -> same list for all; list of N lists -> each worker its own list."""
        if schedule and isinstance(schedule[0], (list, tuple)):
            assert len(schedule) == self.n, "per-worker schedule needs one list per env"
            return [{"schedule": list(s), "own": True} for s in schedule]
        return [{"schedule": schedule, "own": False}] * self.n

    def configure(self, **kw):
        """schedule=[...] (shared) or [[...], ...] (per worker), rewards={...}, cutoff=int|None.
        Takes effect at each worker's next episode start; running episodes keep their settings."""
        per = self._per_worker(kw.pop("schedule")) if "schedule" in kw else [{}] * self.n
        for p, extra in zip(self.pipes, per):
            p.send(("configure", {**kw, **extra}))
        self._gather()

    def close(self):
        for p in self.pipes:
            try:
                p.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        if self.shm is not None:
            self.shm.close(unlink=True)
            self.shm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
