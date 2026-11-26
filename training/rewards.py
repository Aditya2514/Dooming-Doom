"""Training-only reward shaping and privileged critic inputs (Spec v2.1, Rewards / Privileged critic).

Everything here reads env.privileged_state() (ground truth). None of it is ever an actor
input: the actor only sees the shared adapter's output. Runs inside each env worker.

Reward terms (starting values, [guessing] on magnitudes):
    exit         +10, plus 2 x health fraction + 2 x time-left fraction
    key          +2 per new key flag
    door         +0.5 once per ordinary door sector per episode, on the step its clearance
                 (ceiling - floor) first crosses from < 56 to >= 56; 0 if already open at reset
    locked_door  +2.0, same rule for a sector with door_lock > 0
    explore      +0.02 per first visit to a 128-unit cell of the engine position, cap +3 / episode
    kill         +0.5 per kill (every stage)
    damage       -0.01 per point of damage taken (engine counter)
    death        -5 scaled by how early: -5 x (1 - tic / episode_ticks)
    time         -0.001 per decision
Door rewards stay OFF (doors_enabled=False) until the deterministic door test passes; door
openings are still tracked and logged.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

PLAYER_HEIGHT = 56.0
USE_BUTTON = 7
USE_WINDOW = 40          # decisions: 'opening without a recent USE press' diagnostic
PRIV_DIM = 16


@dataclass
class RewardConfig:
    exit: float = 10.0
    exit_health: float = 2.0
    exit_time: float = 2.0
    key: float = 2.0
    door: float = 0.5
    locked_door: float = 2.0
    doors_enabled: bool = False
    explore: float = 0.02
    explore_cap: float = 3.0
    cell: float = 128.0
    kill: float = 0.5
    damage: float = -0.01
    death: float = -5.0
    time: float = -0.001

    def to_dict(self):
        return asdict(self)


TERMS = ("exit", "key", "door", "locked_door", "explore", "kill", "damage", "death", "time")


class RewardShaper:
    """Per-episode reward state. reset(ps) at episode start; step(ps, action) after each decision."""

    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()
        self.exit_xy = None

    def set_exit(self, exit_xy) -> None:
        """Exit position from the map's metadata (critic input only); None if unknown."""
        self.exit_xy = None if exit_xy is None else (float(exit_xy[0]), float(exit_xy[1]))

    # ------------------------------------------------------------------ episode
    def reset(self, ps: dict) -> None:
        self.prev = ps
        self.x0, self.y0, self.z0 = ps["x"], ps["y"], ps["z"]
        self.cells = {self._cell(ps)}
        self.explore_total = 0.0
        self.t = 0
        self.last_use = -10**9
        self.prev_use = False
        lock = ps["door_lock"]
        self.door_mask = (lock >= 0) if lock is not None else np.zeros(0, bool)
        self.locked_mask = (lock > 0) if lock is not None else np.zeros(0, bool)
        self.n_locked = int(self.locked_mask.sum())
        clear = self._clearance(ps)
        self.initially_open = clear >= PLAYER_HEIGHT
        self.opened = np.zeros_like(self.door_mask)
        self.terms = {k: 0.0 for k in TERMS}
        self.counts = {"doors": 0, "locked_doors": 0, "doors_without_use": 0, "keys": 0, "kills": 0}

    def _cell(self, ps) -> tuple[int, int]:
        return (int(math.floor(ps["x"] / self.cfg.cell)), int(math.floor(ps["y"] / self.cfg.cell)))

    @staticmethod
    def _clearance(ps) -> np.ndarray:
        if ps["sector_ceiling"] is None:
            return np.zeros(0)
        return ps["sector_ceiling"] - ps["sector_floor"]

    # --------------------------------------------------------------------- step
    def step(self, ps: dict, action) -> tuple[float, bool]:
        """Reward for the decision just taken, and the aux event (damage or death during it)."""
        c, p = self.cfg, self.prev
        self.t += 1
        use = bool(np.asarray(action).reshape(-1)[USE_BUTTON])
        if use and not self.prev_use:                # USE fires on the button-down edge
            self.last_use = self.t
        self.prev_use = use
        r = dict.fromkeys(TERMS, 0.0)
        frac = min(1.0, ps["tic"] / max(1, ps["episode_ticks"]))

        if ps["exit_reached"]:
            r["exit"] = c.exit + c.exit_health * min(max(ps["health"], 0.0), 100.0) / 100.0 + c.exit_time * (1.0 - frac)
        if ps["dead"]:
            r["death"] = c.death * (1.0 - frac)

        new_keys = int(np.sum((ps["keys"] > 0.5) & (p["keys"] < 0.5)))
        r["key"] = c.key * new_keys
        kills = ps["kills"] - p["kills"]
        r["kill"] = c.kill * max(kills, 0)
        dmg = max(ps["damage_taken"] - p["damage_taken"], 0.0)
        r["damage"] = c.damage * dmg
        r["time"] = c.time

        cell = self._cell(ps)
        if cell not in self.cells:
            self.cells.add(cell)
            gain = min(c.explore, c.explore_cap - self.explore_total)
            if gain > 0:
                r["explore"] = gain
                self.explore_total += gain

        clear = self._clearance(ps)
        if clear.size == self.door_mask.size and clear.size:
            newly = self.door_mask & (clear >= PLAYER_HEIGHT) & ~self.opened & ~self.initially_open
            if newly.any():
                self.opened |= newly
                n_locked = int((newly & self.locked_mask).sum())
                n_plain = int(newly.sum()) - n_locked
                self.counts["doors"] += n_plain
                self.counts["locked_doors"] += n_locked
                if self.t - self.last_use > USE_WINDOW:
                    self.counts["doors_without_use"] += n_plain + n_locked
                if c.doors_enabled:
                    r["door"] = c.door * n_plain
                    r["locked_door"] = c.locked_door * n_locked

        self.counts["keys"] += new_keys
        self.counts["kills"] += max(kills, 0)
        for k, v in r.items():
            self.terms[k] += v
        self.prev = ps
        aux = bool(dmg > 0 or ps["dead"])
        return float(sum(r.values())), aux

    # ----------------------------------------------------------- critic inputs
    def features(self, ps: dict) -> np.ndarray:
        """16 privileged critic inputs (training only). Slot 15: straight-line distance to the
        exit / 4096 when the map's metadata gives the exit position, else 0."""
        f = np.zeros(PRIV_DIM, np.float32)
        frac = min(1.0, ps["tic"] / max(1, ps["episode_ticks"]))
        a = math.radians(ps["angle"])
        f[0] = 1.0 - frac                                        # ticks remaining on this map's cap
        f[1] = ps["health"] / 100.0
        f[2], f[3] = math.sin(a), math.cos(a)
        f[4] = (ps["x"] - self.x0) / 2048.0
        f[5] = (ps["y"] - self.y0) / 2048.0
        f[6] = (ps["z"] - self.z0) / 256.0
        f[7] = ps["kills"] / 10.0
        f[8] = ps["items"] / 10.0
        f[9] = ps["damage_taken"] / 100.0
        f[10] = float(np.sum(ps["keys"])) / 3.0
        f[11] = self.counts["doors"] / 5.0
        f[12] = self.counts["locked_doors"] / 3.0
        f[13] = (1.0 - float((self.opened & self.locked_mask).sum()) / self.n_locked) if self.n_locked else 0.0
        f[14] = min(len(self.cells), 200) / 50.0
        f[15] = 0.0 if self.exit_xy is None else math.hypot(ps["x"] - self.exit_xy[0], ps["y"] - self.exit_xy[1]) / 4096.0
        return f

    def episode_summary(self) -> dict:
        return {"explore_cells": len(self.cells), **self.counts,
                **{f"r_{k}": round(v, 4) for k, v in self.terms.items()}}
