"""Rewards for PPO fine-tuning (training only; the agent never sees them).

Extrinsic terms mirror the published scoring categories and read env.privileged_state():
    exit         +10, plus +2 x health fraction and +2 x time-left fraction
    key          +3 per key picked up
    door         +1 per ordinary door opened, +3 per locked door opened (training/metrics.py)
    kill         +1 per kill
    explore      +0.05 per new 128-unit cell of the TRUE position (the scorer's "unique areas")
    damage       -0.02 per point of damage taken
    death        -5 x (1 - time fraction)       (dying early costs more, like the scorer)
    time         -0.002 per decision
Intrinsic curiosity (count-based exploration bonus, Bellemare et al. 2016 / Strehl & Littman 2008):
    curiosity    +0.04 / sqrt(n) on ENTERING a cell for the n-th time, cells of the agent's OWN
                 dead-reckoned spatial map (submission/doomnet/obs_adapter.py) -- computed from
                 what the agent itself knows, so it is the bonus a deployed agent could compute.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from training.metrics import EpisodeMetrics


@dataclass
class RewardConfig:
    exit: float = 10.0
    exit_health: float = 2.0
    exit_time: float = 2.0
    key: float = 3.0
    door: float = 1.0
    locked_door: float = 3.0
    kill: float = 1.0
    explore: float = 0.05
    damage: float = -0.02
    death: float = -5.0
    time: float = -0.002
    curiosity: float = 0.04


TERMS = ("exit", "key", "door", "kill", "explore", "damage", "death", "time", "curiosity")


class Rewarder:
    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()

    def reset(self, ps: dict, pose: np.ndarray) -> None:
        self.met = EpisodeMetrics()
        self.met.reset(ps)
        self.prev = ps
        self.counts: dict = defaultdict(int)
        self.cell = self._own_cell(pose)
        self.counts[self.cell] += 1
        self.terms = dict.fromkeys(TERMS, 0.0)

    @staticmethod
    def _own_cell(pose) -> tuple[int, int]:
        return int(math.floor(pose[0] / 64.0)), int(math.floor(pose[1] / 64.0))

    def step(self, ps: dict, pose: np.ndarray | None) -> float:
        """Reward for the decision that led from the previous state to `ps`. pose: the adapter's
        dead-reckoned pose for the new observation (None on the last step of an episode)."""
        c, p = self.cfg, self.prev
        before = dict(self.met.counts)
        n_cells = len(self.met.cells)
        self.met.step(ps)
        r = dict.fromkeys(TERMS, 0.0)
        frac = min(1.0, ps["tic"] / max(1, ps["episode_ticks"]))
        if ps["exit_reached"]:
            r["exit"] = c.exit + c.exit_health * min(max(ps["health"], 0.0), 100.0) / 100.0 + c.exit_time * (1 - frac)
        if ps["dead"]:
            r["death"] = c.death * (1.0 - frac)
        r["key"] = c.key * (self.met.counts["keys"] - before["keys"])
        r["door"] = (c.door * (self.met.counts["doors"] - before["doors"])
                     + c.locked_door * (self.met.counts["locked_doors"] - before["locked_doors"]))
        r["kill"] = c.kill * max(ps["kills"] - p["kills"], 0)
        r["explore"] = c.explore * (len(self.met.cells) - n_cells)
        r["damage"] = c.damage * max(ps["damage_taken"] - p["damage_taken"], 0.0)
        r["time"] = c.time
        if pose is not None:
            cell = self._own_cell(pose)
            if cell != self.cell:                        # entering a cell (standing still earns nothing)
                self.counts[cell] += 1
                r["curiosity"] = c.curiosity / math.sqrt(self.counts[cell])
                self.cell = cell
        for k, v in r.items():
            self.terms[k] += v
        self.prev = ps
        return float(sum(r.values()))
