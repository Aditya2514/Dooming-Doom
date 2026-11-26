"""Episode metrics mirroring the published scoring categories (local evaluation only).

Reads env.privileged_state() (ground truth), which the agent never sees:
    keys          key flags gained
    doors         ordinary door sectors opened (clearance first crosses 56 units; doors already
                  open at reset don't count)
    locked_doors  same for door sectors that need a key
    cells         distinct 128-unit cells of the player's position (our stand-in for the
                  scorer's private "unique areas visited")
"""
from __future__ import annotations

import math

import numpy as np

PLAYER_HEIGHT = 56.0
CELL = 128.0


class EpisodeMetrics:
    def reset(self, ps: dict) -> None:
        self.prev_keys = np.asarray(ps["keys"]).copy()
        self.cells = {self._cell(ps)}
        lock = ps["door_lock"]
        self.door_mask = (lock >= 0) if lock is not None else np.zeros(0, bool)
        self.locked_mask = (lock > 0) if lock is not None else np.zeros(0, bool)
        clear = self._clearance(ps)
        self.initially_open = clear >= PLAYER_HEIGHT
        self.opened = np.zeros_like(self.door_mask)
        self.counts = {"keys": 0, "doors": 0, "locked_doors": 0}

    @staticmethod
    def _cell(ps) -> tuple[int, int]:
        return int(math.floor(ps["x"] / CELL)), int(math.floor(ps["y"] / CELL))

    @staticmethod
    def _clearance(ps) -> np.ndarray:
        if ps["sector_ceiling"] is None:
            return np.zeros(0)
        return ps["sector_ceiling"] - ps["sector_floor"]

    def step(self, ps: dict) -> None:
        keys = np.asarray(ps["keys"])
        self.counts["keys"] += int(np.sum((keys > 0.5) & (self.prev_keys < 0.5)))
        self.prev_keys = keys.copy()
        self.cells.add(self._cell(ps))
        clear = self._clearance(ps)
        if clear.size == self.door_mask.size and clear.size:
            newly = self.door_mask & (clear >= PLAYER_HEIGHT) & ~self.opened & ~self.initially_open
            if newly.any():
                self.opened |= newly
                n_locked = int((newly & self.locked_mask).sum())
                self.counts["locked_doors"] += n_locked
                self.counts["doors"] += int(newly.sum()) - n_locked
