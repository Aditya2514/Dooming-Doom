"""Route guidance for reward shaping (training only; the agent never sees any of this).

Potential-based shaping (Ng, Harada & Russell 1999): F = gamma * Phi(s') - Phi(s), which provably
leaves the optimal policy unchanged. Phi(s) = -w * d(s) / 1000, where d is the WALKING distance
(breadth-first flood over the generated map's floor, 32-unit cells) to the current target:
    1. the nearest key not yet held (doors whose key isn't held count as walls),
    2. once every key is held: the exit (doors then open).
Locked doors are walls until their key is held, so the route leads key -> door -> exit in turn.
Idea credit: the route distance field of the swedishembedded/brain DOOM sample (Apache-2.0); no code
taken. Used only by the training reward; flood fill never runs in the submitted agent.
"""
from __future__ import annotations

from collections import deque

import numpy as np

RES = 32.0
# ZDoom lock number -> index of the key flag in the official vitals (8..13 -> 0..5):
# vitals keys: blue card, yellow card, red card, blue skull, yellow skull, red skull
LOCK_TO_KEYFLAG = {1: 2, 2: 0, 3: 1, 4: 5, 5: 3, 6: 4}


class RouteGuide:
    def __init__(self, route: dict):
        rects = route["rects"]
        self.x0 = min(r[0] for r in rects)
        self.y0 = min(r[1] for r in rects)
        W = int((max(r[2] for r in rects) - self.x0) // RES) + 2
        H = int((max(r[3] for r in rects) - self.y0) // RES) + 2
        self.walk = np.zeros((W, H), bool)
        self.lock_of = np.zeros((W, H), np.int8)            # lock number of a locked-door cell, else 0
        for x0, y0, x1, y1, fl, ce, door, lock in rects:
            if ce <= fl and not door:
                continue                                     # solid pillar
            i0, i1 = int((x0 - self.x0) // RES), int((x1 - self.x0 - 1) // RES) + 1
            j0, j1 = int((y0 - self.y0) // RES), int((y1 - self.y0 - 1) // RES) + 1
            self.walk[i0:i1, j0:j1] = True
            if door and lock:
                self.lock_of[i0:i1, j0:j1] = lock
        self.keys = [(float(k[0]), float(k[1]), int(k[2])) for k in route.get("keys", [])]
        self.exit_xy = route.get("exit_xy")
        self._fields: dict = {}

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.x0) // RES), int((y - self.y0) // RES)

    def _field(self, target_xy, open_locks: frozenset) -> np.ndarray:
        key = (round(target_xy[0]), round(target_xy[1]), open_locks)
        f = self._fields.get(key)
        if f is not None:
            return f
        passable = self.walk & ((self.lock_of == 0) | np.isin(self.lock_of, list(open_locks) or [-1]))
        f = np.full(self.walk.shape, np.inf, np.float32)
        s = self.cell(*target_xy)
        if 0 <= s[0] < f.shape[0] and 0 <= s[1] < f.shape[1]:
            f[s] = 0.0
            q = deque([s])
            W, H = f.shape
            while q:
                i, j = q.popleft()
                d = f[i, j] + RES
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= a < W and 0 <= b < H and passable[a, b] and f[a, b] > d:
                        f[a, b] = d
                        q.append((a, b))
        self._fields[key] = f
        return f

    def distance(self, x: float, y: float, keys_held: np.ndarray) -> float | None:
        """Walking distance to the current target, or None if unknown (off-grid / unreachable)."""
        held = {lk for lk, idx in LOCK_TO_KEYFLAG.items() if keys_held[idx] > 0.5}
        missing = [k for k in self.keys if k[2] not in held]
        i, j = self.cell(x, y)
        best = None
        if missing:
            for kx, ky, _lk in missing:
                f = self._field((kx, ky), frozenset(held))
                if 0 <= i < f.shape[0] and 0 <= j < f.shape[1] and np.isfinite(f[i, j]):
                    best = f[i, j] if best is None else min(best, f[i, j])
            return None if best is None else float(best)      # a key is missing: never point past its door
        if self.exit_xy is None:
            return None
        f = self._field(tuple(self.exit_xy), frozenset(held) | frozenset(LOCK_TO_KEYFLAG))  # all doors open
        if 0 <= i < f.shape[0] and 0 <= j < f.shape[1] and np.isfinite(f[i, j]):
            return float(f[i, j])
        return None
