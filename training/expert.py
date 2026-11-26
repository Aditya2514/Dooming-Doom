"""Privileged exploration expert: the TEACHER for imitation learning (training only).

Nothing in submission/ imports this. The submitted agent is a neural network that imitates this
expert from the official observation alone (teacher-student distillation, "learning by cheating").
docs/RULES.md allows any training-time scaffolding; rule trees and path planners are only banned
in the final agent, and this file never runs at evaluation.

The expert may use ground truth (true position and heading, the generated map's floor plan, key
positions), but its *decisions* are kept imitable: it only goes for things the agent has had a
chance to observe.
  * "seen" map: grid cells swept by the 64 official rays (from the true pose), up to SEE_R units.
    The agent sees the same rays, so "go to the nearest place I haven't looked at yet" is a
    behaviour a recurrent student can learn.
  * keys / exit become targets only once their cell has been seen.
  * combat and door handling read the official observation directly (visible entities, rays).

Per decision, in priority order:
  0. dodge:   a visible projectile will pass within DODGE_MISS units in the next DODGE_T tics
              -> strafe away from its line (predicted from its relative velocity)
  1. fight:   an enemy is visible within ENGAGE units -> turn to it, fire when on target
              (holding fire: measured better than pistol tap-fire, see PISTOL_TAP_RANGE)
  2. door:    a closed door is right ahead on our way -> press USE once, then wait for it
  3. unstick: no movement while trying to move -> back off and turn
  4. go to:   a seen key still needed > a seen, reachable exit > a useful visible pickup >
              the nearest unseen reachable cell (frontier) > the exit (fallback)
Paths: BFS over a 16-unit grid of the floor plan (eroded by one cell so the 16-unit player radius
fits), with locked doors passable only once their key is held.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from submission.doomnet.actions import BUTTONS, FWD, USE, ATTACK
from submission.doomnet.actions import INDEX as AI

RES = 16.0
SEE_R = 320.0               # how far a ray sweep counts as "looked at"
ENGAGE = 1100.0             # fight enemies closer than this
DOOR_NEAR = 100.0           # a door this close ahead gets a USE press
USE_COOLDOWN = 14           # decisions between USE presses (a moving door reverses on USE)
REPLAN_EVERY = 4
TARGET_PATIENCE = 90        # decisions before giving up on one frontier target
LOOKAHEAD = 12              # path cells considered for the steering waypoint (12 x 16 = 192 units)
DODGE_MISS = 56.0          # projectile passing closer than this (units) is a threat
DODGE_T = 40.0             # ... within this many tics
ENEMY_TYPES = (1, 2, 3)
T_PROJECTILE = 4
PISTOL, PISTOL_TAP = 2, 5      # weapon slot; decisions between pistol shots (cycle ~19 tics)
# Tapping is OFF (range = inf): measured on mixed validation (80 matched episodes), tapping at every
# range gave 91% teacher survival, tapping beyond 512 units 93%, holding fire 99%. Rate of fire beats
# first-shot accuracy at Doom's fighting distances.
PISTOL_TAP_RANGE = float("inf")
T_HEALTH, T_ARMOR, T_AMMO, T_WEAPON, T_KEY = 5, 6, 7, 8, 9
# ZDoom lock number -> index of the key flag in vitals[8:14] (blue card, yellow card, red card,
# blue skull, yellow skull, red skull)
LOCK_TO_KEYFLAG = {1: 2, 2: 0, 3: 1, 4: 5, 5: 3, 6: 4}
RAY_ANG = np.arange(64) * (2 * math.pi / 64)
SEE_STEPS = np.arange(0.0, SEE_R + 1, 8.0)
NB8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class NavGrid:
    """Floor plan of a generated map on a 16-unit grid (from the generator's route metadata)."""

    def __init__(self, route: dict):
        rects = route["rects"]
        self.x0 = min(r[0] for r in rects) - 2 * RES
        self.y0 = min(r[1] for r in rects) - 2 * RES
        self.W = int(math.ceil((max(r[2] for r in rects) - self.x0) / RES)) + 2
        self.H = int(math.ceil((max(r[3] for r in rects) - self.y0) / RES)) + 2
        walk = np.zeros((self.W, self.H), bool)
        self.lock_of = np.zeros((self.W, self.H), np.int8)
        self.door = np.zeros((self.W, self.H), bool)
        pillars = []
        for x0, y0, x1, y1, fl, ce, door, lock in rects:
            sl = self._slice(x0, y0, x1, y1)
            if ce <= fl and not door:
                pillars.append(sl)
                continue
            walk[sl] = True
            if door:
                self.door[sl] = True
                if lock:
                    self.lock_of[sl] = lock
        for sl in pillars:
            walk[sl] = False
        self.walk = walk
        safe = walk.copy()                                   # erode: every 8-neighbour walkable
        for di, dj in NB8:
            safe &= np.roll(np.roll(walk, di, 0), dj, 1)
        self.safe = safe
        self.keys = [(float(k[0]), float(k[1]), int(k[2])) for k in route.get("keys", [])]
        ex = route.get("exit_xy")
        self.exit_xy = None if ex is None else (float(ex[0]), float(ex[1]))
        self._graphs: dict = {}

    def _slice(self, x0, y0, x1, y1):
        i0 = int(math.floor((x0 - self.x0) / RES))
        i1 = int(math.ceil((x1 - self.x0) / RES))
        j0 = int(math.floor((y0 - self.y0) / RES))
        j1 = int(math.ceil((y1 - self.y0) / RES))
        return slice(i0, i1), slice(j0, j1)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.x0) // RES), int((y - self.y0) // RES)

    def centre(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + (i + 0.5) * RES, self.y0 + (j + 0.5) * RES

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.W and 0 <= j < self.H

    def graph(self, open_locks: frozenset):
        g = self._graphs.get(open_locks)
        if g is not None:
            return g
        lock_ok = (self.lock_of == 0)
        for lk in open_locks:
            lock_ok |= (self.lock_of == lk)
        P = self.safe & lock_ok
        idx = np.full(P.shape, -1, np.int64)
        coords = np.argwhere(P)
        idx[coords[:, 0], coords[:, 1]] = np.arange(len(coords))
        rows, cols = [], []
        for di, dj in NB8:
            ok = P & self._shift(P, di, dj)                   # neighbour (i+di, j+dj) passable too
            if di and dj:                                     # no corner cutting
                ok &= self._shift(P, di, 0) & self._shift(P, 0, dj)
            src = np.argwhere(ok)
            rows.append(idx[src[:, 0], src[:, 1]])
            cols.append(idx[src[:, 0] + di, src[:, 1] + dj])
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        G = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(len(coords), len(coords)))
        g = (P, idx, coords, G)
        self._graphs[open_locks] = g
        return g

    @staticmethod
    def _shift(a: np.ndarray, di: int, dj: int) -> np.ndarray:
        """out[i, j] = a[i + di, j + dj] (False outside)."""
        out = np.zeros_like(a)
        W, H = a.shape
        out[max(0, -di):W - max(0, di), max(0, -dj):H - max(0, dj)] = a[max(0, di):W + min(0, di),
                                                                        max(0, dj):H + min(0, dj)]
        return out

    def nearest_node(self, P: np.ndarray, i: int, j: int, r_max: int = 4):
        if self.inside(i, j) and P[i, j]:
            return i, j
        best, bd = None, 1e9
        for di in range(-r_max, r_max + 1):
            for dj in range(-r_max, r_max + 1):
                a, b = i + di, j + dj
                if self.inside(a, b) and P[a, b] and di * di + dj * dj < bd:
                    best, bd = (a, b), di * di + dj * dj
        return best

    def line_clear(self, P: np.ndarray, x0, y0, x1, y1) -> bool:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 8.0))
        for t in np.linspace(0.0, 1.0, n):
            i, j = self.cell(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if not (self.inside(i, j) and self.walk[i, j]):
                return False
            if self.lock_of[i, j] and not P[i, j] and self.safe[i, j]:
                return False
        return True


class Expert:
    """Call reset() at an episode start, then act(obs, ps) once per decision (before stepping)."""

    def __init__(self, route: dict):
        self.grid = NavGrid(route)

    def reset(self) -> None:
        g = self.grid
        self.seen = np.zeros((g.W, g.H), bool)
        self.t = 0
        self.last_use = -100
        self.last_fire = -100
        self.target = None              # (kind, (i, j))
        self.target_t = 0
        self.path: list = []
        self.path_t = -100
        self.trail: deque = deque(maxlen=6)
        self.exec_hist: deque = deque(maxlen=6)   # macro-actions actually executed (expert's or student's)
        self.rec_t0: int | None = None            # start of a back-off-and-turn recovery
        self.rec_side = AI["TURN_L"]
        self._pending: int | None = None
        self.banned: set = set()         # frontier cells given up on
        self.mode = "start"

    # ------------------------------------------------------------- perception
    def _mark_seen(self, obs, x, y, heading):
        d = np.minimum(np.asarray(obs["rays"]["distance"], np.float64) * 1024.0, SEE_R)
        ang = heading + RAY_ANG
        m = SEE_STEPS[None, :] <= d[:, None]
        xs = (x + np.cos(ang)[:, None] * SEE_STEPS[None, :])[m]
        ys = (y + np.sin(ang)[:, None] * SEE_STEPS[None, :])[m]
        g = self.grid
        ii = ((xs - g.x0) // RES).astype(np.int64)
        jj = ((ys - g.y0) // RES).astype(np.int64)
        ok = (ii >= 1) & (ii < g.W - 1) & (jj >= 1) & (jj < g.H - 1)
        ii, jj = ii[ok], jj[ok]
        for di in (-1, 0, 1):                       # dilate: fill the gaps between rays
            for dj in (-1, 0, 1):
                self.seen[ii + di, jj + dj] = True

    @staticmethod
    def _entities(obs):
        e = obs["visible_entities"]
        m = np.asarray(e["mask"]).astype(bool)
        t = np.asarray(e["entity_type"])[m]
        f = np.asarray(e["features"], np.float64)[m]
        return t, f[:, 0] * 1024.0, np.arctan2(f[:, 1], f[:, 2])

    # ---------------------------------------------------------------- policy
    def executed(self, a: int) -> None:
        """Report the macro-action actually executed after the last act() (DAgger: often the
        student's). The expert's memory (USE cooldown, stuck detection) follows what really
        happened, not its own suggestions; otherwise its labels on student-driven states are wrong
        (e.g. a student turning in place looked 'stuck' and got labelled 'back up', a loop)."""
        self._pending = None
        b = BUTTONS[int(a)]
        self.exec_hist.append(int(a))
        if b[USE]:
            self.last_use = self.t
        if b[ATTACK]:
            self.last_fire = self.t

    def act(self, obs: dict, ps: dict) -> int:
        """Label for the current state. If executed() was not called since the last act(), the
        expert's own suggestion is assumed to have been executed (the expert playing alone)."""
        if self._pending is not None:
            self.executed(self._pending)
        a = self._decide(obs, ps)
        self._pending = a
        return a

    def _decide(self, obs: dict, ps: dict) -> int:
        self.t += 1
        x, y = float(ps["x"]), float(ps["y"])
        heading = math.radians(float(ps["angle"]))
        self._mark_seen(obs, x, y, heading)
        vit = np.asarray(obs["vitals"], np.float64)
        self._health = float(vit[0])
        held = frozenset(lk for lk, fi in LOCK_TO_KEYFLAG.items() if vit[8 + fi] > 0.5)
        types, dists, angs = self._entities(obs)
        moved = math.dist(self.trail[0], (x, y)) if len(self.trail) == self.trail.maxlen else 1e9
        self.trail.append((x, y))

        # 0. dodge, 1. fight
        a = self._dodge(obs)
        if a is not None:
            self.mode = "dodge"
            self.trail.clear()
            return a
        a = self._fight(types, dists, angs, vit)
        if a is not None:
            self.mode = "fight"
            self.trail.clear()
            return a

        # 3. unstick (a running recovery sequence first)
        if self.rec_t0 is not None:
            k = self.t - self.rec_t0
            if k < 4:
                self.mode = "unstick"
                return (AI["BACK"], AI["BACK"], self.rec_side, self.rec_side)[k]
            self.rec_t0 = None

        # 4. navigation target and steering
        wp = self._waypoint(x, y, held, types, dists, angs, heading)
        if wp is None:
            self.mode = "roam"
            return self._roam(obs)
        err = wrap(math.atan2(wp[1] - y, wp[0] - x) - heading)

        # 2. a closed door right ahead on our way
        if abs(err) < math.radians(35) and self._door_ahead(obs, bool(held)):
            self.mode = "door"
            self.trail.clear()
            if self.t - self.last_use >= USE_COOLDOWN:
                return AI["FWD_USE"]
            return AI["FWD"]

        tried_forward = sum(int(BUTTONS[e][FWD]) for e in self.exec_hist) >= 4
        if moved < 10.0 and tried_forward and self.mode in ("go", "frontier"):
            self.trail.clear()
            self.rec_side = AI["TURN_L"] if err > 0 else AI["TURN_R"]
            self.rec_t0 = self.t
            self.path_t = -100                                  # replan afterwards
            self.mode = "unstick"
            return AI["BACK"]

        self.mode = "go" if self.target and self.target[0] != "frontier" else "frontier"
        if abs(err) > math.radians(50):
            return AI["TURN_L"] if err > 0 else AI["TURN_R"]
        if abs(err) > math.radians(9):
            return AI["FWD_L"] if err > 0 else AI["FWD_R"]
        return AI["FWD"]

    # ---------------------------------------------------------------- pieces
    @staticmethod
    def _dodge(obs):
        e = obs["visible_entities"]
        m = np.asarray(e["mask"]).astype(bool) & (np.asarray(e["entity_type"]) == T_PROJECTILE)
        if not m.any():
            return None
        f = np.asarray(e["features"], np.float64)[m]
        best = None
        for d, sn, cs, vf, vr in f:
            d *= 1024.0
            pf, pl = d * cs, d * sn                       # projectile position: forward, left
            vl = -vr                                      # relative velocity: forward, left
            vv = vf * vf + vl * vl
            if vv < 1.0:
                continue
            t = -(pf * vf + pl * vl) / vv                 # time of closest approach (tics)
            if t <= 0 or t > DODGE_T:
                continue
            cl = pl + vl * t
            miss = math.hypot(pf + vf * t, cl)
            if miss < DODGE_MISS and (best is None or t < best[0]):
                best = (t, cl)
        if best is None:
            return None
        rays = np.asarray(obs["rays"]["distance"]) * 1024.0
        room_l, room_r = rays[14:19].min(), rays[46:51].min()
        go_left = best[1] < -8 or (abs(best[1]) <= 8 and room_l >= room_r)   # it passes right of us -> left
        if go_left and room_l < 40:
            go_left = False
        elif not go_left and room_r < 40:
            go_left = True
        return AI["STRAFE_L"] if go_left else AI["STRAFE_R"]

    def _fight(self, types, dists, angs, vit):
        en = np.isin(types, ENEMY_TYPES) & (dists < ENGAGE)
        if not en.any():
            return None
        weapon, ammo = vit[2], vit[3]
        fists = weapon <= 1 or ammo <= 0
        k = np.flatnonzero(en)[np.argmin(dists[en])]
        d, a = dists[k], angs[k]
        if fists and d > 120:
            return None                      # can't hurt it from here: keep exploring
        tol = max(math.radians(4.0), math.atan2(18.0, max(d, 1.0)))
        # Pistol tap-fire (held fire re-fires with spread): implemented but disabled, see PISTOL_TAP_RANGE.
        can_fire = not (weapon == PISTOL and d > PISTOL_TAP_RANGE and self.t - self.last_fire < PISTOL_TAP)
        if abs(a) <= tol:
            if not can_fire:
                return AI["NOOP"]
            if types[k] == 1 and d < 200 and not fists:
                return AI["BACK_ATK"]        # demon closing in: back off while firing
            return AI["ATK"]
        if abs(a) < math.radians(12) and can_fire:
            return AI["ATK_L"] if a > 0 else AI["ATK_R"]
        return AI["TURN_L"] if a > 0 else AI["TURN_R"]

    @staticmethod
    def _door_ahead(obs, have_key: bool) -> bool:
        hit = np.asarray(obs["rays"]["hit_type"])
        dist = np.asarray(obs["rays"]["distance"]) * 1024.0
        for r in (62, 63, 0, 1, 2):
            if dist[r] < DOOR_NEAR and (hit[r] == 2 or (hit[r] == 3 and have_key)):
                return True
        return False

    def _roam(self, obs) -> int:
        d = np.asarray(obs["rays"]["distance"])
        if d[0] > 0.15:
            return AI["FWD"]
        return AI["TURN_L"] if d[1:17].max() >= d[48:64].max() else AI["TURN_R"]

    def _choose_target(self, start, P, idx, coords, G, held, types, dists, angs, heading, x, y):
        g = self.grid
        order, pred = breadth_first_order(G, idx[start], directed=True, return_predecessors=True)
        reach = np.zeros(len(coords), bool)
        reach[order] = True

        def reachable_cell(cx, cy):
            c = g.nearest_node(P, *g.cell(cx, cy), r_max=3)
            if c is None or not reach[idx[c]]:
                return None
            return c

        # needed keys we have seen
        for kx, ky, lk in g.keys:
            if lk in held:
                continue
            ki, kj = g.cell(kx, ky)
            if g.inside(ki, kj) and self.seen[max(0, ki - 1):ki + 2, max(0, kj - 1):kj + 2].any():
                c = reachable_cell(kx, ky)
                if c is not None:
                    return ("key", c), order, pred
        # the exit, once seen
        if g.exit_xy is not None:
            ei, ej = g.cell(*g.exit_xy)
            if g.inside(ei, ej) and self.seen[ei, ej]:
                c = reachable_cell(*g.exit_xy)
                if c is not None:
                    return ("exit", c), order, pred
        # a visible pickup worth a detour
        for t, dd, aa in zip(types, dists, angs):
            want = (t == T_WEAPON and dd < 600) or (t == T_HEALTH and dd < 450 and self._health < 70.0) \
                or (t == T_ARMOR and dd < 300)
            if want:
                c = reachable_cell(x + dd * math.cos(heading + aa), y + dd * math.sin(heading + aa))
                if c is not None:
                    return ("item", c), order, pred
        # nearest unseen reachable cell
        oc = coords[order]
        unseen = ~self.seen[oc[:, 0], oc[:, 1]]
        if self.banned:
            unseen &= np.array([(int(a), int(b)) not in self.banned for a, b in oc])
        hits = np.flatnonzero(unseen)
        if len(hits):
            c = oc[hits[0]]
            return ("frontier", (int(c[0]), int(c[1]))), order, pred
        if g.exit_xy is not None:
            c = reachable_cell(*g.exit_xy)
            if c is not None:
                return ("exit", c), order, pred
        return None, order, pred

    _health = 100.0

    def _waypoint(self, x, y, held, types, dists, angs, heading):
        g = self.grid
        P, idx, coords, G = g.graph(held)
        start = g.nearest_node(P, *g.cell(x, y))
        if start is None:
            return None
        need_plan = (self.t - self.path_t >= REPLAN_EVERY or not self.path or self.target is None)
        if self.target is not None:
            kind, (ti, tj) = self.target
            if kind == "frontier" and self.seen[ti, tj]:
                need_plan = True
                self.target = None
            elif self.t - self.target_t > TARGET_PATIENCE:
                if kind == "frontier":
                    self.banned.add((ti, tj))
                need_plan = True
                self.target = None
        if need_plan:
            tgt, order, pred = self._choose_target(start, P, idx, coords, G, held, types, dists, angs, heading, x, y)
            if tgt is None:
                self.target, self.path = None, []
                return None
            if self.target is None or tgt[0] != self.target[0] or tgt[1] != self.target[1]:
                if not (self.target and self.target[0] == "frontier" and tgt[0] == "frontier"
                        and not self.seen[self.target[1]] and self._reach_ok(self.target[1], idx, pred, start)):
                    self.target = tgt
                    self.target_t = self.t
            # path start -> target from the BFS tree
            path, n = [], idx[self.target[1]]
            s = idx[start]
            if n < 0:
                self.target = None
                return None
            while n != s and n >= 0:
                path.append(n)
                n = pred[n]
            if n != s:
                self.target = None
                return None
            path.reverse()
            self.path = [tuple(coords[k]) for k in path]
            self.path_t = self.t
        # drop path cells already passed
        ci = g.cell(x, y)
        best = 0
        for k, c in enumerate(self.path[:LOOKAHEAD]):
            if max(abs(c[0] - ci[0]), abs(c[1] - ci[1])) <= 1:
                best = k + 1
        self.path = self.path[best:]
        if not self.path:
            tc = g.centre(*self.target[1]) if self.target else None
            if tc is not None and math.dist(tc, (x, y)) > 24:
                return tc
            self.target = None
            return None
        wp = g.centre(*self.path[0])
        for c in self.path[1:LOOKAHEAD]:
            cx, cy = g.centre(*c)
            if g.line_clear(P, x, y, cx, cy):
                wp = (cx, cy)
            else:
                break
        return wp

    @staticmethod
    def _reach_ok(cell, idx, pred, start) -> bool:
        n = idx[cell]
        return n >= 0 and (pred[n] >= 0 or n == idx[start])
