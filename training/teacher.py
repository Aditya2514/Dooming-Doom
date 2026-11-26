"""Reaction labels: a training-time teacher for the most basic button reactions.

TRAINING ONLY. Nothing in submission/ imports this; the submitted agent is the network alone
(docs/RULES.md section 1: "any training-time scaffolding" is allowed; rule trees and aimbots in
the final agent are not). The labels below are added to PPO as a supervised loss on a few buttons
(PPOConfig.bc_coef), so the network learns the reaction from its own inputs.

Why (measured 27 Nov on the submitted agent, 10,344 validation decisions): P(FIRE) was 0.42
whether or not an enemy was straight ahead, the FIRE logit barely moved (std 0.18), and being
blocked did not change the buttons. PPO's reward signal never linked these buttons to the
situation; a dense label does.

Per decision, from the raw observation (and, for "blocked", the true position):
  enemy visible (entity types 1-3, nearest one is the target), angle a (CCW = left positive):
      FIRE   = 1 if |a| <= tol (on target; fists: only within 80 units), else 0
      TURN_L = 1 if a > tol, TURN_R = 1 if a < -tol, both 0 when on target
  no enemy visible:
      FIRE   = 0                                   (nothing to hit: don't waste ammo)
      blocked (FORWARD pressed and moved < 2 units for the last 2 decisions):
          turn towards the more open side (longest ray within 90 deg), TURN_L/TURN_R labelled
  Every other button is unlabelled (mask 0): PPO alone decides it.
tol = max(2 deg, atan(20 / distance)): a monster is ~20 units in radius.
"""
from __future__ import annotations

import math

import numpy as np

FWD, BACK, STR_L, STR_R, TURN_L, TURN_R, FIRE, USE, WPN = range(9)
ENEMY_TYPES = (1, 2, 3)
MONSTER_RADIUS = 20.0
MIN_TOL = math.radians(2.0)
FIST_RANGE = 80.0
BLOCK_MOVE = 2.0
BLOCK_STREAK = 2
RAY_STEP = 2 * math.pi / 64


class ReactionTeacher:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_xy = None
        self.streak = 0

    def labels(self, obs: dict, ps: dict, prev_action) -> tuple[np.ndarray, np.ndarray, int]:
        """-> (labels (9,), mask (9,), situation) with situation 1 = enemy on target,
        2 = enemy visible but off target, 3 = blocked, 0 = none."""
        lab = np.zeros(9, np.float32)
        msk = np.zeros(9, np.float32)
        xy = (float(ps["x"]), float(ps["y"]))
        if self.prev_xy is not None and prev_action is not None and prev_action[FWD]:
            moved = math.hypot(xy[0] - self.prev_xy[0], xy[1] - self.prev_xy[1])
            self.streak = self.streak + 1 if moved < BLOCK_MOVE else 0
        else:
            self.streak = 0
        self.prev_xy = xy

        ents = obs["visible_entities"]
        m = np.asarray(ents["mask"]).astype(bool) & np.isin(np.asarray(ents["entity_type"]), ENEMY_TYPES)
        msk[FIRE] = 1.0
        if m.any():
            f = np.asarray(ents["features"], np.float64)[m]
            i = int(np.argmin(f[:, 0]))
            dist = f[i, 0] * 1024.0
            ang = math.atan2(f[i, 1], f[i, 2])
            tol = max(MIN_TOL, math.atan2(MONSTER_RADIUS, max(dist, 1.0)))
            msk[TURN_L] = msk[TURN_R] = 1.0
            if abs(ang) <= tol:
                fists = float(obs["vitals"][2]) <= 1
                lab[FIRE] = 1.0 if (not fists or dist <= FIST_RANGE) else 0.0
                return lab, msk, 1
            lab[TURN_L if ang > 0 else TURN_R] = 1.0
            return lab, msk, 2
        if self.streak >= BLOCK_STREAK:
            d = np.asarray(obs["rays"]["distance"], np.float64).reshape(64)
            left, right = d[1:17].max(), d[48:64].max()        # rays are CCW from straight ahead
            msk[TURN_L] = msk[TURN_R] = 1.0
            lab[TURN_L if left >= right else TURN_R] = 1.0
            return lab, msk, 3
        return lab, msk, 0
