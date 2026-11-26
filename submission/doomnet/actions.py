"""Macro-action table: the policy picks ONE of these per decision; each maps to the 9 official buttons.

Why not 9 independent buttons: with independent Bernoulli outputs an untrained (or uncertain)
policy presses contradictory combinations (forward + backward, turn left + turn right) and spams
USE, which makes doors bounce (USE on a moving door reverses it). A categorical over a small set
of meaningful combinations removes those failure modes by construction and makes both imitation
(one cross-entropy label per step) and RL credit assignment far easier.

Button order (docs/CONTRACT.md section 3):
    0 FORWARD  1 BACKWARD  2 MOVE_LEFT  3 MOVE_RIGHT  4 TURN_LEFT  5 TURN_RIGHT  6 ATTACK  7 USE  8 NEXT_WEAPON
"""
from __future__ import annotations

import numpy as np

FWD, BACK, STR_L, STR_R, TURN_L, TURN_R, ATTACK, USE, NEXT_WEAPON = range(9)

ACTIONS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("NOOP", ()),
    ("FWD", (FWD,)),
    ("FWD_L", (FWD, TURN_L)),
    ("FWD_R", (FWD, TURN_R)),
    ("TURN_L", (TURN_L,)),
    ("TURN_R", (TURN_R,)),
    ("BACK", (BACK,)),
    ("STRAFE_L", (STR_L,)),
    ("STRAFE_R", (STR_R,)),
    ("ATK", (ATTACK,)),
    ("ATK_L", (ATTACK, TURN_L)),
    ("ATK_R", (ATTACK, TURN_R)),
    ("FWD_ATK", (FWD, ATTACK)),
    ("USE", (USE,)),
    ("FWD_USE", (FWD, USE)),
    ("BACK_ATK", (BACK, ATTACK)),
)
NUM_ACTIONS = len(ACTIONS)
NAMES = tuple(n for n, _ in ACTIONS)
INDEX = {n: i for i, n in enumerate(NAMES)}

BUTTONS = np.zeros((NUM_ACTIONS, 9), np.int64)
for _i, (_n, _b) in enumerate(ACTIONS):
    BUTTONS[_i, list(_b)] = 1


def to_buttons(a: int) -> np.ndarray:
    """Macro-action index -> (9,) int64 array of 0/1 buttons (a fresh copy)."""
    return BUTTONS[int(a)].copy()
