"""Reaction labels (training/teacher.py): fire on target, turn towards enemies, unstick."""
from __future__ import annotations

import math

import numpy as np
import torch

from training.teacher import FIRE, FWD, TURN_L, TURN_R, ReactionTeacher


def obs_with(enemy_deg=None, dist=300.0, etype=2, weapon=2, left_open=False):
    feats = np.zeros((32, 5), np.float32)
    types = np.zeros(32, np.int64)
    mask = np.zeros(32, np.int8)
    if enemy_deg is not None:
        a = math.radians(enemy_deg)
        feats[0] = [dist / 1024, math.sin(a), math.cos(a), 0, 0]
        types[0], mask[0] = etype, 1
    d = np.full(64, 0.05, np.float32)
    if left_open:
        d[8] = 0.9
    else:
        d[56] = 0.9
    vit = np.zeros(17, np.float32)
    vit[2] = weapon
    return {"visible_entities": {"features": feats, "entity_type": types, "mask": mask},
            "rays": {"distance": d, "hit_type": np.ones(64, np.int64)}, "vitals": vit}


PS = {"x": 0.0, "y": 0.0}


def test_fire_only_on_target():
    t = ReactionTeacher()
    lab, msk, sit = t.labels(obs_with(0.5), PS, None)
    assert sit == 1 and lab[FIRE] == 1 and msk[FIRE] == 1 and lab[TURN_L] == lab[TURN_R] == 0
    lab, msk, sit = t.labels(obs_with(None), PS, None)
    assert sit == 0 and lab[FIRE] == 0 and msk[FIRE] == 1 and msk[TURN_L] == 0     # no enemy: hold fire
    lab, _, _ = t.labels(obs_with(0.5, etype=9), PS, None)                       # a key is not an enemy
    assert lab[FIRE] == 0


def test_turn_towards_enemy_left_positive():
    t = ReactionTeacher()
    lab, msk, sit = t.labels(obs_with(30), PS, None)
    assert sit == 2 and lab[TURN_L] == 1 and lab[TURN_R] == 0 and lab[FIRE] == 0 and msk[TURN_R] == 1
    lab, _, _ = t.labels(obs_with(-30), PS, None)
    assert lab[TURN_R] == 1 and lab[TURN_L] == 0


def test_tolerance_grows_when_close_and_fists_need_range():
    t = ReactionTeacher()
    assert t.labels(obs_with(8, dist=100), PS, None)[2] == 1      # atan(20/100) = 11 deg
    assert t.labels(obs_with(8, dist=600), PS, None)[2] == 2      # atan(20/600) = 2 deg
    lab, _, sit = t.labels(obs_with(0, dist=300, weapon=1), PS, None)
    assert sit == 1 and lab[FIRE] == 0                            # fists out of reach
    assert t.labels(obs_with(0, dist=60, weapon=1), PS, None)[0][FIRE] == 1


def test_blocked_turns_to_open_side():
    t = ReactionTeacher()
    fwd = np.zeros(9)
    fwd[FWD] = 1
    t.labels(obs_with(), PS, None)
    assert t.labels(obs_with(), PS, fwd)[2] == 0                  # one blocked decision: not yet
    lab, msk, sit = t.labels(obs_with(left_open=True), PS, fwd)
    assert sit == 3 and lab[TURN_L] == 1 and msk[TURN_L] == msk[TURN_R] == 1
    lab, _, _ = t.labels(obs_with(left_open=False), PS, fwd)
    assert lab[TURN_R] == 1
    assert t.labels(obs_with(), {"x": 50.0, "y": 0.0}, fwd)[2] == 0   # moving again: streak over


def test_bc_loss_teaches_fire_on_target():
    """The PPO loss term on its own: a few optimiser steps make FIRE follow the label."""
    torch.manual_seed(0)
    logits_param = torch.zeros(64, 9, requires_grad=True)
    lab = torch.zeros(64, 9)
    lab[:32, FIRE] = 1.0
    msk = torch.zeros(64, 9)
    msk[:, FIRE] = 1.0
    opt = torch.optim.Adam([logits_param], lr=0.1)
    for _ in range(50):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits_param, lab, reduction="none")
        loss = (bce * msk).sum() / msk.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    p = torch.sigmoid(logits_param[:, FIRE])
    assert p[:32].min() > 0.9 and p[32:].max() < 0.1
    assert torch.equal(logits_param[:, TURN_L], torch.zeros(64))  # unlabelled buttons untouched
