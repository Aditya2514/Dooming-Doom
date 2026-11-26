"""The privileged teacher (training/expert.py) on real generated maps."""
from __future__ import annotations

import numpy as np
import pytest

from training.expert import Expert, NavGrid

gen = pytest.importorskip("training.maps.gen")


def _entries(family, n):
    try:
        return gen.load_index("validation", family)[:n]
    except FileNotFoundError:
        pytest.skip("generated maps missing: python -m training.maps.gen --all")


def test_navgrid_marks_floor_and_locks():
    e = _entries("keydoor", 1)[0]
    g = NavGrid(e["route"])
    assert g.safe.sum() > 100 and (g.safe <= g.walk).all()
    assert (g.lock_of > 0).any()
    sx, sy = e["route"]["rects"][0][0] + 64, e["route"]["rects"][0][1] + 64
    i, j = g.cell(sx, sy)
    assert g.walk[i, j]
    # locked door cells are not passable without the key, passable with it
    P0, *_ = g.graph(frozenset())
    lk = int(g.lock_of[g.lock_of > 0][0])
    P1, *_ = g.graph(frozenset({lk}))
    assert (P1.sum() > P0.sum()) and not (P0 & (g.lock_of == lk)).any()


@pytest.mark.parametrize("family", ["rooms", "keydoor"])
def test_expert_reaches_exit(family):
    from submission.doomnet.actions import to_buttons
    from training.maprunner import MapRunner

    runner = MapRunner()
    try:
        for e in _entries(family, 2):
            env = runner.load(e)
            obs, _ = env.reset(seed=3)
            ex = Expert(e["route"])
            ex.reset()
            for _ in range(int(e["ticks"]) // 4 + 1):
                obs, _, te, tr, _ = env.step(to_buttons(ex.act(obs, env.privileged_state())))
                if te or tr:
                    break
            assert env.privileged_state()["exit_reached"], e["label"]
    finally:
        runner.close()


def test_labels_follow_executed_actions():
    """DAgger: when the STUDENT turns in place (no forward presses), the teacher must not read that as
    'stuck while moving' and label 'back up' (that fed a back-up/turn loop); and the USE cooldown
    counts real presses only."""
    from submission.doomnet.actions import INDEX
    from training.maprunner import MapRunner

    e = _entries("rooms", 1)[0]
    runner = MapRunner()
    try:
        env = runner.load(e)
        obs, _ = env.reset(seed=0)
        ex = Expert(e["route"])
        ex.reset()
        from submission.doomnet.actions import to_buttons
        labels = []
        for _ in range(16):
            labels.append(ex.act(obs, env.privileged_state()))
            ex.executed(INDEX["TURN_L"])                       # the student spins instead
            obs, *_ = env.step(to_buttons(INDEX["TURN_L"]))
        assert INDEX["BACK"] not in labels
    finally:
        runner.close()
