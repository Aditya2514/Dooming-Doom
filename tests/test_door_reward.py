"""Deterministic door-reward test on a real WAD (Spec v2.1 Rewards: 'Deterministic test
before enabling'). Scripted actions: walk forward; when blocked, press USE once (button-down)
and keep walking. Door rewards may only be enabled in training while this test passes."""
from __future__ import annotations

import numpy as np
import pytest

from training.maps import doortest
from training.rewards import RewardConfig, RewardShaper

FORWARD, USE = 0, 7


@pytest.fixture(scope="module")
def wad():
    return str(doortest.write())


def drive(wad: str, map_name: str, steps: int = 400, reopen: bool = False):
    from doom_env import make_env

    env = make_env(wad, map_name, episode_ticks=4200, render_buffers=False)
    try:
        env.reset(seed=0)
        ps = env.privileged_state()
        sh = RewardShaper(RewardConfig(doors_enabled=True))
        sh.reset(ps)
        xs, wait, uses, clear_log = [ps["x"]], 0, 0, []
        for t in range(steps):
            a = np.zeros(9, np.int64)
            stuck = len(xs) >= 4 and abs(xs[-1] - xs[-4]) < 1.0
            if wait > 0:
                wait -= 1
                a[FORWARD] = 0 if reopen else 1
            elif stuck and (not reopen or uses < 2):
                a[USE] = 1
                uses += 1
                wait = 90 if reopen else 25        # reopen: stand still until the door has closed
            else:
                a[FORWARD] = 1
            _, _, te, tr, _ = env.step(a)
            ps = env.privileged_state()
            sh.step(ps, a)
            xs.append(ps["x"])
            clear_log.append(float(ps["sector_ceiling"][1] - ps["sector_floor"][1]))
            if te or tr or ps["x"] > 1250:
                break
        return sh, ps, clear_log
    finally:
        env.close()


def test_doors_with_key(wad):
    sh, ps, _ = drive(wad, "MAP01")
    assert ps["keys"].tolist() == [0, 0, 1, 0, 0, 0]          # red card picked up on the way
    assert ps["x"] > 1150                                     # went through all four doors
    assert sh.counts["doors"] == 2 and sh.counts["locked_doors"] == 1
    assert sh.terms["door"] == pytest.approx(1.0)             # A + D (raised floor); C open at reset: 0
    assert sh.terms["locked_door"] == pytest.approx(2.0)
    assert sh.counts["doors_without_use"] == 0


def test_locked_door_without_key(wad):
    sh, ps, _ = drive(wad, "MAP02")
    assert ps["keys"].sum() == 0
    assert ps["x"] < 1120                                     # stopped at the locked door
    assert sh.terms["locked_door"] == 0.0 and sh.counts["locked_doors"] == 0
    assert sh.terms["door"] == pytest.approx(1.0)


def test_reopening_pays_once(wad):
    sh, ps, clear = drive(wad, "MAP03", steps=260, reopen=True)
    c = np.array(clear)
    opened = np.nonzero(c >= 56)[0]
    assert opened.size, "door never opened"
    closed_again = np.nonzero(c[opened[0]:] < 56)[0]
    assert closed_again.size, "door never closed again"
    assert (c[opened[0] + closed_again[0]:] >= 56).any(), "door never reopened"
    assert sh.terms["door"] == pytest.approx(0.5)
