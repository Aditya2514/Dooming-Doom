"""Door / key counting (training/metrics.py, our stand-in for the scorer's progression metric) on a
real WAD. Scripted actions: walk forward; when blocked, press USE once (button-down) and keep walking."""
from __future__ import annotations

import numpy as np
import pytest

from training.maps import doortest
from training.metrics import EpisodeMetrics

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
        sh = EpisodeMetrics()
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
            sh.step(ps)
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
    assert sh.counts["doors"] == 2 and sh.counts["locked_doors"] == 1   # A + D (raised floor); C open at reset
    assert sh.counts["keys"] == 1


def test_locked_door_without_key(wad):
    sh, ps, _ = drive(wad, "MAP02")
    assert ps["keys"].sum() == 0
    assert ps["x"] < 1120                                     # stopped at the locked door
    assert sh.counts["locked_doors"] == 0 and sh.counts["doors"] == 2


def test_reopening_counts_once(wad):
    sh, ps, clear = drive(wad, "MAP03", steps=260, reopen=True)
    c = np.array(clear)
    opened = np.nonzero(c >= 56)[0]
    assert opened.size, "door never opened"
    closed_again = np.nonzero(c[opened[0]:] < 56)[0]
    assert closed_again.size, "door never closed again"
    assert (c[opened[0] + closed_again[0]:] >= 56).any(), "door never reopened"
    assert sh.counts["doors"] == 1


@pytest.mark.parametrize("lock", range(1, 7))
def test_every_lock_opens_only_with_its_key(lock, tmp_path):
    """Cards and skulls (ZDoom locks 1-6) used by the key-door maps: the matching key opens the
    door (and sets exactly one key flag); without it the door stays shut."""
    from doom_env import make_env
    from training.maps import udmf

    def corridor(key):
        R = udmf.Rect
        rects = [R(0, 0, 256, 128), R(256, 0, 288, 128, door=True, lock=lock), R(288, 0, 544, 128)]
        things = [udmf.Thing(64, 64, udmf.THING_PLAYER1, 0)] + ([udmf.Thing(160, 64, key)] if key else [])
        return udmf.build_textmap(rects, things)

    wad = str(udmf.write_wad(tmp_path / "locks.wad", {"MAP01": corridor(udmf.KEY_FOR_LOCK[lock]), "MAP02": corridor(None)}))

    def drive(m):
        env = make_env(wad, m, episode_ticks=2000, render_buffers=False)
        try:
            env.reset(seed=0)
            xs, wait = [], 0
            for _ in range(120):
                ps = env.privileged_state()
                xs.append(ps["x"])
                a = np.zeros(9, np.int64)
                if wait:
                    wait -= 1
                    a[FORWARD] = 1
                elif len(xs) > 4 and abs(xs[-1] - xs[-4]) < 1:
                    a[USE], wait = 1, 25
                else:
                    a[FORWARD] = 1
                env.step(a)
            return env.privileged_state()
        finally:
            env.close()

    with_key, without = drive("MAP01"), drive("MAP02")
    assert with_key["x"] > 300 and with_key["keys"].sum() == 1
    assert without["x"] < 256 and without["keys"].sum() == 0
