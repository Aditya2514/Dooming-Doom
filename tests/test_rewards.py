"""Training reward terms and critic inputs on synthetic privileged states (Spec v2.1 Rewards).

The WAD-based deterministic door test (scripted actions on a test map with an ordinary
door, a locked door, a door open at reset and an elevated floor) lands with the map
generator; door rewards stay disabled until it passes."""
from __future__ import annotations

import numpy as np
import pytest

from training.rewards import PRIV_DIM, RewardConfig, RewardShaper

# sectors: 0 room, 1 ordinary door (closed), 2 locked door (closed), 3 ordinary door open at reset
LOCK = np.array([-1, 0, 2, 0])
FLOOR = np.zeros(4)
CEIL0 = np.array([128.0, 0.0, 0.0, 72.0])


def ps(**kw):
    base = dict(tic=0, episode_ticks=2100, x=0.0, y=0.0, z=0.0, angle=0.0, health=100.0, kills=0, items=0,
                damage_taken=0.0, keys=np.zeros(6, np.float32), dead=False, timeout=False, exit_reached=False,
                finished=False, sector_floor=FLOOR.copy(), sector_ceiling=CEIL0.copy(), door_lock=LOCK.copy())
    base.update(kw)
    return base


NOOP = np.zeros(9)
USE = np.eye(9)[7]


def shaper(**cfg):
    s = RewardShaper(RewardConfig(**cfg))
    s.reset(ps())
    return s


def test_time_only_when_nothing_happens():
    s = shaper()
    r, aux = s.step(ps(tic=4), NOOP)
    assert r == pytest.approx(-0.001) and not aux


def test_exit_bonus_health_and_time():
    s = shaper()
    r, _ = s.step(ps(tic=1050, health=50.0, exit_reached=True), NOOP)
    assert r == pytest.approx(10 + 2 * 0.5 + 2 * 0.5 - 0.001)


def test_early_death_scaled_and_aux():
    s = shaper()
    r, aux = s.step(ps(tic=525, dead=True, health=0.0, damage_taken=100.0), NOOP)
    assert r == pytest.approx(-5 * 0.75 - 0.01 * 100 - 0.001) and aux


def test_keys_kills_damage():
    s = shaper()
    k = np.zeros(6, np.float32)
    k[0] = 1
    r, aux = s.step(ps(tic=4, keys=k, kills=2, damage_taken=10.0), NOOP)
    assert r == pytest.approx(2.0 + 2 * 0.5 - 0.1 - 0.001) and aux
    r, _ = s.step(ps(tic=8, keys=k, kills=2, damage_taken=10.0), NOOP)       # key held, no new key
    assert r == pytest.approx(-0.001)


def test_exploration_first_visit_and_cap():
    s = shaper(explore_cap=0.05)
    r1, _ = s.step(ps(x=200.0), NOOP)          # new cell
    r2, _ = s.step(ps(x=210.0), NOOP)          # same cell
    r3, _ = s.step(ps(x=0.0), NOOP)            # spawn cell, already visited
    r4, _ = s.step(ps(x=400.0), NOOP)          # new cell
    r5, _ = s.step(ps(x=600.0), NOOP)          # new cell, but only 0.01 of the cap is left
    assert [round(r + 0.001, 6) for r in (r1, r2, r3, r4, r5)] == [0.02, 0.0, 0.0, 0.02, 0.01]


def _ceil(door=0.0, locked=0.0, open_=72.0):
    return np.array([128.0, door, locked, open_])


def test_doors_off_by_default_but_tracked():
    s = shaper()
    r, _ = s.step(ps(sector_ceiling=_ceil(door=60.0)), USE)
    assert r == pytest.approx(-0.001) and s.counts["doors"] == 1


def test_door_rules_when_enabled():
    s = shaper(doors_enabled=True)
    r, _ = s.step(ps(sector_ceiling=_ceil(door=30.0)), USE)               # opening, not yet 56
    assert r == pytest.approx(-0.001)
    r, _ = s.step(ps(sector_ceiling=_ceil(door=56.0)), NOOP)              # crosses 56: +0.5 once
    assert r == pytest.approx(0.5 - 0.001)
    r, _ = s.step(ps(sector_ceiling=_ceil(door=0.0)), NOOP)               # closes
    r, _ = s.step(ps(sector_ceiling=_ceil(door=64.0)), NOOP)              # re-opens: 0
    assert r == pytest.approx(-0.001)
    r, _ = s.step(ps(sector_ceiling=_ceil(door=64.0, locked=60.0)), USE)  # locked sector: +2.0
    assert r == pytest.approx(2.0 - 0.001)
    r, _ = s.step(ps(sector_ceiling=_ceil(door=64.0, locked=60.0, open_=72.0)), NOOP)
    assert r == pytest.approx(-0.001)                                      # open at reset: never paid
    assert s.counts == {**s.counts, "doors": 1, "locked_doors": 1}


def test_opening_without_recent_use_is_counted():
    s = shaper()
    for t in range(45):
        s.step(ps(tic=4 * t), NOOP)
    s.step(ps(sector_ceiling=_ceil(door=60.0)), NOOP)
    assert s.counts["doors_without_use"] == 1


def test_critic_features_shape_and_ticks_remaining():
    s = shaper()
    f = s.features(ps(tic=525))
    assert f.shape == (PRIV_DIM,) and f.dtype == np.float32
    assert f[0] == pytest.approx(0.75)
    assert f[13] == pytest.approx(1.0)              # one locked sector, not opened yet


def test_ammo_spent_costs_pickups_free():
    s = RewardShaper(RewardConfig())
    s.reset(ps(ammo_total=50.0))
    r, _ = s.step(ps(ammo_total=47.0), NOOP)               # 3 shots
    assert r == pytest.approx(-0.001 + 3 * -0.002)
    r, _ = s.step(ps(ammo_total=57.0), NOOP)               # picked up a clip
    assert r == pytest.approx(-0.001)
    assert s.terms["ammo"] == pytest.approx(-0.006)


def test_stuck_cost_when_no_progress_over_8_decisions():
    s = shaper()
    rs = [s.step(ps(x=0.0, y=0.0), NOOP)[0] for _ in range(10)]       # standing still / pushing a wall
    assert rs[:7] == [pytest.approx(-0.001)] * 7                       # window not full yet
    assert rs[7] == pytest.approx(-0.001 - 0.003) and s.counts["stuck"] == 3
    s = shaper()
    rs = [s.step(ps(x=10.0 * k, y=0.0), NOOP)[0] for k in range(1, 12)]  # moving 10 units per decision
    assert all(r <= -0.001 + 0.02 + 1e-9 for r in rs) and s.counts["stuck"] == 0


def test_stuck_cost_not_charged_on_the_final_step():
    s = shaper()
    for _ in range(8):
        s.step(ps(), NOOP)
    r, _ = s.step(ps(exit_reached=True, finished=True, tic=1000), NOOP)
    assert s.terms["stuck"] == pytest.approx(-0.003) and r == pytest.approx(10 + 2 + 2 * (1 - 1000 / 2100) - 0.001)
