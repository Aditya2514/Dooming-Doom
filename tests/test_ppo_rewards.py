"""PPO fine-tuning reward (training/ppo_rewards.py) on synthetic privileged states."""
from __future__ import annotations

import numpy as np
import pytest

from training.ppo_rewards import RewardConfig, Rewarder


def ps(**kw):
    base = dict(tic=0, episode_ticks=2100, x=0.0, y=0.0, health=100.0, kills=0, damage_taken=0.0,
                keys=np.zeros(6, np.float32), dead=False, exit_reached=False,
                sector_floor=None, sector_ceiling=None, door_lock=None)
    base.update(kw)
    return base


def test_terms():
    c = RewardConfig()
    r = Rewarder(c)
    r.reset(ps(), np.zeros(3))
    assert r.step(ps(tic=4), np.zeros(3)) == pytest.approx(c.time)                  # standing still
    got = r.step(ps(tic=8, kills=1, damage_taken=10.0), np.zeros(3))
    assert got == pytest.approx(c.time + c.kill + 10 * c.damage)
    k = np.zeros(6, np.float32)
    k[2] = 1
    got = r.step(ps(tic=12, kills=1, damage_taken=10.0, keys=k, x=300.0), np.array([300.0, 0, 0]))
    assert got == pytest.approx(c.time + c.key + c.explore + c.curiosity)            # key, new true cell, new own cell


def test_curiosity_decays_and_needs_movement():
    c = RewardConfig()
    r = Rewarder(c)
    r.reset(ps(), np.zeros(3))
    a, b = np.array([100.0, 0, 0]), np.zeros(3)
    first = r.step(ps(tic=4), a) - c.time                       # enter cell (1, 0): 1st visit
    r.step(ps(tic=8), b)                                        # back to the start cell: 2nd visit
    second = r.step(ps(tic=12), a) - c.time                     # cell (1, 0) again: 2nd visit
    assert first == pytest.approx(c.curiosity) and second == pytest.approx(c.curiosity / np.sqrt(2))
    assert r.step(ps(tic=16), a) == pytest.approx(c.time)       # staying put earns nothing


def test_exit_and_early_death():
    c = RewardConfig()
    r = Rewarder(c)
    r.reset(ps(), np.zeros(3))
    assert r.step(ps(tic=1050, exit_reached=True, health=50.0), None) == pytest.approx(
        c.time + c.exit + c.exit_health * 0.5 + c.exit_time * 0.5)
    r.reset(ps(), np.zeros(3))
    assert r.step(ps(tic=525, dead=True), None) == pytest.approx(c.time + c.death * 0.75)
