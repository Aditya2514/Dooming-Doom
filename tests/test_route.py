"""Route guidance (training-only potential shaping): walking distances, locked doors as walls until
the key is held, target switch key -> exit, and the shaping term in the reward."""
import numpy as np
import pytest

from training.rewards import RewardConfig, RewardShaper
from training.route import RouteGuide

# corridor along x: room A [0,256) | locked door [256,288) lock 1 (red card) | room B [288,544)
RECTS = [[0, 0, 256, 128, 0, 128, False, 0], [256, 0, 288, 128, 0, 0, True, 1], [288, 0, 544, 128, 0, 128, False, 0]]
NOKEY = np.zeros(6)
RED = np.array([0, 0, 1, 0, 0, 0.0])          # red card = key flag index 2 (lock 1)


def test_distance_to_key_then_exit_through_the_door():
    g = RouteGuide({"rects": RECTS, "keys": [[200, 64, 1]], "exit_xy": [500, 64]})
    d_far, d_near = g.distance(16, 64, NOKEY), g.distance(160, 64, NOKEY)
    assert d_far is not None and d_near is not None and d_near < d_far          # key first
    d_exit = g.distance(200, 64, RED)                                         # key held: exit, door open
    assert d_exit == pytest.approx(9 * 32, abs=32)


def test_locked_door_is_a_wall_without_the_key():
    g = RouteGuide({"rects": RECTS, "keys": [[400, 64, 1]], "exit_xy": [500, 64]})
    assert g.distance(16, 64, NOKEY) is None            # key behind its own locked door: unknown, no hint
    g2 = RouteGuide({"rects": RECTS, "keys": [], "exit_xy": [500, 64]})
    assert g2.distance(16, 64, NOKEY) is not None       # no keys on the map: exit, doors treated as open


def ps(x, keys=NOKEY, **kw):
    base = dict(tic=0, episode_ticks=2100, x=float(x), y=64.0, z=0.0, angle=0.0, health=100.0, kills=0, items=0,
                damage_taken=0.0, keys=keys, dead=False, timeout=False, exit_reached=False, finished=False,
                sector_floor=None, sector_ceiling=None, door_lock=None)
    base.update(kw)
    return base


def test_shaping_rewards_progress_toward_the_target():
    s = RewardShaper(RewardConfig(stuck=0.0, explore=0.0))
    s.set_route("m", {"rects": RECTS, "keys": [[200, 64, 1]], "exit_xy": [500, 64]})
    s.reset(ps(16))
    fwd, _ = s.step(ps(48), np.zeros(9))
    back, _ = s.step(ps(16), np.zeros(9))
    assert fwd > -0.001 + 0.02 and back < -0.001 - 0.02           # ~ +/-0.032 for one 32-unit cell
    s2 = RewardShaper(RewardConfig(route=0.0))
    s2.set_route("m", {"rects": RECTS, "keys": [], "exit_xy": [500, 64]})
    s2.reset(ps(16))
    assert s2.guide is None


def test_standing_still_earns_no_route_reward():
    s = RewardShaper(RewardConfig(stuck=0.0, explore=0.0))
    s.set_route("m", {"rects": RECTS, "keys": [[200, 64, 1]], "exit_xy": [500, 64]})
    s.reset(ps(16))
    for _ in range(20):
        s.step(ps(16), np.zeros(9))
    assert s.terms["route"] == pytest.approx(0.0, abs=1e-9)
