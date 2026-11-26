"""SpatialMap (submission/doomnet/obs_adapter.py) and the expert's projectile dodge."""
from __future__ import annotations

import math

import numpy as np

from submission.doomnet import obs_adapter as A
from submission.doomnet.actions import INDEX
from training.expert import Expert

C = A.MAP_CROP // 2          # crop centre: rows C-1 / C straddle the agent


def _rays(dist=1.0, hit=0):
    return np.full(64, dist, np.float32), np.full(64, hit, np.int64)


def test_visited_trail_is_behind_after_walking_forward():
    m = A.SpatialMap()
    for k in range(6):                                    # walk +x (heading 0), 64 units per step
        m.update(np.array([64.0 * k, 0.0, 0.0]), *_rays(0.0))
    crop = m.crop(np.array([64.0 * 5, 0.0, 0.0]))
    rows = np.nonzero(crop[0].any(1))[0]
    assert rows.min() >= C - 1                            # nothing visited ahead of us
    assert rows.max() >= C + 3                            # the trail extends behind


def test_crop_is_heading_aligned():
    """The same world point appears in the same crop cell when pose and point rotate together."""
    m = A.SpatialMap()
    m.update(np.array([300.0, 0.0, 0.0]), *_rays(0.0))   # mark one visited cell at (300, 0)
    ahead = m.crop(np.array([0.0, 0.0, 0.0]))[0]
    left = m.crop(np.array([0.0, 0.0, -math.pi / 2]))[0]  # facing -y: the point is now on our left
    r, c = np.argwhere(ahead)[0]
    assert c in (C - 1, C) and r < C - 1                  # straight ahead
    r2, c2 = np.argwhere(left)[0]
    assert r2 in (C - 1, C) and c2 < C - 1                # to the left (col 0 = far left)


def test_rays_mark_seen_and_doors():
    m = A.SpatialMap()
    d, h = _rays(256 / 1024, 1)
    h[0] = 2                                             # a door straight ahead at 256 units
    m.update(np.array([0.0, 0.0, 0.0]), d, h)
    crop = m.crop(np.array([0.0, 0.0, 0.0]))
    assert crop[1].sum() > 20                            # swept free space all around
    door = np.argwhere(crop[2])
    assert len(door) == 1 and door[0][0] < C - 1         # the door is ahead
    m.reset()
    assert not m.crop(np.zeros(3)).any()


def _obs_with_projectile(dist, ang, vf, vr):
    feats = np.zeros((32, 5), np.float32)
    feats[0] = [dist / 1024, math.sin(ang), math.cos(ang), vf, vr]
    types = np.zeros(32, np.int64)
    types[0] = 4
    mask = np.zeros(32, np.int8)
    mask[0] = 1
    return {"visible_entities": {"features": feats, "entity_type": types, "mask": mask},
            "rays": {"distance": np.ones(64, np.float32), "hit_type": np.zeros(64, np.int64)}}


def test_dodge_strafes_away_from_incoming_projectile():
    # fireball 300 units ahead, slightly to our left (5 deg), flying straight at us
    obs = _obs_with_projectile(300, math.radians(5), -10.0, 0.0)
    assert Expert._dodge(obs) == INDEX["STRAFE_R"]
    obs = _obs_with_projectile(300, math.radians(-5), -10.0, 0.0)
    assert Expert._dodge(obs) == INDEX["STRAFE_L"]


def test_no_dodge_for_receding_or_wide_projectiles():
    assert Expert._dodge(_obs_with_projectile(300, 0.0, +10.0, 0.0)) is None          # flying away
    assert Expert._dodge(_obs_with_projectile(300, math.radians(40), -10.0, 0.0)) is None  # passes ~190 units wide
