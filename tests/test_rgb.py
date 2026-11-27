"""RGB branch: adapter output, zero-initialised widening, loading, and the exit-label helper."""
import math

import numpy as np
import pytest
import torch

from submission.doomnet import obs_adapter as A
from submission.doomnet.policy import INPUT_KEYS, DoomPolicy, policy_from_state_dict


def test_downsample_rgb_shape_and_values():
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:4, :4, 0] = 200                                  # one 4x4 block, red channel
    out = A.downsample_rgb(frame)
    assert out.shape == A.RGB_SHAPE == (3, 30, 40) and out.dtype == np.uint8
    assert out[0, 0, 0] == 200 and out[1:, 0, 0].sum() == 0 and out[0, 0, 1] == 0


def _batch(n=6):
    from doom_env import observation_space

    ad, sp, xs = A.ObsAdapter(), observation_space(), []
    for _ in range(n):
        xs.append(ad.observe(sp.sample()))
        ad.record_action(np.zeros(9))
    b = A.stack(xs)
    return {k: torch.from_numpy(b[k]) for k in INPUT_KEYS}


def test_widening_keeps_the_function():
    torch.manual_seed(0)
    old = DoomPolicy(use_rgb=False).eval()
    new = policy_from_state_dict(old.state_dict(), use_rgb=True).eval()
    assert new.use_rgb and not old.use_rgb
    x = _batch()
    lo, _ = old.step(x, old.initial_state(6))
    ln, _ = new.step(x, new.initial_state(6))
    assert torch.allclose(lo, ln, atol=1e-5)


def test_rgb_checkpoint_roundtrip_and_aux_head():
    torch.manual_seed(1)
    net = DoomPolicy(use_rgb=True).eval()
    again = policy_from_state_dict(net.state_dict()).eval()
    assert again.use_rgb
    x = _batch()
    assert torch.allclose(net.step(x, net.initial_state(6))[0], again.step(x, again.initial_state(6))[0])
    assert net.exit_logits(x["rgb"]).shape == (6, 5)


def test_exit_view_class():
    from training.expert import exit_view_class

    rays = {"distance": np.ones(64, np.float32), "hit_type": np.ones(64, np.int64)}
    obs = {"rays": rays}
    ps = {"x": 0.0, "y": 0.0, "angle": 0.0}
    assert exit_view_class(obs, ps, None) == 0
    assert exit_view_class(obs, ps, (200.0, 0.0)) in (2, 3)            # straight ahead: a middle bin
    assert exit_view_class(obs, ps, (-200.0, 0.0)) == 0                # behind
    assert exit_view_class(obs, ps, (5000.0, 0.0)) == 0                # too far
    left = exit_view_class(obs, ps, (200.0, 200.0 * math.tan(math.radians(30))))
    right = exit_view_class(obs, ps, (200.0, -200.0 * math.tan(math.radians(30))))
    assert left < right                                                  # bins run left to right
    blocked = {"rays": {"distance": np.full(64, 0.05, np.float32), "hit_type": np.ones(64, np.int64)}}
    assert exit_view_class(blocked, ps, (200.0, 0.0)) == 0              # wall in between


def test_agent_refuses_missing_weights(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    from doom_env import action_space, observation_space

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("agent_missing", repo / "submission" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.WEIGHTS_FILE = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError):
        mod.DoomAgent(observation_space(), action_space(), "cpu")
