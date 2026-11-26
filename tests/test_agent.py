"""DoomAgent (submission path) vs the training path, and the network's spec checks."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from submission.doomnet import obs_adapter as A
from submission.doomnet.model import Actor, count_params
from training.model import ActorCritic

REPO = Path(__file__).resolve().parents[1]


def load_agent_module():
    """Load submission/agent.py the way the scorer does (env/doom_env/remote.py)."""
    spec = importlib.util.spec_from_file_location("submission_agent_test", REPO / "submission" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_agent_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parameter_counts_match_spec():
    m = ActorCritic()
    assert count_params(m) == 726_731
    assert count_params(m.actor) == 691_401
    sizes = {n: count_params(getattr(m.actor, n)) for n in ("rays", "entities", "scalars", "fusion", "core", "policy")}
    assert sizes == {"rays": 155_008, "entities": 39_424, "scalars": 1_856, "fusion": 66_304,
                     "core": 394_752, "policy": 34_057}
    assert count_params(m.aux) == 257 and count_params(m.critic) == 35_073


def test_ray_encoder_is_circular():
    """Rotating every ray (and its hit type) by k positions must equal convolving the rotated
    sequence: circular padding means ray 63 neighbours ray 0 (no edge artefacts)."""
    torch.manual_seed(0)
    enc = Actor().rays
    x = torch.randn(1, 7, 64)
    conv = enc.convs[0]
    y = conv(x)
    y_rot = conv(torch.roll(x, 5, dims=-1))
    assert torch.allclose(torch.roll(y, 5, dims=-1), y_rot, atol=1e-5)


def test_no_entities_is_finite():
    torch.manual_seed(0)
    a = Actor()
    B = 2
    x = {"rays": torch.zeros(B, 7, 64), "ent_type": torch.zeros(B, 32, dtype=torch.long),
         "ent_feat": torch.zeros(B, 32, 5), "ent_mask": torch.zeros(B, 32, dtype=torch.bool),
         "scalars": torch.zeros(B, 28)}
    logits, h = a.step(x, a.initial_state(B), torch.ones(B))
    assert torch.isfinite(logits).all() and torch.isfinite(h).all()


def test_agent_matches_training_path_on_real_episode(tmp_path, monkeypatch):
    """Identical-tensor test (Spec v2.1): the submitted DoomAgent and the training-side
    adapter + network give bit-identical logits on the same game, across a map reset."""
    from doom_env import make_env

    torch.manual_seed(3)
    model = ActorCritic()
    weights = tmp_path / "actor.pt"
    torch.save({"state_dict": model.actor.state_dict(), "meta": {"greedy": True}}, weights)
    mod = load_agent_module()
    monkeypatch.setattr(mod, "WEIGHTS_FILE", weights)
    env = make_env(str(REPO / "maps" / "two_rooms.wad"), episode_ticks=120, render_buffers=False)
    agent = mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    assert agent.greedy

    train_adapter = A.ObsAdapter()
    h = model.initial_state(1)
    n = 0
    try:
        for ep in range(2):
            obs, _ = env.reset(seed=ep)
            agent.reset()
            train_adapter.reset()
            start = torch.ones(1)
            done = False
            while not done:
                action = agent.act(obs)
                x = A.stack([train_adapter.observe(obs)])
                with torch.no_grad():
                    logits, h = model.actor.step({k: torch.from_numpy(v) for k, v in x.items()}, h, start)
                start = torch.zeros(1)
                assert torch.equal(logits, agent.last_logits), (ep, n)
                assert np.array_equal(action, (logits[0] > 0).numpy().astype(np.int8))
                assert action.shape == (9,) and action.dtype == np.int8 and set(np.unique(action)) <= {0, 1}
                train_adapter.record_action(action)
                obs, _, te, tr, _ = env.step(action)
                done = te or tr
                n += 1
    finally:
        env.close()
    assert n == 60


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_agent_runs_on_cuda():
    from doom_env import make_env
    mod = load_agent_module()
    env = make_env(str(REPO / "maps" / "two_rooms.wad"), episode_ticks=40, render_buffers=False)
    try:
        agent = mod.DoomAgent(env.observation_space, env.action_space, "cuda")
        obs, _ = env.reset(seed=0)
        a = agent.act(obs)
        assert a.shape == (9,)
    finally:
        env.close()
