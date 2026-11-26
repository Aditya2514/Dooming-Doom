"""Shared observation adapter (Spec v2.1): shapes, encoding, tracked inputs, isolation.

The full "training worker vs DoomAgent.act()" identical-tensor test is added once both
exist; until then test_replay_gives_identical_tensors drives the adapter the way each
side will (live, step by step vs replaying a recorded stream) on real wrapper output.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from submission.doomnet import obs_adapter as A

REPO = Path(__file__).resolve().parents[1]


def sample_obs(seed: int) -> dict:
    from doom_env import observation_space

    space = observation_space()
    space.seed(seed)
    return space.sample()


# ----------------------------------------------------------------------------- encoding
@pytest.mark.parametrize("seed", range(5))
def test_shapes_and_dtypes(seed):
    x = A.ObsAdapter().observe(sample_obs(seed))
    assert x["rays"].shape == (7, 64) and x["rays"].dtype == np.float32
    assert x["ent_type"].shape == (32,) and x["ent_type"].dtype == np.int64
    assert x["ent_feat"].shape == (32, 5) and x["ent_feat"].dtype == np.float32
    assert x["ent_mask"].shape == (32,) and x["ent_mask"].dtype == bool
    assert x["scalars"].shape == (28,) and x["scalars"].dtype == np.float32
    assert A.NUM_SCALARS == 28 and A.RAY_CHANNELS == 7


def test_ray_one_hot():
    obs = sample_obs(0)
    hit = np.arange(64) % 6
    obs["rays"]["hit_type"] = hit
    obs["rays"]["distance"] = np.linspace(0, 1, 64, dtype=np.float32)
    x = A.ObsAdapter().observe(obs)
    assert np.allclose(x["rays"][0], obs["rays"]["distance"])
    one_hot = x["rays"][1:]
    assert np.all(one_hot.sum(0) == 1.0)
    assert np.array_equal(one_hot.argmax(0), hit)


def test_vitals_normalisation():
    v = np.zeros(17, np.float32)
    v[A.HEALTH], v[A.ARMOR], v[A.WEAPON], v[A.WEAPON_AMMO] = 100, 50, 7, np.e - 1
    v[4:8] = [0, np.e**5 - 1, 0, 0]
    v[8:14] = [1, 0, 0, 0, 0, 1]
    v[14:17] = [16, -8, 0]
    out = A.normalise_vitals(v)
    assert out[A.HEALTH] == pytest.approx(1.0) and out[A.ARMOR] == pytest.approx(0.5)
    assert out[A.WEAPON] == pytest.approx(1.0)
    assert out[A.WEAPON_AMMO] == pytest.approx(0.2)
    assert out[5] == pytest.approx(1.0, rel=1e-5)
    assert np.array_equal(out[8:14], v[8:14])
    assert np.allclose(out[14:17], [1.0, -0.5, 0.0])


def test_entity_padding_zeroed_and_velocity_scaled():
    obs = sample_obs(1)
    mask = np.zeros(32, np.int8)
    mask[:3] = 1
    obs["visible_entities"]["mask"] = mask
    feats = np.full((32, 5), 32.0, np.float32)
    obs["visible_entities"]["features"] = feats
    obs["visible_entities"]["entity_type"] = np.full(32, 5)
    x = A.ObsAdapter().observe(obs)
    assert np.all(x["ent_feat"][3:] == 0) and np.all(x["ent_type"][3:] == 0)
    assert np.allclose(x["ent_feat"][:3, :3], 32.0) and np.allclose(x["ent_feat"][:3, 3:], 2.0)
    assert np.array_equal(x["ent_mask"], mask.astype(bool))


# ------------------------------------------------------------------------ tracked inputs
def test_tracked_inputs_and_reset():
    ad = A.ObsAdapter()
    o0 = sample_obs(2)
    o0["vitals"][A.HEALTH] = 100
    o0["vitals"][8:14] = 0
    x0 = ad.observe(o0)
    assert np.all(x0["scalars"][A.S_PREV_ACTION] == 0)
    assert x0["scalars"][A.S_PREV_REWARD] == 0 and x0["scalars"][A.S_ELAPSED] == 0

    a = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0])
    ad.record_action(a)
    o1 = {**o0, "vitals": o0["vitals"].copy()}
    o1["vitals"][A.HEALTH] = 80                    # lost 20
    o1["vitals"][8] = 1                            # blue card picked up
    x1 = ad.observe(o1)
    assert np.array_equal(x1["scalars"][A.S_PREV_ACTION], a.astype(np.float32))
    assert x1["scalars"][A.S_PREV_REWARD] == pytest.approx(1.0 - 0.2)
    assert x1["scalars"][A.S_ELAPSED] == pytest.approx(1 / 1050)

    ad.reset()
    x2 = ad.observe(o1)
    assert np.all(x2["scalars"][A.S_PREV_ACTION] == 0)
    assert x2["scalars"][A.S_PREV_REWARD] == 0 and x2["scalars"][A.S_ELAPSED] == 0


def test_replay_gives_identical_tensors():
    """Live adapter inside an env loop (training side) vs a fresh adapter fed the recorded
    observation/action stream (DoomAgent side): every tensor must match exactly, including
    across an episode reset."""
    from doom_env import make_env

    env = make_env(str(REPO / "maps" / "two_rooms.wad"), episode_ticks=160, render_buffers=False)
    rng = np.random.default_rng(0)
    live = A.ObsAdapter()
    stream, live_out = [], []
    try:
        for ep in range(2):
            obs, _ = env.reset(seed=ep)
            live.reset()
            stream.append(("reset", None))
            done = False
            while not done:
                x = live.observe(obs)
                a = rng.integers(0, 2, 9)
                live.record_action(a)
                stream.append(("step", (obs, a)))
                live_out.append(x)
                obs, _, te, tr, _ = env.step(a)
                done = te or tr
    finally:
        env.close()

    agent_side = A.ObsAdapter()
    replay_out = []
    for kind, item in stream:
        if kind == "reset":
            agent_side.reset()
            continue
        obs, a = item
        replay_out.append(agent_side.observe(obs))
        agent_side.record_action(a)

    assert len(live_out) == len(replay_out) == 80          # 2 episodes x 40 decisions
    for x, y in zip(live_out, replay_out):
        for k in A.INPUT_KEYS:
            assert x[k].dtype == y[k].dtype and np.array_equal(x[k], y[k]), k
    # real wrapper output stays in the documented ranges
    b = A.stack(live_out)
    assert b["rays"].shape == (80, 7, 64) and np.all((b["rays"][:, 0] >= 0) & (b["rays"][:, 0] <= 1))
    assert np.all(b["rays"][:, 1:].sum(1) == 1)


def test_to_torch_round_trip():
    torch = pytest.importorskip("torch")
    ad = A.ObsAdapter()
    b = A.stack([ad.observe(sample_obs(s)) for s in range(3)])
    t = A.to_torch(b)
    for k in A.INPUT_KEYS:
        assert np.array_equal(t[k].numpy(), b[k])
    assert t["ent_mask"].dtype == torch.bool and t["ent_type"].dtype == torch.int64


# ---------------------------------------------------------------------------- isolation
def test_submission_never_imports_training():
    bad = []
    for path in (REPO / "submission").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            bad += [f"{path.relative_to(REPO)}: {n}" for n in names if n.split(".")[0] == "training"]
    assert not bad, bad


def test_adapter_has_no_torch_import_at_module_level():
    """Env workers import the adapter; they must not pull in torch."""
    tree = ast.parse((REPO / "submission" / "doomnet" / "obs_adapter.py").read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    mods = [a.name for n in top if isinstance(n, ast.Import) for a in n.names] + \
           [n.module for n in top if isinstance(n, ast.ImportFrom)]
    assert "torch" not in mods
