"""Submission path: macro-action table, network, and DoomAgent contract."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from doom_env import action_space, observation_space
from submission.doomnet import obs_adapter as A
from submission.doomnet.actions import ACTIONS, BUTTONS, NUM_ACTIONS, to_buttons
from submission.doomnet.policy import INPUT_KEYS, DoomPolicy

REPO = Path(__file__).resolve().parents[1]


def load_agent_module():
    """Load submission/agent.py the way the scorer does."""
    spec = importlib.util.spec_from_file_location("submission_agent_test", REPO / "submission" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_agent_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_action_table_is_valid():
    assert BUTTONS.shape == (NUM_ACTIONS, 9)
    assert set(np.unique(BUTTONS)) <= {0, 1}
    assert len({tuple(r) for r in BUTTONS}) == NUM_ACTIONS          # no duplicates
    for row in BUTTONS:
        assert not (row[0] and row[1]) and not (row[2] and row[3]) and not (row[4] and row[5])  # no contradictions
    assert action_space().contains(to_buttons(1).astype(np.int8))
    assert [n for n, _ in ACTIONS][0] == "NOOP"


def _obs_seq(n, seed=0):
    space = observation_space()
    space.seed(seed)
    ad = A.ObsAdapter()
    xs = []
    rng = np.random.default_rng(seed)
    for _ in range(n):
        xs.append(ad.observe(space.sample()))
        ad.record_action(BUTTONS[rng.integers(NUM_ACTIONS)])
    return xs


def test_step_matches_sequence():
    """Acting (one step at a time) and training (whole episode through nn.GRU) compute the same logits."""
    torch.manual_seed(0)
    net = DoomPolicy().eval()
    xs = _obs_seq(12)
    batch = A.stack(xs)
    seq = {k: torch.from_numpy(batch[k]).unsqueeze(1) for k in INPUT_KEYS}      # (T, 1, ...)
    with torch.no_grad():
        logits_seq, _, _ = net(seq, net.initial_state(1))
    h = net.initial_state(1)
    for t in range(len(xs)):
        lt, h = net.step({k: torch.from_numpy(np.asarray(xs[t][k])[None]) for k in INPUT_KEYS}, h)
        assert torch.allclose(lt[0], logits_seq[t, 0], atol=1e-5)


def test_entity_encoder_handles_no_entities():
    net = DoomPolicy().eval()
    x = A.stack(_obs_seq(1))
    x["ent_mask"][:] = False
    x["ent_type"][:] = 0
    x["ent_feat"][:] = 0
    lt, _ = net.step({k: torch.from_numpy(x[k]) for k in INPUT_KEYS}, net.initial_state(1))
    assert torch.isfinite(lt).all()


def test_agent_contract(tmp_path):
    """DoomAgent works with and without weights, returns valid actions, and reset() is deterministic."""
    mod = load_agent_module()
    net = DoomPolicy()
    w = tmp_path / "actor.pt"
    torch.save({"state_dict": net.state_dict(), "meta": {"greedy": False}}, w)
    mod.WEIGHTS_FILE = w
    agent = mod.DoomAgent(observation_space(), action_space(), "cpu")
    space = observation_space()
    space.seed(1)
    obs = [space.sample() for _ in range(20)]

    def play():
        agent.reset()
        out = []
        for o in obs:
            a = agent.act(o)
            assert a.shape == (9,) and set(np.unique(a)) <= {0, 1}
            assert action_space().contains(np.asarray(a, np.int8))
            out.append(a.copy())
        return np.stack(out)

    assert np.array_equal(play(), play())


def test_submission_does_not_import_training():
    import ast

    for f in (REPO / "submission").rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n == "training" or n.startswith("training.") for n in names), f
