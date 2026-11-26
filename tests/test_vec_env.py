"""Vector env on the real wrapper (Spec v2.1 T+0 'Autoreset test', F1 endings, map switching)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from submission.doomnet import obs_adapter as A
from training.vec_env import DoomVecEnv, MapRunner

REPO = Path(__file__).resolve().parents[1]
TEST_MAP = str(REPO / "maps" / "two_rooms.wad")

# buttons that never move the player (turn, attack, use, weapon): reward is the time term only
STILL = [4, 5, 6, 7, 8]


def scripted(t: int, n: int) -> np.ndarray:
    a = np.zeros((n, 9), np.int8)
    for i in range(n):
        a[i, STILL[(t + i) % len(STILL)]] = 1
        a[i, STILL[(t + 2 * i + 1) % len(STILL)]] = 1
    return a


def test_autoreset_every_transition_is_real():
    """Force 3-decision episodes; every stored transition must match the action the engine
    executed, with reward and end flags from the right episode (same-step auto-reset)."""
    n = 2
    with DoomVecEnv([{"wad": TEST_MAP, "map": "MAP01", "ticks": 12}], n_envs=n, seed=0) as env:
        obs = env.reset()
        assert not env.torch_in_workers
        assert np.all(obs["scalars"][:, A.S_PREV_ACTION] == 0) and np.all(obs["scalars"][:, A.S_ELAPSED] == 0)
        n_eps = 0
        for t in range(9):
            a = scripted(t, n)
            obs, r, term, trunc, info = env.step(a)
            ends = (t % 3) == 2
            assert np.all(term == ends) and not trunc.any(), (t, term, trunc)
            assert np.allclose(r, -0.001), r                     # time term only, from THIS episode
            if ends:
                # returned obs is the next episode's first observation
                assert np.all(obs["scalars"][:, A.S_PREV_ACTION] == 0)
                assert np.all(obs["scalars"][:, A.S_ELAPSED] == 0)
                eps = info["episodes"]
                assert len(eps) == n and all(e["length"] == 3 and e["tick_cap"] and not e["cutoff"] for e in eps)
                n_eps += len(eps)
            else:
                # the next observation records exactly the buttons the engine executed
                assert np.array_equal(obs["scalars"][:, A.S_PREV_ACTION], a.astype(np.float32))
                assert np.allclose(obs["scalars"][:, A.S_ELAPSED], ((t % 3) + 1) / A.MAX_DECISIONS)
                assert not info["episodes"]
        assert n_eps == 3 * n


def test_cutoff_is_trunc_with_genuine_final_obs():
    with DoomVecEnv([{"wad": TEST_MAP, "map": "MAP01", "ticks": 400}], n_envs=1, seed=0, cutoff=4) as env:
        env.reset()
        for t in range(8):
            a = scripted(t, 1)
            obs, r, term, trunc, info = env.step(a)
            cut = (t % 4) == 3
            assert bool(trunc[0]) == cut and not term[0]
            if cut:
                fo = info["final_obs"]
                assert np.array_equal(fo["scalars"][0, A.S_PREV_ACTION], a[0].astype(np.float32))
                assert fo["scalars"][0, A.S_ELAPSED] == pytest.approx(4 / A.MAX_DECISIONS)
                assert obs["scalars"][0, A.S_ELAPSED] == 0
                assert info["episodes"][0]["cutoff"] and not info["episodes"][0]["tick_cap"]


def test_in_place_map_switch_equals_fresh_engine():
    """Switching maps inside one engine must give exactly the observations of a fresh engine."""
    from doom_env import make_env
    from doom_env.wrapper import IWAD

    rng = np.random.default_rng(0)
    actions = rng.integers(0, 2, (25, 9))

    def run(env):
        obs, _ = env.reset(seed=7)
        out = [obs]
        for a in actions:
            obs, _, te, tr, _ = env.step(a)
            out.append(obs)
            if te or tr:
                break
        return out

    runner = MapRunner(fix_ipc=False)
    try:
        e = runner.load({"wad": IWAD, "map": "MAP01", "ticks": 400})
        run(e)
        e = runner.load({"wad": IWAD, "map": "MAP02", "ticks": 200})
        assert e.map_name == "MAP02" and e.episode_ticks == 200
        switched = run(e)
    finally:
        runner.close()
    fresh_env = make_env(IWAD, "MAP02", episode_ticks=200, render_buffers=False)
    try:
        fresh = run(fresh_env)
    finally:
        fresh_env.close()
    assert len(switched) == len(fresh)
    for o1, o2 in zip(switched, fresh):
        assert np.array_equal(o1["vitals"], o2["vitals"])
        for k in ("distance", "hit_type"):
            assert np.array_equal(o1["rays"][k], o2["rays"][k])
        for k in ("features", "entity_type", "mask"):
            assert np.array_equal(o1["visible_entities"][k], o2["visible_entities"][k])


def test_schedule_order_per_worker():
    sched = [{"wad": TEST_MAP, "map": "MAP01", "ticks": 8, "label": f"m{i}"} for i in range(5)]
    with DoomVecEnv(sched, n_envs=2, seed=0) as env:
        env.reset()
        seen = {0: [], 1: []}
        for t in range(6):
            _, _, _, _, info = env.step(np.zeros((2, 9), np.int8))
            for e in info["episodes"]:
                seen[e["worker"]].append(e["map"])
    assert seen[0] == ["m0", "m2", "m4"] and seen[1] == ["m1", "m3", "m0"]


def test_configure_switches_schedule_at_next_episode():
    a = [{"wad": TEST_MAP, "map": "MAP01", "ticks": 8, "label": "a"}]
    b = [{"wad": TEST_MAP, "map": "MAP01", "ticks": 12, "label": "b"}]
    with DoomVecEnv(a, n_envs=2, seed=0) as env:
        env.reset()
        env.step(np.zeros((2, 9), np.int8))                 # mid-episode (2-decision episodes)
        env.configure(schedule=b)
        labels = []
        for _ in range(6):
            _, _, _, _, info = env.step(np.zeros((2, 9), np.int8))
            labels += [(e["map"], e["length"]) for e in info["episodes"]]
    assert labels[:2] == [("a", 2), ("a", 2)]               # running episodes keep their settings
    assert set(labels[2:]) == {("b", 3)}                    # next episodes use the new schedule
