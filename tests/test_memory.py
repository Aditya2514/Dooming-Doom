"""Episodic memory: write rules and the T+6 gate tests (Spec v2.1, Memory 'Tests' + warm start)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from submission.doomnet import obs_adapter as A
from submission.doomnet.memory import N_RECENT, N_SLOTS, MemoryBook
from submission.doomnet.model import Actor, MemoryActor, count_params
from training.model import ActorCritic

REPO = Path(__file__).resolve().parents[1]


def blank_x(dist=0.5, hit=1, keys=None, key_entity=False):
    x = {"rays": np.zeros((7, 64), np.float32), "ent_type": np.zeros(32, np.int64),
         "ent_feat": np.zeros((32, 5), np.float32), "ent_mask": np.zeros(32, bool),
         "scalars": np.zeros(28, np.float32)}
    x["rays"][0] = dist
    x["rays"][1 + hit] = 1.0
    if keys is not None:
        x["scalars"][8:14] = keys
    if key_entity:
        x["ent_type"][0], x["ent_mask"][0] = 9, True
    return x


def with_central(x, hit):
    x = {k: v.copy() for k, v in x.items()}
    for r in (62, 63, 0, 1, 2):
        x["rays"][1:, r] = 0
        x["rays"][1 + hit, r] = 1
    return x


# ------------------------------------------------------------------ write rules
def test_recent_slots_every_8_decisions():
    b = MemoryBook()
    for t in range(80):
        b.write(blank_x(), t)
    recent = [r.t for r in b.slots[:N_RECENT]]
    assert sorted(recent) == [16, 24, 32, 40, 48, 56, 64, 72]
    assert all(r is None for r in b.slots[N_RECENT:])


def test_events_written_when_condition_starts_with_dedup():
    b = MemoryBook()
    door = with_central(blank_x(), 2)
    writes = []
    for t in range(1, 60):                 # t=0 never used, so no recent write at t=0
        x = door if (t % 10) < 5 else blank_x()   # door appears at t = 10, 20, 30, ...
        new = b.write(x, t)
        writes += [(t, r.reason) for _, r in new if r.reason != "recent"]
    # door starts at t=1,10,20,30,40,50; dedup 16 keeps 1, 20, 40
    assert [t for t, r in writes if r == "door"] == [1, 20, 40]


def test_locked_door_key_and_inventory_events():
    b = MemoryBook()
    b.write(blank_x(), 1)
    new = b.write(with_central(blank_x(), 3), 2)
    assert [r.reason for _, r in new] == ["locked_door"]
    new = b.write(blank_x(key_entity=True), 3)
    assert [r.reason for _, r in new] == ["key_seen"]
    k = np.zeros(6, np.float32)
    k[2] = 1
    new = b.write(blank_x(keys=k), 4)
    assert [r.reason for _, r in new] == ["inventory"]


def test_turning_alone_is_not_a_new_area():
    b = MemoryBook()
    x = blank_x()
    x["rays"][0] = np.linspace(0.1, 1.0, 64)
    b.write(x, 1)
    y = {k: v.copy() for k, v in x.items()}
    y["rays"][0] = np.roll(x["rays"][0], 9)          # same room, rotated view
    assert not [r for _, r in b.write(y, 2) if r.reason == "new_area"]
    z = {k: v.copy() for k, v in x.items()}
    z["rays"][0] = 0.9                               # very different profile
    assert [r.reason for _, r in b.write(z, 3)] == ["new_area"]


def test_eviction_rule():
    b = MemoryBook()
    t = 1
    for _ in range(8):                  # fill all 8 event slots with 'door' (dedup: 16 apart)
        b.write(blank_x(), t)
        b.write(with_central(blank_x(), 2), t + 1)
        t += 20
    ev = b.slots[N_RECENT:]
    assert all(r is not None and r.reason == "door" for r in ev)
    oldest = min(r.t for r in ev)
    b.write(blank_x(), t)
    b.write(with_central(blank_x(), 3), t + 1)       # locked_door holds < 2 -> evict oldest of 'door'
    ev = b.slots[N_RECENT:]
    assert sum(r.reason == "locked_door" for r in ev) == 1 and oldest not in [r.t for r in ev]


# --------------------------------------------------------------- network / gate
def test_parameter_counts():
    m = ActorCritic(MemoryActor())
    assert count_params(m) == 921_259 and count_params(m.actor) == 885_929


def real_stream(n_steps=120, map_spec=("rooms", "train", 0)):
    """Adapter outputs from a real rooms map (doors in view, records get written)."""
    from doom_env import make_env
    from training.maps.gen import load_index

    e = load_index(map_spec[1], map_spec[0])[map_spec[2]]
    env = make_env(e["wad"], e["map"], episode_ticks=4 * n_steps, render_buffers=False)
    ad, xs = A.ObsAdapter(), []
    rng = np.random.default_rng(0)
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(n_steps):
            x = ad.observe(obs)
            a = rng.integers(0, 2, 9)
            a[0] = 1
            ad.record_action(a)
            xs.append(x)
            obs, _, te, tr, _ = env.step(a)
            if te or tr:
                break
    finally:
        env.close()
    return xs


@pytest.fixture(scope="module")
def stream():
    gen = REPO / "maps" / "generated" / "train" / "index_rooms.json"
    if not gen.exists():
        from training.maps.gen import write_pool
        write_pool("rooms", "train", 5)
    return real_stream()


def run_agent_path(actor: MemoryActor, xs: list[dict]) -> list[torch.Tensor]:
    """DoomAgent-style inference: book + per-slot cache of static encodings, ages at read."""
    book = MemoryBook()
    cache = torch.zeros(N_SLOTS, actor.REC_DIM)
    h = actor.initial_state(1)
    out = []
    with torch.no_grad():
        for t, x in enumerate(xs):
            valid = torch.as_tensor(book.valid())[None]
            mem = {"s": cache[None] * valid[..., None], "age": torch.as_tensor(book.ages(t))[None], "valid": valid}
            xt = {k: torch.from_numpy(v[None]) for k, v in x.items()}
            logits, h, (ray_f, ent_f) = actor.step(xt, h, torch.tensor([1.0 if t == 0 else 0.0]), mem)
            out.append(logits[0])
            for slot, rec in book.write(x, t):
                cache[slot] = actor.record_static(ray_f, ent_f, xt["scalars"][:, 8:14])[0]
    return out


def test_records_get_written_on_real_maps(stream):
    book = MemoryBook()
    reasons = []
    for t, x in enumerate(stream):
        reasons += [r.reason for _, r in book.write(x, t)]
    assert reasons.count("recent") == (len(stream) + 7) // 8
    assert len(set(reasons) - {"recent"}) >= 1, reasons


def test_cache_matches_fresh_encoding(stream):
    """Cached static encodings (from the acting pass) equal re-encoding the stored records."""
    torch.manual_seed(0)
    actor = MemoryActor().eval()
    book, cached = MemoryBook(), {}
    with torch.no_grad():
        for t, x in enumerate(stream):
            xt = {k: torch.from_numpy(v[None]) for k, v in x.items()}
            _, _, (ray_f, ent_f) = actor.step(xt, actor.initial_state(1), torch.ones(1))
            for slot, rec in book.write(x, t):
                cached[id(rec)] = (rec, actor.record_static(ray_f, ent_f, xt["scalars"][:, 8:14])[0])
        recs = [r for r, _ in cached.values()]
        fresh = actor.encode_records({k: torch.from_numpy(np.stack([r.x[k] for r in recs])) for k in A.INPUT_KEYS})
    for i, (_, s) in enumerate(cached.values()):
        assert torch.allclose(s, fresh[i], atol=1e-6)


def test_reset_isolation(stream):
    """Episode B after episode A gives exactly the outputs of B on a fresh agent: nothing from A
    reaches the query, the retrieved context or the GRU state."""
    torch.manual_seed(1)
    actor = MemoryActor().eval()
    a_xs, b_xs = stream[:60], stream[60:]
    fresh = run_agent_path(actor, b_xs)
    # same agent object, episode A first, then a reset (new book, zero state) and B
    run_agent_path(actor, a_xs)
    after = run_agent_path(actor, b_xs)
    for x, y in zip(fresh, after):
        assert torch.equal(x, y)


def test_warm_start_reproduces_baseline(stream):
    torch.manual_seed(2)
    base = Actor().eval()
    mem = MemoryActor().eval()
    mem.load_baseline(base.state_dict())
    h_b = base.initial_state(1)
    out_m = run_agent_path(mem, stream)
    with torch.no_grad():
        for t, x in enumerate(stream):
            xt = {k: torch.from_numpy(v[None]) for k, v in x.items()}
            lb, h_b = base.step(xt, h_b, torch.tensor([1.0 if t == 0 else 0.0]))
            assert torch.allclose(lb[0], out_m[t], atol=1e-5), t


def test_ppo_recompute_with_memory():
    """Recompute: same observations, hidden states and memory contents reproduce the stored
    log-probs and values; first minibatch ratio 1; episode starts see an empty memory."""
    from training.maps.gen import load_index
    from training.ppo import PPO, PPOConfig
    from training.vec_env import DoomVecEnv

    maps = load_index("train", "rooms")[:4]
    for e in maps:
        e["ticks"] = 240                       # short episodes: resets inside the rollout
    torch.manual_seed(0)
    with DoomVecEnv(maps, n_envs=3, seed=0) as env:
        ppo = PPO(ActorCritic(MemoryActor()), env,
                  PPOConfig(n_envs=3, n_steps=96, chunk=32, epochs=1, n_minibatches=3, total_steps=10**9),
                  DoomVecEnv.obs_spec)
        for _ in range(2):
            ppo.collect()
            assert np.all(ppo.mem.ids[ppo.buf.ep_start] == -1)           # reset before reading
            assert (ppo.mem.ids >= 0).any()                                # memory is actually used
            r = ppo.recompute_check()
            assert r["ok"], r
            u = ppo.update()
        assert u["n_minibatch_steps"] > 0


def test_doomagent_memory_matches_training_path(stream, tmp_path, monkeypatch):
    """The submitted DoomAgent (memory variant) and the learner's RolloutMemory acting path give
    identical logits on the same observation stream."""
    from doom_env import action_space, observation_space
    from test_agent import load_agent_module
    from training.memory_rollout import RolloutMemory

    torch.manual_seed(5)
    model = ActorCritic(MemoryActor()).eval()
    w = tmp_path / "actor.pt"
    torch.save({"state_dict": model.actor.state_dict(), "meta": {"greedy": True, "arch": "memory"}}, w)
    mod = load_agent_module()
    monkeypatch.setattr(mod, "WEIGHTS_FILE", w)
    agent = mod.DoomAgent(observation_space(), action_space(), "cpu")
    assert agent.memory

    rm = RolloutMemory(1, len(stream), torch.device("cpu"))
    rm.begin_rollout(model.actor)
    h = model.initial_state(1)
    agent.reset()
    # the agent consumes raw observations; feed it the adapter-equivalent by bypassing observe()
    with torch.no_grad():
        for t, x in enumerate(stream):
            start = np.array([t == 0])
            rm.reset_envs(start)
            mem = rm.read(t)
            xt = {k: torch.from_numpy(v[None]) for k, v in x.items()}
            xt["priv"] = torch.zeros(1, 16)          # critic input; irrelevant to the policy logits
            logits, _, _, h = model(xt, h, torch.from_numpy(start).float(), mem=mem)
            rm.write(model.actor, {k: v[None] for k, v in x.items()}, model.last_parts)

            monkeypatch.setattr(agent.adapter, "observe", lambda _obs, x=x: x)
            agent.act(None)
            assert torch.equal(agent.last_logits, logits), t


def _agent_with(tmp_path, monkeypatch, ablation):
    from doom_env import action_space, observation_space
    from test_agent import load_agent_module

    torch.manual_seed(7)
    actor = MemoryActor()
    w = tmp_path / "actor.pt"
    torch.save({"state_dict": actor.state_dict(), "meta": {"greedy": True, "arch": "memory"}}, w)
    mod = load_agent_module()
    monkeypatch.setattr(mod, "WEIGHTS_FILE", w)
    monkeypatch.setattr(mod.DoomAgent, "ablation", ablation)
    return mod.DoomAgent(observation_space(), action_space(), "cpu")


def _logits(agent, xs, monkeypatch):
    out = []
    agent.reset()
    for x in xs:
        monkeypatch.setattr(agent.adapter, "observe", lambda _o, x=x: x)
        agent.act(None)
        out.append(agent.last_logits.clone())
    return out


def test_ablations_change_what_is_read(stream, tmp_path, monkeypatch):
    xs = stream[:40]
    base = _logits(_agent_with(tmp_path, monkeypatch, None), xs, monkeypatch)
    off_agent = _agent_with(tmp_path, monkeypatch, "memory_off")
    off = _logits(off_agent, xs, monkeypatch)
    assert torch.equal(off[0], base[0])                       # step 0: memory empty either way
    assert not torch.equal(off[-1], base[-1])                 # later: records exist and are hidden
    assert off_agent.book.valid().any()                       # bookkeeping still runs

    nq = _logits(_agent_with(tmp_path, monkeypatch, "no_inv_query"), xs, monkeypatch)
    xs_key = [{k: v.copy() for k, v in x.items()} for x in xs]
    for x in xs_key:
        x["scalars"][8] = 1.0                                 # a key flag held: query input differs
    nq_key = _logits(_agent_with(tmp_path, monkeypatch, "no_inv_query"), xs_key, monkeypatch)
    base_key = _logits(_agent_with(tmp_path, monkeypatch, None), xs_key, monkeypatch)
    assert not torch.equal(base_key[-1], base[-1])
    assert nq and nq_key


def test_clear_at_key_and_foreign(stream, tmp_path, monkeypatch):
    xs = [{k: v.copy() for k, v in x.items()} for x in stream[:40]]
    for x in xs[20:]:
        x["scalars"][8] = 1.0                                 # key picked up at decision 20
    agent = _agent_with(tmp_path, monkeypatch, "clear_at_key")
    _logits(agent, xs, monkeypatch)
    assert all(r is None or r.t >= 20 for r in agent.book.slots)

    fa = _agent_with(tmp_path, monkeypatch, "foreign")
    first = _logits(fa, stream[:40], monkeypatch)              # episode 1 fills the donor book
    donor_valid = fa.book.valid().copy()
    second = _logits(fa, stream[40:80], monkeypatch)           # episode 2 reads episode 1's records
    assert donor_valid.any() and fa.donor[0].valid().tolist() == donor_valid.tolist()
    fresh = _logits(_agent_with(tmp_path, monkeypatch, None), stream[40:80], monkeypatch)
    assert not torch.equal(second[0], fresh[0])                # from the first decision, memory is foreign
    assert first
