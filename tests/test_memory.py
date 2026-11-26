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
         "scalars": np.zeros(28, np.float32), "pose": np.zeros(3, np.float32)}
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
def _wake_rel_enc(actor, std=0.1):
    """rel_enc's last layer is zero at init (old checkpoints load unchanged); tests that check
    the pose path give it weights."""
    with torch.no_grad():
        torch.nn.init.normal_(actor.rel_enc[2].weight, std=std)
        torch.nn.init.normal_(actor.rel_enc[2].bias, std=std)


def test_parameter_counts():
    m = ActorCritic(MemoryActor())
    assert count_params(m) == 930_091 and count_params(m.actor) == 894_761     # + rel_enc 8,832


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
        model = ActorCritic(MemoryActor())
        _wake_rel_enc(model.actor)               # the recompute must also reproduce the pose path
        ppo = PPO(model, env,
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
    _wake_rel_enc(model.actor)                   # so record positions change the logits
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


# ------------------------------------------------------------- pose-tagged memory
def _vitals(fwd=0.0, right=0.0):
    v = np.zeros(17, np.float32)
    v[14], v[15] = fwd, right
    return v


def test_pose_tracker_turn_model():
    """First decision never turns; a held turn is slow for its first 5 tics; both buttons cancel."""
    pt = A.PoseTracker()
    left = np.zeros(9)
    left[4] = 1
    both = left.copy()
    both[5] = 1
    yaws = [pt.update(_vitals(), None)[2]]
    for a in (left, left, left, left, np.zeros(9), left, both):
        yaws.append(pt.update(_vitals(), a)[2])
    d = np.degrees(np.diff(yaws))
    slow, fast = 320 / 65536 * 360, 640 / 65536 * 360
    np.testing.assert_allclose(d, [0, 4 * slow, slow + 3 * fast, 4 * fast, 0, 4 * slow, 0], atol=1e-4)


def test_pose_tracker_straight_line_and_frame():
    """Walking forward at 5 units/tic for 10 decisions moves 10 x 4.05 x 5 along the start heading;
    strafing right while facing +y moves towards +x."""
    pt = A.PoseTracker()
    pt.update(_vitals(5.0), None)
    for _ in range(10):
        p = pt.update(_vitals(5.0), np.zeros(9))
    np.testing.assert_allclose(p, [10 * (1.46 + 2.59) * 5, 0, 0], atol=1e-3)
    pt.yaw, pt.prev_v = np.pi / 2, (0.0, 5.0)              # facing +y, strafing right
    p2 = pt.update(_vitals(0.0, 5.0), np.zeros(9))
    np.testing.assert_allclose(p2[:2] - p[:2], [(1.46 + 2.59) * 5, 0], atol=1e-3)


def test_rel_features_geometry():
    pose = torch.tensor([[0.0, 0.0, 0.0]])
    rpose = torch.zeros(1, 16, 3)
    rpose[0, 0] = torch.tensor([100.0, 0.0, 0.0])          # 100 units straight ahead
    f = MemoryActor.rel_features(pose, rpose)
    assert f.shape == (1, 16, 7)
    np.testing.assert_allclose(f[0, 0].numpy(), [100 / 1024, 0, 100 / 1024, 1, 0, 1, 0], atol=1e-6)
    turned = torch.tensor([[0.0, 0.0, np.pi / 2]])         # turned left 90 deg: it is now on the right
    g = MemoryActor.rel_features(turned, rpose)[0, 0].numpy()
    np.testing.assert_allclose(g[:5], [0, -100 / 1024, 100 / 1024, 0, -1], atol=1e-5)
    np.testing.assert_allclose(g[5:], [0, -1], atol=1e-5)  # the record faced 90 deg to our right


def run_agent_path_pose(actor, xs):
    """run_agent_path with record poses (what DoomAgent does)."""
    book, cache, h, out = MemoryBook(), torch.zeros(N_SLOTS, actor.REC_DIM), actor.initial_state(1), []
    with torch.no_grad():
        for t, x in enumerate(xs):
            valid = torch.as_tensor(book.valid())[None]
            mem = {"s": cache[None] * valid[..., None], "age": torch.as_tensor(book.ages(t))[None], "valid": valid,
                   "rpose": torch.as_tensor(book.poses())[None]}
            xt = {k: torch.from_numpy(v[None]) for k, v in x.items()}
            logits, h, (ray_f, ent_f) = actor.step(xt, h, torch.tensor([1.0 if t == 0 else 0.0]), mem)
            out.append(logits[0])
            for slot, _rec in book.write(x, t):
                cache[slot] = actor.record_static(ray_f, ent_f, xt["scalars"][:, 8:14])[0]
    return out


def test_pre_pose_checkpoint_loads_with_identical_outputs(stream):
    """A memory checkpoint saved before rel_enc existed loads into the new MemoryActor and gives
    exactly the old outputs (rel_enc's last layer zero), even with poses present."""
    torch.manual_seed(11)
    old = MemoryActor().eval()
    sd = {k: v for k, v in old.state_dict().items() if not k.startswith("rel_enc.")}
    torch.manual_seed(12)
    new = MemoryActor().eval()
    _wake_rel_enc(new)
    new.load_state_dict(sd)
    assert torch.count_nonzero(new.rel_enc[2].weight) == 0
    a = run_agent_path(old, stream[:40])                   # old model: positions never read
    b = run_agent_path_pose(new, stream[:40])
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_record_positions_reach_the_logits(stream):
    torch.manual_seed(13)
    actor = MemoryActor().eval()
    _wake_rel_enc(actor)
    a = run_agent_path_pose(actor, stream[:40])
    moved = [{**x, "pose": x["pose"] + np.array([300, -200, 0.5], np.float32) * (t > 20)}
             for t, x in enumerate(stream[:40])]
    b = run_agent_path_pose(actor, moved)
    assert torch.equal(a[10], b[10]) and not torch.equal(a[-1], b[-1])


def test_optimizer_state_survives_adding_rel_enc():
    from training.ppo import _remap_opt
    torch.manual_seed(14)
    m = ActorCritic(MemoryActor())
    names = [n for n, _ in m.named_parameters()]
    new = {n for n in names if n.startswith("actor.rel_enc.")}
    old_params = [p for n, p in m.named_parameters() if n not in new]
    opt_old = torch.optim.Adam(old_params, lr=1e-3)
    for p in old_params:
        p.grad = torch.randn_like(p)
    opt_old.step()
    opt_new = torch.optim.Adam(m.parameters(), lr=1e-3)
    opt_new.load_state_dict(_remap_opt(opt_old.state_dict(), names, new))
    by_name = dict(m.named_parameters())
    for n in ("actor.fusion.0.weight", names[-1]):          # before and after rel_enc in the list
        p = by_name[n]
        assert torch.equal(opt_new.state[p]["exp_avg"], opt_old.state[p]["exp_avg"]), n
    assert all(len(opt_new.state[by_name[n]]) == 0 for n in new)


def test_pose_tracker_follows_the_engine():
    """Real engine, walking forward and turning at random: dead-reckoned heading and position
    stay close to the engine's own pose (in the episode's start frame). Catches sign / frame
    errors; pushing into walls (not done here) is the known source of position drift."""
    import math
    from doom_env import make_env
    from training.maps.gen import load_index

    e = load_index("train", "rooms")[1]
    env = make_env(e["wad"], e["map"], episode_ticks=4 * 150, render_buffers=False)
    ad, rng = A.ObsAdapter(), np.random.default_rng(3)
    errs, herr = [], []
    try:
        obs, _ = env.reset(seed=0)
        p0 = env.privileged_state()
        a0 = math.radians(p0["angle"])
        for _ in range(150):
            x = ad.observe(obs)
            ps = env.privileged_state()
            dx, dy = ps["x"] - p0["x"], ps["y"] - p0["y"]
            c, s = math.cos(-a0), math.sin(-a0)
            tx, ty = dx * c - dy * s, dx * s + dy * c
            errs.append(math.hypot(x["pose"][0] - tx, x["pose"][1] - ty))
            herr.append(abs((math.degrees(x["pose"][2]) - (ps["angle"] - p0["angle"]) + 180) % 360 - 180))
            a = (rng.random(9) < [0.9, 0, 0, 0, 0.15, 0.05, 0, 0, 0]).astype(np.int64)
            ad.record_action(a)
            obs, _, te, tr, _ = env.step(a)
            if te or tr:
                break
    finally:
        env.close()
    assert max(herr) < 4.0, max(herr)
    assert np.median(errs) < 48 and max(errs) < 120, (np.median(errs), max(errs))


def test_no_pose_ablation(stream, tmp_path, monkeypatch):
    """no_pose reads the same records without their positions: equal to the pose-free path,
    different from the full agent once records exist (rel_enc trained)."""
    from doom_env import action_space, observation_space
    from test_agent import load_agent_module

    torch.manual_seed(15)
    actor = MemoryActor()
    _wake_rel_enc(actor)
    w = tmp_path / "actor.pt"
    torch.save({"state_dict": actor.state_dict(), "meta": {"greedy": True, "arch": "memory"}}, w)
    mod = load_agent_module()
    monkeypatch.setattr(mod, "WEIGHTS_FILE", w)
    xs = stream[:40]
    monkeypatch.setattr(mod.DoomAgent, "ablation", None)
    full = _logits(mod.DoomAgent(observation_space(), action_space(), "cpu"), xs, monkeypatch)
    monkeypatch.setattr(mod.DoomAgent, "ablation", "no_pose")
    nopose = _logits(mod.DoomAgent(observation_space(), action_space(), "cpu"), xs, monkeypatch)
    ref = run_agent_path(actor.eval(), xs)
    assert torch.equal(nopose[-1][0], ref[-1])
    assert not torch.equal(full[-1], nopose[-1])
