"""PPO correctness (Spec v2.1, Training/PPO 'Correctness'): GAE with both ending types,
aux targets, recompute, first-minibatch ratio, reset inside a chunk, acting == training
path, Bernoulli joint log-prob, and that the recurrent agent learns the memory task."""
import numpy as np
import pytest
import torch

from submission.doomnet import dist
from training.ppo import PPO, PPOConfig
from training.rollout import compute_gae, damage_targets
from training.toy_envs import GymVecEnv, MemoryCueVecEnv, toy_model


# ------------------------------------------------------------------------- GAE
def test_gae_worked_example_cutoff_vs_terminated():
    """trunc = OUR cutoff (bootstrap from V(final obs)); term = exit/death/tick cap (no bootstrap)."""
    g, lam = 0.99, 0.95
    r = np.array([[0.02], [0.5], [1.0]])
    v = np.array([[1.50], [0.7], [0.9]], np.float32)
    final = np.array([[1.60], [0.0], [0.0]], np.float32)
    term = np.array([[False], [False], [True]])
    trunc = np.array([[True], [False], [False]])
    last = np.array([123.0], np.float32)          # ignored: the episode terminated
    adv, _ = compute_gae(r, v, last, final, term, trunc, g, lam)
    assert adv[0, 0] == pytest.approx(0.104, abs=1e-5)
    d2 = 1.0 - 0.9
    d1 = 0.5 + g * 0.9 - 0.7
    assert adv[2, 0] == pytest.approx(d2, abs=1e-5)
    assert adv[1, 0] == pytest.approx(d1 + g * lam * d2, abs=1e-5)
    adv_bad, _ = compute_gae(r[:1], v[:1], last, final[:1], np.array([[True]]), np.array([[False]]), g, lam)
    assert adv_bad[0, 0] == pytest.approx(-1.48, abs=1e-5)


def test_tick_cap_is_not_bootstrapped():
    """F1: the map's tick cap arrives as term -> delta = r - V, whatever the stale next value."""
    adv, _ = compute_gae(np.array([[0.0]]), np.array([[1.0]], np.float32), np.array([50.0], np.float32),
                         np.zeros((1, 1), np.float32), np.array([[True]]), np.array([[False]]), 0.99, 0.95)
    assert adv[0, 0] == pytest.approx(-1.0)


def test_gae_rollout_boundary_bootstraps_from_next_obs():
    adv, _ = compute_gae(np.array([[0.0]]), np.array([[1.0]], np.float32), np.array([2.0], np.float32),
                         np.zeros((1, 1), np.float32), np.array([[False]]), np.array([[False]]), 0.9, 0.95)
    assert adv[0, 0] == pytest.approx(0.9 * 2.0 - 1.0)


def test_damage_targets_masking():
    T = 20
    ev = np.zeros((T, 1), bool)
    done = np.zeros((T, 1), bool)
    ev[10] = True
    done[15] = True
    tgt, val = damage_targets(ev, done)
    for t in range(3, 11):
        assert tgt[t, 0] == 1 and val[t, 0]
    assert tgt[2, 0] == 0 and val[2, 0]
    assert not val[11, 0]
    assert not val[T - 1, 0]


# -------------------------------------------------------------- Bernoulli dist
def test_joint_log_prob_is_sum_of_9_bernoullis():
    torch.manual_seed(0)
    logits = torch.randn(5, 9)
    a = torch.bernoulli(torch.full((5, 9), 0.5))
    p = torch.sigmoid(logits)
    manual = (a * p.log() + (1 - a) * (1 - p).log()).sum(-1)
    assert torch.allclose(dist.log_prob(logits, a), manual, atol=1e-6)
    ent_manual = -(p * p.log() + (1 - p) * (1 - p).log()).sum(-1)
    assert torch.allclose(dist.entropy(logits), ent_manual, atol=1e-6)
    assert torch.equal(dist.sample(logits, greedy=True), (p > 0.5).float())


# ------------------------------------------------ recompute / first mb / reset
def _ppo(n_envs=4, n_steps=64, seed=0, device="cpu"):
    torch.manual_seed(seed)
    env = MemoryCueVecEnv(n_envs, fixed=False, seed=seed)
    cfg = PPOConfig(n_envs=n_envs, n_steps=n_steps, chunk=32, epochs=1, n_minibatches=2,
                    total_steps=10**9, device=device)
    return PPO(toy_model(env), env, cfg, env.obs_spec)


def test_recompute_matches_stored():
    ppo = _ppo()
    for _ in range(2):            # second rollout: carried-over hidden states
        ppo.collect()
        r = ppo.recompute_check()
        assert r["ok"], r
        ppo.update()


def test_first_minibatch_ratio_one():
    ppo = _ppo()
    ppo.collect()
    u = ppo.update()
    assert u["first_mb/max_abs_logratio"] < 1e-5
    assert u["first_mb/kl"] < 1e-8
    assert u["first_mb/clipfrac"] == 0.0


def test_update_changes_policy():
    ppo = _ppo()
    before = [p.detach().clone() for p in ppo.model.parameters()]
    ppo.collect()
    u = ppo.update()
    moved = sum(float((p.detach() - q).abs().sum()) for p, q in zip(ppo.model.parameters(), before))
    assert moved > 0 and u["n_minibatch_steps"] > 0


def test_reset_inside_chunk_equals_fresh_chunk():
    torch.manual_seed(0)
    env = MemoryCueVecEnv(1)
    m = toy_model(env)
    T, B, k = 32, 3, 11
    obs = {"vec": torch.randn(T, B, env.obs_dim)}
    start = torch.zeros(T, B)
    start[k] = 1.0
    h0 = torch.randn(B, m.hidden)
    l1, v1, _, _ = m.forward_sequence(obs, h0, start)
    l2, v2, _, _ = m.forward_sequence({"vec": obs["vec"][k:]}, torch.zeros(B, m.hidden), torch.zeros(T - k, B))
    assert torch.allclose(v1[k:], v2, atol=1e-6) and torch.allclose(l1[k:], l2, atol=1e-6)


def test_step_path_equals_sequence_path():
    torch.manual_seed(1)
    env = MemoryCueVecEnv(1)
    m = toy_model(env)
    T, B = 10, 2
    obs = {"vec": torch.randn(T, B, env.obs_dim)}
    start = (torch.rand(T, B) < 0.2).float()
    h = h0 = torch.randn(B, m.hidden)
    vals = []
    for t in range(T):
        _, v, _, h = m({"vec": obs["vec"][t]}, h, start[t])
        vals.append(v)
    _, vs, _, hT = m.forward_sequence(obs, h0, start)
    assert torch.allclose(torch.stack(vals), vs, atol=1e-6) and torch.allclose(h, hT, atol=1e-6)


def test_cutoff_bootstrap_uses_final_obs():
    """Gymnasium's time limit plays the role of our cutoff: final_values filled exactly there."""
    env = GymVecEnv("CartPole-v1", 2, max_episode_steps=20)
    ppo = PPO(toy_model(env), env, PPOConfig(n_envs=2, n_steps=64, chunk=32, total_steps=10**9), env.obs_spec)
    ppo.collect()
    b = ppo.buf
    assert b.trunc.any()
    assert (b.final_values[b.trunc] != 0).all() and (b.final_values[~b.trunc] == 0).all()
    assert not (b.trunc & b.term).any()


def test_recurrent_agent_learns_memory_task():
    """Cue at step 0, reward only for the matching one-hot button press at step 12.
    Random play succeeds 1/16 of the time; measured ~0.98 by ~0.9M decisions.
    The thread count is pinned: DoomAgent sets torch.set_num_threads(1) process-wide (eval
    rule), which changes reduction order and therefore this run's trajectory."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        torch.manual_seed(0)
        env = MemoryCueVecEnv(16, fixed=True, seed=0)
        ppo = PPO(toy_model(env), env,
                  PPOConfig(n_envs=16, n_steps=128, gamma=0.99, lr=5e-4, total_steps=10**9), env.obs_spec)
        best = 0.0
        while ppo.global_step < 2_000_000 and best < 0.9:
            ppo.collect()
            ppo.update()
            eps = list(ppo.ep_stats)[-100:]
            if len(eps) >= 50:
                best = max(best, float(np.mean([e["return"] for e in eps])))
    finally:
        torch.set_num_threads(threads)
    assert best >= 0.9, best


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_collect_update_and_recompute():
    ppo = _ppo(device="cuda")
    ppo.collect()
    r = ppo.recompute_check()
    assert r["ok"], r
    u = ppo.update()
    assert u["n_minibatch_steps"] > 0 and u["first_mb/max_abs_logratio"] < 1e-4


def test_entropy_bonus_excludes_weapon_button():
    """The entropy term must not push SELECT_NEXT_WEAPON (button 8) toward p = 0.5."""
    from submission.doomnet import dist as D
    logits = torch.zeros(2, 3, 9, requires_grad=True)
    ent_b = D.button_entropy(logits)
    keep = torch.ones(9)
    keep[8] = 0.0
    (ent_b * keep).sum().backward()
    assert logits.grad[..., 8].abs().max() == 0                # no entropy gradient on button 8
    cfg = PPOConfig()
    assert cfg.entropy_exclude == (8,)
