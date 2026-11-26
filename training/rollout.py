"""Rollout buffer, GAE, aux targets, reward scaling (ported from v1, adapted to Spec v2.1).

Episode endings (spec F1). The vector env reports two flags per step:
    term  = the episode is over for scoring: exit, death OR the map's tick cap
            (the wrapper returns truncated=True for the cap, but the scored attempt is
            over and the returned observation is stale). No bootstrap.
    trunc = OUR cutoff (harness stopped the episode early). Bootstrap from V(final obs),
            a genuine observation, computed with the pre-reset hidden state.
A rollout ending mid-episode bootstraps from V(next obs) as usual.

Layout: every field is [T steps, N envs, ...]. Training sequences are 32-step chunks; the
hidden state stored per chunk is the state immediately before the chunk's first observation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def compute_gae(rewards: np.ndarray, values: np.ndarray, last_value: np.ndarray,
                final_values: np.ndarray, term: np.ndarray, trunc: np.ndarray,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """All arrays [T, N] except last_value [N].

    delta_t = r_t + gamma (1 - term_t) V_{t+1} - V_t
    A_t     = delta_t + gamma lam (1 - done_t) A_{t+1},   done = term | trunc
    On trunc (our cutoff) V_{t+1} is V(final obs); it is never folded into the reward.
    """
    T = rewards.shape[0]
    term = term.astype(np.float64)
    done = np.maximum(term, trunc.astype(np.float64))
    next_values = np.concatenate([values[1:], last_value[None]], 0)
    next_values = np.where(trunc, final_values, next_values)
    deltas = rewards + gamma * (1.0 - term) * next_values - values
    adv = np.zeros_like(values, dtype=np.float64)
    running = np.zeros(values.shape[1], dtype=np.float64)
    for t in reversed(range(T)):
        running = deltas[t] + gamma * lam * (1.0 - done[t]) * running
        adv[t] = running
    returns = adv + values
    return adv.astype(np.float32), returns.astype(np.float32)


def damage_targets(events: np.ndarray, done: np.ndarray, horizon: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Aux 'damage or death within the next 8 decisions' on the ORDERED rollout.

    events[t]: damage/death happened during transition t. Positive as soon as an event
    occurs within the horizon (inside the episode); negative only if the full horizon is
    observed inside the episode with none; otherwise masked. Returns (target, valid) [T, N].
    """
    T, N = events.shape
    target = np.zeros((T, N), np.float32)
    valid = np.zeros((T, N), bool)
    for t in range(T):
        alive = np.ones(N, bool)
        hit = np.zeros(N, bool)
        for k in range(horizon):
            s = t + k
            if s >= T:
                break
            hit |= alive & events[s]
            alive &= ~done[s]
        full = t + horizon <= T
        ended_early = np.zeros(N, bool)
        for k in range(horizon - 1):
            if t + k < T:
                ended_early |= done[t + k]
        target[t] = hit
        valid[t] = hit | (full & ~ended_early)
    return target, valid


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray):
        bm, bv, bc = float(x.mean()), float(x.var()), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class RewardScaler:
    """Divide rewards by a running std of discounted returns, updated once per rollout."""

    def __init__(self, n_envs: int, gamma: float):
        self.gamma = gamma
        self.ret = np.zeros(n_envs, np.float64)
        self.rms = RunningMeanStd()

    def process_rollout(self, rewards: np.ndarray, done: np.ndarray) -> np.ndarray:
        rets = np.empty_like(rewards, dtype=np.float64)
        for t in range(rewards.shape[0]):
            self.ret = self.ret * self.gamma + rewards[t]
            rets[t] = self.ret
            self.ret[done[t]] = 0.0
        self.rms.update(rets)
        return (rewards / np.sqrt(self.rms.var + 1e-8)).astype(np.float32)

    @property
    def scale(self) -> float:
        return float(np.sqrt(self.rms.var + 1e-8))


class RolloutBuffer:
    def __init__(self, obs_spec: dict, n_steps: int, n_envs: int, n_buttons: int,
                 hidden: int, chunk: int, n_aux: int = 1, device="cpu"):
        assert n_steps % chunk == 0
        self.T, self.N, self.chunk, self.device = n_steps, n_envs, chunk, device
        T, N = n_steps, n_envs
        self.obs = {k: np.zeros((T, N) + tuple(s), d) for k, (s, d) in obs_spec.items()}
        self.actions = np.zeros((T, N, n_buttons), np.float32)
        self.logp = np.zeros((T, N), np.float32)
        self.values = np.zeros((T, N), np.float32)
        self.final_values = np.zeros((T, N), np.float32)
        self.rewards = np.zeros((T, N), np.float32)
        self.term = np.zeros((T, N), bool)
        self.trunc = np.zeros((T, N), bool)
        self.ep_start = np.zeros((T, N), bool)
        self.aux_events = np.zeros((T, N, n_aux), bool)
        self.h0 = np.zeros((T // chunk, N, hidden), np.float32)

    def chunked(self, adv: np.ndarray, returns: np.ndarray, aux_t: np.ndarray,
                aux_v: np.ndarray) -> dict[str, Tensor]:
        """[T, N, ...] -> [chunk, S, ...] with S = (T/chunk) * N sequences (s = c * N + n).
        Built AFTER targets were computed on the ordered rollout."""
        C, L, N = self.T // self.chunk, self.chunk, self.N

        def ch(x):
            x = np.asarray(x)
            x = x.reshape((C, L, N) + x.shape[2:])
            x = np.swapaxes(x, 1, 2).reshape((C * N, L) + x.shape[3:])
            return torch.from_numpy(np.ascontiguousarray(np.swapaxes(x, 0, 1))).to(self.device)

        out = {f"obs.{k}": ch(v) for k, v in self.obs.items()}
        out.update(actions=ch(self.actions), logp=ch(self.logp), values=ch(self.values),
                   adv=ch(adv), returns=ch(returns), ep_start=ch(self.ep_start),
                   aux_t=ch(aux_t), aux_v=ch(aux_v))
        out["h0"] = torch.from_numpy(self.h0.reshape(C * N, -1)).to(self.device)
        return out
