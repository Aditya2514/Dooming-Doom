"""Toy vector envs + actor for the PPO correctness tests and ladder (same contract as
training/vec_env.py, button actions instead of the DOOM observation)."""
from __future__ import annotations

import numpy as np
import torch.nn as nn

from submission.doomnet.model import RecurrentActor

OBS_KEY = "vec"


class GymVecEnv:
    """Gymnasium envs with Discrete(2) actions exposed as ONE button (1 = action 1).
    Gymnasium truncation (time limit) -> trunc (our cutoff: bootstrap), termination -> term."""

    n_buttons = 1

    def __init__(self, env_id: str, n: int, seed: int = 0, **kw):
        import gymnasium as gym
        self.envs = [gym.make(env_id, **kw) for _ in range(n)]
        assert self.envs[0].action_space.n == 2
        self.n, self.seed = n, seed
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self._ret = np.zeros(n)
        self._len = np.zeros(n, int)

    @property
    def obs_spec(self):
        return {OBS_KEY: ((self.obs_dim,), np.dtype(np.float32))}

    def reset(self):
        obs = [e.reset(seed=self.seed + i)[0] for i, e in enumerate(self.envs)]
        self._ret[:] = 0
        self._len[:] = 0
        return {OBS_KEY: np.asarray(obs, np.float32)}

    def step(self, actions):
        obs, rews = np.zeros((self.n, self.obs_dim), np.float32), np.zeros(self.n, np.float32)
        term, trunc = np.zeros(self.n, bool), np.zeros(self.n, bool)
        final = np.zeros_like(obs)
        episodes = []
        for i, e in enumerate(self.envs):
            o, r, te, tr, _ = e.step(int(actions[i, 0]))
            self._ret[i] += r
            self._len[i] += 1
            rews[i], term[i], trunc[i] = r, te, tr and not te
            if te or tr:
                final[i] = o
                episodes.append({"return": self._ret[i], "length": int(self._len[i])})
                self._ret[i], self._len[i] = 0, 0
                o, _ = e.reset()
            obs[i] = o
        info = {"final_obs": {OBS_KEY: final}, "episodes": episodes, "aux_events": np.zeros((self.n, 1), bool)}
        return {OBS_KEY: obs}, rews, term, trunc, info


class MemoryCueVecEnv:
    """Cue at step `cue_step` (one of K symbols); reward +1 only for pressing exactly the
    matching button (one-hot over K buttons) `delay` steps later; the episode then ends.

    fixed=True: cue at step 0, query at step 12 (spec memory toy task).
    fixed=False: cue at a random step 0-40 (chunk-boundary variant).
    obs: [cue one-hot (K), query flag, cue flag].
    """

    def __init__(self, n: int, k: int = 4, delay: int = 12, fixed: bool = True, seed: int = 0, chunk: int = 32):
        self.n, self.k, self.delay, self.fixed, self.chunk = n, k, delay, fixed, chunk
        self.n_buttons = k
        self.rng = np.random.default_rng(seed)
        self.obs_dim = k + 2
        self.t = np.zeros(n, int)
        self.cue = np.zeros(n, int)
        self.cue_step = np.zeros(n, int)
        self.global_t = np.zeros(n, int)

    @property
    def obs_spec(self):
        return {OBS_KEY: ((self.obs_dim,), np.dtype(np.float32))}

    def _new(self, i):
        self.t[i] = 0
        self.cue[i] = self.rng.integers(self.k)
        self.cue_step[i] = 0 if self.fixed else self.rng.integers(0, 41)

    def _obs(self, i):
        o = np.zeros(self.obs_dim, np.float32)
        if self.t[i] == self.cue_step[i]:
            o[self.cue[i]] = 1.0
            o[self.k + 1] = 1.0
        if self.t[i] == self.cue_step[i] + self.delay:
            o[self.k] = 1.0
        return o

    def reset(self):
        for i in range(self.n):
            self._new(i)
        return {OBS_KEY: np.stack([self._obs(i) for i in range(self.n)])}

    def step(self, actions):
        rews, term = np.zeros(self.n, np.float32), np.zeros(self.n, bool)
        trunc = np.zeros(self.n, bool)
        obs = np.zeros((self.n, self.obs_dim), np.float32)
        episodes = []
        for i in range(self.n):
            self.global_t[i] += 1
            if self.t[i] == self.cue_step[i] + self.delay:
                target = np.zeros(self.k)
                target[self.cue[i]] = 1
                ok = bool(np.array_equal(np.asarray(actions[i]) != 0, target != 0))
                rews[i] = float(ok)
                term[i] = True
                q_global = self.global_t[i] - 1
                same = (q_global // self.chunk) == ((q_global - self.delay) // self.chunk)
                episodes.append({"return": float(ok), "length": int(self.t[i] + 1),
                                 "cue_step": int(self.cue_step[i]), "same_chunk": bool(same)})
                self._new(i)
            else:
                self.t[i] += 1
            obs[i] = self._obs(i)
        info = {"final_obs": {OBS_KEY: np.zeros_like(obs)}, "episodes": episodes,
                "aux_events": np.zeros((self.n, 1), bool)}
        return {OBS_KEY: obs}, rews, term, trunc, info


class ToyActor(RecurrentActor):
    """MLP encoder -> GRUCell -> button logits; same machinery as the DOOM Actor."""

    input_keys = (OBS_KEY,)

    def __init__(self, obs_dim: int, n_buttons: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self._build_core(hidden, n_buttons, policy_hidden=hidden)
        self._init_weights()

    def encode(self, x):
        return self.enc(x[OBS_KEY])


def toy_model(env, hidden: int = 64):
    from training.model import ActorCritic
    return ActorCritic(ToyActor(env.obs_dim, env.n_buttons, hidden), priv_dim=0, n_aux=1,
                       critic_hidden=hidden)
