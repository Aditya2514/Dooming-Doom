"""Recurrent PPO with GAE, written from scratch (Spec v2.1, Training / PPO). Ported from v1.

Vector-env contract (training/vec_env.py and the toy envs implement it):
    reset() -> obs                    dict of [N, ...] numpy arrays (may include "priv")
    step(actions [N, 9]) -> obs, reward [N], term [N], trunc [N], info
        term : episode over for scoring (exit, death, tick cap): no bootstrap (F1)
        trunc: our own cutoff: bootstrap from info["final_obs"] (genuine observation)
        info["aux_events"]: [N, 1] bool (damage or death during this step)
        info["episodes"]:   list of dicts for episodes that ended this step
    Auto-reset in the same step: the returned obs of a finished env is the reset observation.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from submission.doomnet import dist
from training.memory_rollout import RolloutMemory
from training.rollout import RewardScaler, RolloutBuffer, compute_gae, damage_targets


@dataclass
class PPOConfig:
    n_envs: int = 14
    n_steps: int = 256
    chunk: int = 32
    epochs: int = 4
    n_minibatches: int = 4
    lr: float = 2.5e-4
    lr_decay: bool = False           # spec v2.1: Adam 2.5e-4 (no decay stated)
    gamma: float = 0.99              # 0.99 in Stage 1 (combat), 0.995 elsewhere
    lam: float = 0.95
    clip: float = 0.2
    ent_start: float = 0.01          # entropy (sum over 9 buttons) 0.01 -> 0.003 over training
    ent_end: float = 0.003
    # Buttons excluded from the entropy bonus. SELECT_NEXT_WEAPON (8): where the weapon doesn't
    # matter to the reward (no enemies), the bonus alone pulls a button to p = 0.5; measured on the
    # 5060, presses climbed back from 0.02 to 0.42 per decision after the prior was set, leaving the
    # agent on fists half the time. Its randomness only does harm, so it gets no bonus.
    entropy_exclude: tuple = (8,)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    aux_coef: float = 0.1
    reward_scaling: bool = True
    total_steps: int = 40_000_000    # whole-run decision budget (drives the entropy schedule)
    seed: int = 0
    device: str = "cpu"
    extra: dict = field(default_factory=dict)


def _f(t) -> float:
    return float(t.detach()) if torch.is_tensor(t) else float(t)


def obs_to_torch(obs: dict, device) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in obs.items()}


class PPO:
    def __init__(self, model, envs, cfg: PPOConfig, obs_spec: dict, logger=None):
        self.model, self.envs, self.cfg = model, envs, cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        self.obs_spec = obs_spec
        n_aux = model.aux.out_features if model.aux is not None else 1
        self.buf = RolloutBuffer(obs_spec, cfg.n_steps, cfg.n_envs, model.n_buttons, model.hidden,
                                 cfg.chunk, n_aux=n_aux, device=self.device)
        self.scaler = RewardScaler(cfg.n_envs, cfg.gamma)
        self.logger = logger or (lambda d: None)
        self.global_step = 0
        self.n_updates = 0
        self.ep_stats = deque(maxlen=100)
        self.meta: dict = {}
        self.obs = envs.reset()
        self.h = model.initial_state(cfg.n_envs, self.device)
        self.ep_start = np.ones(cfg.n_envs, bool)
        self.mem = RolloutMemory(cfg.n_envs, cfg.n_steps, self.device) if model.has_memory else None

    # ------------------------------------------------------------------ helpers
    def _t(self, obs):
        return obs_to_torch(obs, self.device)

    def progress(self) -> float:
        return min(1.0, self.global_step / max(1, self.cfg.total_steps))

    def ent_coef(self) -> float:
        return self.cfg.ent_start + (self.cfg.ent_end - self.cfg.ent_start) * self.progress()

    def set_gamma(self, gamma: float):
        self.cfg.gamma = gamma
        self.scaler.gamma = gamma

    def reset_acting_state(self):
        self.obs = self.envs.reset()
        self.h = self.model.initial_state(self.cfg.n_envs, self.device)
        self.ep_start = np.ones(self.cfg.n_envs, bool)
        self.scaler.ret[:] = 0.0
        if self.mem is not None:
            self.mem = RolloutMemory(self.cfg.n_envs, self.cfg.n_steps, self.device)

    # --------------------------------------------------------------- collection
    @torch.no_grad()
    def collect(self) -> dict:
        b, cfg = self.buf, self.cfg
        self.model.eval()
        t0 = time.time()
        env_s = 0.0
        mem = self.mem
        if mem is not None:
            mem.begin_rollout(self.model.actor)
        for t in range(cfg.n_steps):
            for k in b.obs:
                b.obs[k][t] = self.obs[k]
            b.ep_start[t] = self.ep_start
            if t % cfg.chunk == 0:
                b.h0[t // cfg.chunk] = self.h.cpu().numpy()
            start_t = torch.from_numpy(self.ep_start).to(self.device)
            m_t = None
            if mem is not None:
                mem.reset_envs(self.ep_start)
                m_t = mem.read(t)
            actions, logp, value, _, h_new, _ = self.model.act(self._t(self.obs), self.h, start_t, mem=m_t)
            if mem is not None:
                mem.write(self.model.actor, self.obs, self.model.last_parts)
            act_np = actions.cpu().numpy()
            te = time.time()
            obs, rew, term, trunc, info = self.envs.step(act_np.astype(np.int8))
            env_s += time.time() - te
            b.actions[t], b.logp[t], b.values[t] = act_np, logp.cpu().numpy(), value.cpu().numpy()
            b.rewards[t], b.term[t], b.trunc[t] = rew, term, trunc
            b.aux_events[t] = info["aux_events"]
            b.final_values[t] = 0.0
            if trunc.any():
                idx = np.nonzero(trunc)[0]
                fo = {k: v[idx] for k, v in info["final_obs"].items()}
                m_f = mem.current(idx)[0] if mem is not None else None   # memory after writing o_t
                _, v_final, _, _ = self.model(self._t(fo), h_new[torch.as_tensor(idx, device=self.device)],
                                              torch.zeros(len(idx), device=self.device), mem=m_f)
                b.final_values[t, idx] = v_final.cpu().numpy()
            self.ep_stats.extend(info.get("episodes", []))
            self.obs, self.h = obs, h_new
            self.ep_start = term | trunc
        m_last = None
        if mem is not None:
            mem.reset_envs(self.ep_start)
            m_last = mem.current()[0]
            self.mem_table = mem.table_tensors()
        _, last_value, _, _ = self.model(self._t(self.obs), self.h,
                                         torch.from_numpy(self.ep_start).to(self.device), mem=m_last)
        self.last_value = last_value.cpu().numpy()
        self.global_step += cfg.n_steps * cfg.n_envs
        dt = time.time() - t0
        return {"collect_s": dt, "env_s": env_s, "collect_dps": cfg.n_steps * cfg.n_envs / dt}

    # ------------------------------------------------------------------ update
    def compute_targets(self):
        b, cfg = self.buf, self.cfg
        done = b.term | b.trunc
        rewards = self.scaler.process_rollout(b.rewards, done) if cfg.reward_scaling else b.rewards
        adv, ret = compute_gae(rewards, b.values, self.last_value, b.final_values, b.term, b.trunc,
                               cfg.gamma, cfg.lam)
        aux_t = np.zeros(b.aux_events.shape, np.float32)
        aux_v = np.zeros(b.aux_events.shape, bool)
        for j in range(b.aux_events.shape[-1]):
            aux_t[..., j], aux_v[..., j] = damage_targets(b.aux_events[..., j], done)
        return adv, ret, aux_t, aux_v

    def _chunk_data(self, adv, ret, aux_t, aux_v) -> dict:
        data = self.buf.chunked(adv, ret, aux_t, aux_v)
        if self.mem is not None:
            data.update(self.buf.chunk_arrays({"mem_ids": self.mem.ids, "mem_age": self.mem.age}))
            S = data["h0"].shape[0]
            data["seq_env"] = torch.arange(S, device=self.device) % self.cfg.n_envs
        return data

    def _mem_for(self, mb) -> dict | None:
        if self.mem is None:
            return None
        return RolloutMemory.gather(self.model.actor, self.mem_table, mb["seq_env"], mb["mem_ids"], mb["mem_age"])

    def _minibatch_loss(self, mb, ent_coef):
        cfg = self.cfg
        obs = {k[4:]: v for k, v in mb.items() if k.startswith("obs.")}
        logits, values, aux, _ = self.model.forward_sequence(obs, mb["h0"], mb["ep_start"], mem=self._mem_for(mb))
        logp = dist.log_prob(logits, mb["actions"])
        ent_b = dist.button_entropy(logits)
        if cfg.entropy_exclude and logits.shape[-1] > max(cfg.entropy_exclude):
            keep = torch.ones(logits.shape[-1], device=logits.device)
            keep[list(cfg.entropy_exclude)] = 0.0
            ent_b = ent_b * keep
        entropy = ent_b.sum(-1)
        log_ratio = logp - mb["logp"]
        ratio = log_ratio.exp()
        approx_kl = (ratio - 1 - log_ratio).mean()
        adv = mb["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        v_loss = F.mse_loss(values, mb["returns"])          # unclipped (spec F10)
        loss = pg + cfg.vf_coef * v_loss - ent_coef * entropy.mean()
        stats = {}
        if aux.shape[-1] > 0:
            valid = mb["aux_v"][..., 0]
            n = valid.sum()
            l = (F.binary_cross_entropy_with_logits(aux[..., 0][valid], mb["aux_t"][..., 0][valid])
                 if n > 0 else aux.sum() * 0.0)
            loss = loss + cfg.aux_coef * l
            stats["aux/damage_loss"] = _f(l)
            stats["aux/damage_pos_rate"] = float(mb["aux_t"][..., 0][valid].mean()) if n > 0 else 0.0
        stats.update({
            "loss/policy": _f(pg), "loss/value": _f(v_loss), "loss/entropy": _f(entropy.mean()),
            "kl": _f(approx_kl), "clipfrac": _f(((ratio - 1).abs() > cfg.clip).float().mean()),
            "max_abs_logratio": _f(log_ratio.abs().max()),
        })
        return loss, approx_kl, stats

    def update(self) -> dict:
        cfg = self.cfg
        t0 = time.time()
        self.model.train()
        adv, ret, aux_t, aux_v = self.compute_targets()
        data = self._chunk_data(adv, ret, aux_t, aux_v)
        S = data["h0"].shape[0]
        mb_size = S // cfg.n_minibatches
        if cfg.lr_decay:
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - self.progress())
        ent_coef = self.ent_coef()
        log, stopped, first = [], False, None
        for epoch in range(cfg.epochs):
            perm = torch.randperm(S)
            for i in range(cfg.n_minibatches):
                idx = perm[i * mb_size:(i + 1) * mb_size].to(self.device)
                mb = {k: (v[idx] if k in ("h0", "seq_env") else v[:, idx]) for k, v in data.items()}
                loss, kl, stats = self._minibatch_loss(mb, ent_coef)
                if first is None:
                    first = stats          # ratio must be 1 / KL 0 / clipfrac 0 here
                if _f(kl) > cfg.target_kl:
                    stopped = True
                    break
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats["grad_norm"] = _f(gn)
                log.append(stats)
            if stopped:
                break
        self.n_updates += 1
        out = {k: float(np.mean([s[k] for s in log])) for k in (log[0] if log else {})}
        y, v = ret.ravel(), self.buf.values.ravel()
        out.update({
            "first_mb/kl": first["kl"], "first_mb/max_abs_logratio": first["max_abs_logratio"],
            "first_mb/clipfrac": first["clipfrac"], "kl_early_stop": int(stopped),
            "n_minibatch_steps": len(log), "lr": self.opt.param_groups[0]["lr"],
            "ent_coef": ent_coef, "reward_scale": self.scaler.scale, "gamma": cfg.gamma,
            "explained_var": float(1 - np.var(y - v) / (np.var(y) + 1e-8)),
            "update_s": time.time() - t0,
        })
        return out

    # --------------------------------------------------------------- recompute
    @torch.no_grad()
    def recompute_check(self) -> dict:
        """Re-run stored chunks from stored start states: must reproduce stored log-probs and
        values (abs 1e-5 / rel 1e-4). Call after collect(), before update()."""
        self.model.eval()
        zeros = np.zeros((self.cfg.n_steps, self.cfg.n_envs), np.float32)
        za = np.zeros(self.buf.aux_events.shape, np.float32)
        data = self._chunk_data(zeros, zeros, za, za.astype(bool))
        obs = {k[4:]: v for k, v in data.items() if k.startswith("obs.")}
        logits, values, _, _ = self.model.forward_sequence(obs, data["h0"], data["ep_start"],
                                                           mem=self._mem_for(data))
        logp = dist.log_prob(logits, data["actions"])
        e_lp = (logp - data["logp"]).abs()
        e_v = (values - data["values"]).abs()
        ok_lp = bool((e_lp <= 1e-5 + 1e-4 * data["logp"].abs()).all())
        ok_v = bool((e_v <= 1e-5 + 1e-4 * data["values"].abs()).all())
        return {"max_err_logp": float(e_lp.max()), "max_err_value": float(e_v.max()), "ok": ok_lp and ok_v}

    # ------------------------------------------------------------- checkpoints
    def state_dict(self) -> dict:
        return {"meta": dict(self.meta), "model": self.model.state_dict(), "opt": self.opt.state_dict(),
                "reward_rms": self.scaler.rms.state_dict(), "cfg": asdict(self.cfg),
                "global_step": self.global_step, "n_updates": self.n_updates}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(self.state_dict(), tmp)
        tmp.replace(path)

    def load(self, path, reset_acting_state: bool = True):
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["opt"])
        self.scaler.rms.load_state_dict(d["reward_rms"])
        self.global_step, self.n_updates = d["global_step"], d["n_updates"]
        self.meta = dict(d.get("meta", {}))
        if reset_acting_state:
            self.reset_acting_state()
        return d

    # -------------------------------------------------------------------- loop
    def train(self, n_updates: int | None = None, stop_fn=None) -> list[dict]:
        history = []
        while n_updates is None or self.n_updates < n_updates:
            if self.global_step >= self.cfg.total_steps:
                break
            t0 = time.time()
            c = self.collect()
            u = self.update()
            row = {"step": self.global_step, "update": self.n_updates, **c, **u,
                   "dps": self.cfg.n_steps * self.cfg.n_envs / (time.time() - t0)}
            eps = list(self.ep_stats)
            if eps:
                row["ep_return"] = float(np.mean([e["return"] for e in eps]))
                row["ep_length"] = float(np.mean([e["length"] for e in eps]))
            history.append(row)
            self.logger(row)
            if stop_fn is not None and stop_fn(row):
                break
        return history
