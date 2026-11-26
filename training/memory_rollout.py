"""Episodic memory on the learner side of PPO (Spec v2.1, Memory: Storage / Training details).

The learner owns one MemoryBook per env (the same class DoomAgent uses) and feeds it the
adapter outputs it already receives, so the env workers are unchanged.

Storage (F3, F5). Per rollout and env, a RECORD TABLE lists every record any step of the
rollout can see: the records alive at the rollout start plus every record written during
it (evicted records keep their row). Each transition stores a SNAPSHOT of its 16 slot ->
table-row ids and ages, never a pointer to something that changes later. The table holds
the records' observations (fp32 / ints, F7), so they can be re-encoded with new weights.

Caching (F2). During collection the weights are fixed, so each record's static encoding s_i
is computed once, when it is written (from the encoder features of that very decision, no
extra pass), and kept in a per-slot device cache; at every rollout start the live records
are re-encoded with the updated weights. Ages are added at every read, never cached.

PPO updates (F5). Per minibatch the whole table (N envs x R rows) is encoded once and
gathered wherever used: ~N*R encodings instead of positions x 16.
"""
from __future__ import annotations

import numpy as np
import torch

from submission.doomnet import obs_adapter as A
from submission.doomnet.memory import N_SLOTS, MemoryBook


class RolloutMemory:
    def __init__(self, n_envs: int, n_steps: int, device):
        self.N, self.T, self.device = n_envs, n_steps, device
        self.books = [MemoryBook() for _ in range(n_envs)]
        self.ep_t = np.zeros(n_envs, np.int64)
        self.ids = np.full((n_steps, n_envs, N_SLOTS), -1, np.int64)     # snapshot per transition
        self.age = np.zeros((n_steps, n_envs, N_SLOTS), np.float32)
        self.slot_rid = np.full((n_envs, N_SLOTS), -1, np.int64)
        self.slot_t = np.zeros((n_envs, N_SLOTS), np.int64)
        self.s = None                                                     # (N, 16, 128) device cache
        self.tables: list[list] = [[] for _ in range(n_envs)]

    # ------------------------------------------------------------- lifecycle
    @torch.no_grad()
    def begin_rollout(self, actor) -> None:
        """New table per env from the live records; re-encode them with the current weights."""
        self.tables = [[] for _ in range(self.N)]
        self.s = torch.zeros(self.N, N_SLOTS, actor.REC_DIM, device=self.device)
        self.slot_rid[:] = -1
        live = []
        for n, book in enumerate(self.books):
            for j, rec in enumerate(book.slots):
                if rec is None:
                    continue
                rec.rid = len(self.tables[n])
                self.tables[n].append(rec)
                self.slot_rid[n, j], self.slot_t[n, j] = rec.rid, rec.t
                live.append((n, j, rec))
        if live:
            x = {k: torch.from_numpy(np.stack([r.x[k] for _, _, r in live])).to(self.device) for k in A.INPUT_KEYS}
            s = actor.encode_records(x)
            for i, (n, j, rec) in enumerate(live):
                self.s[n, j] = s[i]

    def reset_envs(self, starts: np.ndarray) -> None:
        """Episode starts: clear the memory BEFORE the first read of the new episode (F4)."""
        for n in np.nonzero(starts)[0]:
            self.books[n].reset()
            self.ep_t[n] = 0
            self.slot_rid[n] = -1
            if self.s is not None:
                self.s[n] = 0.0

    def current(self, idx=None) -> dict:
        """Memory tensors for the current decision of envs idx (all by default)."""
        idx = np.arange(self.N) if idx is None else np.asarray(idx)
        valid = self.slot_rid[idx] >= 0
        age = np.where(valid, (self.ep_t[idx, None] - self.slot_t[idx]) / A.MAX_DECISIONS, 0.0).astype(np.float32)
        t = torch.as_tensor
        return {"s": self.s[torch.as_tensor(idx, device=self.device)], "age": t(age, device=self.device),
                "valid": t(valid, device=self.device)}, valid, age

    def read(self, t: int) -> dict:
        """Memory for the current decision of every env; snapshots ids/ages into the buffer."""
        mem, valid, age = self.current()
        self.ids[t] = np.where(valid, self.slot_rid, -1)
        self.age[t] = age
        return mem

    @torch.no_grad()
    def write(self, actor, obs: dict, parts) -> None:
        """After acting on obs (dict of [N, ...] numpy): store records; cache their static encodings."""
        ray_f, ent_f = parts
        writes = []
        for n, book in enumerate(self.books):
            x = {k: obs[k][n] for k in A.INPUT_KEYS}
            for slot, rec in book.write(x, int(self.ep_t[n])):
                rec.rid = len(self.tables[n])
                self.tables[n].append(rec)
                self.slot_rid[n, slot], self.slot_t[n, slot] = rec.rid, rec.t
                writes.append((n, slot))
        if writes:
            ns = torch.as_tensor([n for n, _ in writes], device=self.device)
            keys = torch.from_numpy(np.stack([obs["scalars"][n, 8:14] for n, _ in writes])).to(self.device)
            s = actor.record_static(ray_f[ns], ent_f[ns], keys)
            for i, (n, slot) in enumerate(writes):
                self.s[n, slot] = s[i]
        self.ep_t += 1

    # ---------------------------------------------------------------- update
    def table_tensors(self) -> dict:
        """Padded record tables {key: (N, R, ...)} on the learner's device."""
        R = max(1, max(len(t) for t in self.tables))
        out = {}
        for k in A.INPUT_KEYS:
            sample = next((t[0].x[k] for t in self.tables if t), None)
            if sample is None:
                from training.vec_env import OBS_SPEC
                shape, dt = OBS_SPEC[k]
                sample = np.zeros(shape, dt)
            arr = np.zeros((self.N, R) + sample.shape, sample.dtype)
            for n, tab in enumerate(self.tables):
                for r, rec in enumerate(tab):
                    arr[n, r] = rec.x[k]
            out[k] = torch.from_numpy(arr).to(self.device)
        return out

    @staticmethod
    def gather(actor, table: dict, seq_env, ids, age) -> dict:
        """Encode the table once (F5) and gather per position: ids (T, B, 16) -> s (T, B, 16, 128)."""
        N, R = table["scalars"].shape[:2]
        flat = {k: v.reshape((N * R,) + v.shape[2:]) for k, v in table.items()}
        s_tab = actor.encode_records(flat).view(N, R, -1)
        valid = ids >= 0
        idc = ids.clamp(min=0)
        env = seq_env.view(1, -1, 1).expand_as(idc)
        s = s_tab[env, idc] * valid.unsqueeze(-1)
        return {"s": s, "age": age, "valid": valid}
