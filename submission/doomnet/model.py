"""The acting network (Spec v2.1, Network). Used by DoomAgent at eval and by training.

Baseline acting path, 691,401 parameters:
    rays (7,64)      -> RayEncoder (circular Conv1d)        -> 128
    entities (32,..) -> EntityEncoder (attention + null)    -> 64
    scalars (28)     -> ScalarEncoder                       -> 64
    concat 256 -> Fusion (Linear + LayerNorm + ReLU)        -> 256
    GRUCell 256 -> policy MLP 256 -> 128 -> 9 Bernoulli logits

The critic and the auxiliary head are training-only and live in training/model.py.
Episode starts reset the hidden state BEFORE the step (h <- h * (1 - start)), identically
in step() (acting) and sequence() (training), so both share one code path.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from . import obs_adapter as A

# fp32 throughout (spec). On Ampere+ GPUs cuDNN/cuBLAS would otherwise use TF32 (~10-bit
# mantissa), which breaks the 1e-5 recompute test between acting and training paths.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

HIDDEN = 256
# Prior for SELECT_NEXT_WEAPON (button 8): start rarely pressed (p ~ 0.02) instead of 0.5. Each press
# toggles the weapon; at 0.5 an untrained policy spends half its time on fists (measured: a trained
# Stage 1 baseline held fists 52% of the time, pressed it on 35% of decisions, and kept most of
# its bullets). A learned bias, not a rule: training can still raise it (e.g. to pick a shotgun).
WEAPON_BUTTON, WEAPON_PRIOR_BIAS = 8, -4.0


def ortho_init(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class RayEncoder(nn.Module):
    """(B, 7, 64) -> 128. Circular padding: ray 63 is next to ray 0. 64 -> 64 -> 32 -> 16. 155,008 params."""

    def __init__(self):
        super().__init__()
        c = dict(padding_mode="circular")
        self.convs = nn.Sequential(
            nn.Conv1d(A.RAY_CHANNELS, 32, 5, padding=2, **c), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, **c), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1, **c), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * (A.NUM_RAYS // 4), 128), nn.ReLU())

    def forward(self, rays: Tensor) -> Tensor:
        return self.fc(self.convs(rays).flatten(1))


class EntityEncoder(nn.Module):
    """32 x (type, 5 features) -> 64. 39,424 params.

    Type embedding 12x16 . MLP 21->64->64 . masked 4-head self-attention + LayerNorm .
    learned query over [learned null token, entities] with 4-head attention pooling + LayerNorm.
    The null token is always a valid key, so a scene with no visible entity is finite.
    """

    def __init__(self, d: int = 64, heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(A.NUM_ENTITY_TYPES, 16)
        self.mlp = nn.Sequential(nn.Linear(16 + A.ENT_FEATURES, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.pool_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, ent_type: Tensor, ent_feat: Tensor, ent_mask: Tensor) -> Tensor:
        valid = ent_mask.bool()
        x = self.mlp(torch.cat([self.emb(ent_type), ent_feat], -1)) * valid.unsqueeze(-1)
        pad = ~valid
        # rows with no entity: let slot 0 attend in self-attention only (it stays excluded from
        # pooling) so softmax never sees an all -inf row
        sa_pad = pad.clone()
        sa_pad[:, 0] &= valid.any(1)
        a, _ = self.self_attn(x, x, x, key_padding_mask=sa_pad, need_weights=False)
        x = self.ln1(x + a)
        B = x.shape[0]
        keys = torch.cat([self.null.expand(B, 1, -1), x], 1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], 1)
        p, _ = self.pool_attn(self.query.expand(B, 1, -1), keys, keys, key_padding_mask=kpm, need_weights=False)
        return self.ln2(p.squeeze(1))


class ScalarEncoder(nn.Module):
    """28 -> 64, one Linear + ReLU. 1,856 params."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(A.NUM_SCALARS, 64), nn.ReLU())

    def forward(self, s: Tensor) -> Tensor:
        return self.fc(s)


class RecurrentActor(nn.Module):
    """Shared recurrent acting machinery. Subclasses implement `encode`."""

    hidden: int
    n_buttons: int

    def _build_core(self, hidden: int, n_buttons: int, policy_hidden: int = 128):
        self.hidden, self.n_buttons = hidden, n_buttons
        self.core = nn.GRUCell(hidden, hidden)
        self.policy = nn.Sequential(nn.Linear(hidden, policy_hidden), nn.ReLU(), nn.Linear(policy_hidden, n_buttons))

    def _init_weights(self):
        ortho_init(self, math.sqrt(2))
        nn.init.orthogonal_(self.policy[-1].weight, 0.01)
        if self.n_buttons > WEAPON_BUTTON:
            with torch.no_grad():
                self.policy[-1].bias[WEAPON_BUTTON] = WEAPON_PRIOR_BIAS
        nn.init.orthogonal_(self.core.weight_hh)
        nn.init.xavier_uniform_(self.core.weight_ih)
        nn.init.zeros_(self.core.bias_ih)
        nn.init.zeros_(self.core.bias_hh)

    def encode(self, x: dict[str, Tensor]) -> Tensor:     # (B, ...) -> (B, hidden)
        raise NotImplementedError

    def initial_state(self, n: int, device=None) -> Tensor:
        return torch.zeros(n, self.hidden, device=device)

    def core_step(self, e: Tensor, h: Tensor, start: Tensor) -> Tensor:
        h = h * (1.0 - start.float()).unsqueeze(-1)
        return self.core(e, h)

    def step(self, x: dict[str, Tensor], h: Tensor, start: Tensor) -> tuple[Tensor, Tensor]:
        """One decision for a batch. Returns (logits (B, 9), new hidden (B, H))."""
        h = self.core_step(self.encode(x), h, start)
        return self.policy(h), h

    def sequence(self, x: dict[str, Tensor], h0: Tensor, starts: Tensor):
        """x: (T, B, ...), h0: (B, H) state before step 0, starts: (T, B).
        Encoders run batched over all T*B positions, then a masked step-by-step GRU loop.
        Returns (logits (T, B, 9), hidden per step (T, B, H), last hidden (B, H))."""
        T, B = starts.shape
        flat = {k: v.reshape((T * B,) + v.shape[2:]) for k, v in x.items() if k in self.input_keys}
        e = self.encode(flat).view(T, B, -1)
        h, zs = h0, []
        for t in range(T):
            h = self.core_step(e[t], h, starts[t])
            zs.append(h)
        z = torch.stack(zs)
        return self.policy(z), z, h

    input_keys: tuple = A.INPUT_KEYS


class Actor(RecurrentActor):
    """GRU baseline acting network. 691,401 parameters."""

    def __init__(self):
        super().__init__()
        self.rays = RayEncoder()
        self.entities = EntityEncoder()
        self.scalars = ScalarEncoder()
        self.fusion = nn.Sequential(nn.Linear(128 + 64 + 64, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU())
        self._build_core(HIDDEN, A.NUM_BUTTONS)
        self._init_weights()

    def encode(self, x: dict[str, Tensor]) -> Tensor:
        z = torch.cat([
            self.rays(x["rays"]),
            self.entities(x["ent_type"], x["ent_feat"], x["ent_mask"]),
            self.scalars(x["scalars"]),
        ], -1)
        return self.fusion(z)


# ============================================================================ memory variant
class MemoryActor(Actor):
    """GRU + inventory-conditioned episodic memory (Spec v2.1, Memory). 885,929 parameters.

    Read order per decision (F4):
      1. h_{t-1} <- 0 at an episode start
      2. this episode's records (empty slots masked)
      3. c_t = [rays 128, entities 64, scalars 64]
      4. q_t = Linear([c_t, h_{t-1}, key flags 6]) -> 128
      5. e_i = s_i + w_age * age_i, with s_i = Linear([ray_enc(rays_i), ent_enc(ents_i),
         Linear(key flags_i) 32]) the cacheable static part (encoders shared with step 3)
      6. 4-head attention of q_t over [learned null record, e_1..e_16] -> m_t (128)
      7. fusion([c_t, m_t]) 384 -> 256 -> GRU
    The null record is always a valid key, so an empty memory is finite.
    """

    memory = True
    REC_DIM = 128
    # Evaluation-only instrumentation (Spec v2.1: ablations + demo overlay). Both OFF in the
    # submission: nothing in submission/ sets them.
    query_keys_off = False      # ablation: drop the explicit inventory input from the query
    debug_attention = False     # demo: keep the last attention weights in self.last_attention

    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(nn.Linear(256 + self.REC_DIM, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU())
        self.rec_keys = nn.Linear(6, 32)
        self.rec_proj = nn.Linear(128 + 64 + 32, self.REC_DIM)
        self.age_w = nn.Parameter(torch.zeros(self.REC_DIM))
        self.mem_query = nn.Linear(256 + HIDDEN + 6, self.REC_DIM)
        self.mem_attn = nn.MultiheadAttention(self.REC_DIM, 4, batch_first=True)
        self.mem_null = nn.Parameter(torch.zeros(1, 1, self.REC_DIM))
        self._init_weights()
        nn.init.normal_(self.age_w, std=0.02)
        nn.init.normal_(self.mem_null, std=0.02)

    # ----------------------------------------------------------------- pieces
    def parts(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        return (self.rays(x["rays"]), self.entities(x["ent_type"], x["ent_feat"], x["ent_mask"]),
                self.scalars(x["scalars"]))

    def record_static(self, ray_f: Tensor, ent_f: Tensor, keys: Tensor) -> Tensor:
        """Static (cacheable) record encoding s_i from encoder features + key flags at write time."""
        return self.rec_proj(torch.cat([ray_f, ent_f, self.rec_keys(keys)], -1))

    def encode_records(self, x: dict[str, Tensor]) -> Tensor:
        """Records' stored observations (K, ...) -> s (K, 128)."""
        ray_f, ent_f, _ = self.parts(x)
        return self.record_static(ray_f, ent_f, x["scalars"][..., 8:14])

    def read(self, c: Tensor, h: Tensor, keys: Tensor, s: Tensor, age: Tensor, valid: Tensor) -> Tensor:
        """c (B,256), h (B,H) already masked, keys (B,6), s (B,16,128), age (B,16), valid (B,16) -> m (B,128)."""
        B = c.shape[0]
        q = self.mem_query(torch.cat([c, h, keys], -1)).unsqueeze(1)
        e = s + age.unsqueeze(-1) * self.age_w
        kv = torch.cat([self.mem_null.expand(B, 1, -1), e], 1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=c.device), ~valid.bool()], 1)
        if self.debug_attention:
            m, w = self.mem_attn(q, kv, kv, key_padding_mask=kpm, need_weights=True)
            self.last_attention = w.squeeze(1).detach()          # (B, 17): null record + 16 slots
        else:
            m, _ = self.mem_attn(q, kv, kv, key_padding_mask=kpm, need_weights=False)
        return m.squeeze(1)

    def _step_core(self, c, h, start, keys, mem):
        h = h * (1.0 - start.float()).unsqueeze(-1)                         # F4: mask before the query
        if self.query_keys_off:
            keys = torch.zeros_like(keys)
        m = self.read(c, h, keys, mem["s"], mem["age"], mem["valid"])
        h = self.core(self.fusion(torch.cat([c, m], -1)), h)
        return h

    # ------------------------------------------------------------------- API
    def step(self, x, h, start, mem=None):
        """mem: {"s": (B,16,128), "age": (B,16), "valid": (B,16)}. Returns (logits, h, parts);
        parts = (ray_f, ent_f) of x so a new record's static encoding needs no extra pass."""
        if mem is None:
            mem = self.empty_memory(h.shape[0], h.device)
        ray_f, ent_f, sc_f = self.parts(x)
        c = torch.cat([ray_f, ent_f, sc_f], -1)
        h = self._step_core(c, h, start, x["scalars"][..., 8:14], mem)
        return self.policy(h), h, (ray_f, ent_f)

    def sequence(self, x, h0, starts, mem=None):
        """x (T,B,...), mem: {"s": (T,B,16,128), "age": (T,B,16), "valid": (T,B,16)}."""
        T, B = starts.shape
        flat = {k: v.reshape((T * B,) + v.shape[2:]) for k, v in x.items() if k in self.input_keys}
        ray_f, ent_f, sc_f = self.parts(flat)
        c = torch.cat([ray_f, ent_f, sc_f], -1).view(T, B, -1)
        keys = x["scalars"][..., 8:14]
        if mem is None:
            mem = {k: v.unsqueeze(0).expand(T, *v.shape) for k, v in self.empty_memory(B, h0.device).items()}
        h, zs = h0, []
        for t in range(T):
            h = self._step_core(c[t], h, starts[t], keys[t], {k: v[t] for k, v in mem.items()})
            zs.append(h)
        z = torch.stack(zs)
        return self.policy(z), z, h

    def empty_memory(self, B: int, device) -> dict:
        from .memory import N_SLOTS
        return {"s": torch.zeros(B, N_SLOTS, self.REC_DIM, device=device),
                "age": torch.zeros(B, N_SLOTS, device=device),
                "valid": torch.zeros(B, N_SLOTS, dtype=torch.bool, device=device)}

    @torch.no_grad()
    def load_baseline(self, actor_state: dict) -> None:
        """Warm start from a baseline Actor state dict: copy every shared weight; the fusion
        layer's 128 memory input columns start at ZERO, so outputs equal the baseline's exactly.
        Memory-only layers keep their fresh initialisation."""
        own = self.state_dict()
        for k, v in actor_state.items():
            if k == "fusion.0.weight":
                w = torch.zeros_like(own[k])
                w[:, :v.shape[1]] = v
                own[k] = w
            else:
                own[k] = v
        self.load_state_dict(own)
