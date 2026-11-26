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
