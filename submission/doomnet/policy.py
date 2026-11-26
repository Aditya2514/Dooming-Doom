"""The acting network: official observation (via obs_adapter) -> one macro-action per decision.

    rays (7, 64)       -> RayEncoder: circular Conv1d over 360 deg (ray 63 touches ray 0)  -> 128
    entities (32 x ..) -> EntityEncoder: type embedding + MLP, masked self-attention, attention
                          pooling with a learned query over [null token, entities]            -> 64
                          (permutation-invariant; handles 0..32 entities)
    scalars (28)       -> Linear: vitals, previous buttons, previous observable reward, time  -> 64
    map (4, 24, 24)    -> MapEncoder: CNN over the ego-centric crop of the dead-reckoned spatial
                          map (visited / swept by rays / doors / locked doors)               -> 128
    concat -> Linear + LayerNorm + ReLU -> GRU 256 (memory across the whole map)
    -> policy head: logits over the macro-actions (doomnet/actions.py)
    -> value head: state value (used only by RL fine-tuning; ignored when acting)

Training (training/dagger.py) runs whole episodes through nn.GRU (cuDNN); acting runs the very
same module one step at a time, so the two paths compute identical functions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from . import obs_adapter as A
from .actions import NUM_ACTIONS

HIDDEN = 256
INPUT_KEYS = ("rays", "ent_type", "ent_feat", "ent_mask", "scalars", "map")


def ortho_init(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class RayEncoder(nn.Module):
    """(N, 7, 64) -> (N, 128). 64 -> 64 -> 32 -> 16 positions, circular padding."""

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
    """(N, 32) types + (N, 32, 5) features + (N, 32) mask -> (N, 64)."""

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
        sa_pad = pad.clone()
        sa_pad[:, 0] &= valid.any(1)          # all-empty rows: let slot 0 attend (never pooled)
        a, _ = self.self_attn(x, x, x, key_padding_mask=sa_pad, need_weights=False)
        x = self.ln1(x + a)
        B = x.shape[0]
        keys = torch.cat([self.null.expand(B, 1, -1), x], 1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], 1)
        p, _ = self.pool_attn(self.query.expand(B, 1, -1), keys, keys, key_padding_mask=kpm, need_weights=False)
        return self.ln2(p.squeeze(1))


class MapEncoder(nn.Module):
    """(N, 4, 24, 24) ego-centric spatial-map crop -> (N, 128). 24 -> 12 -> 6."""

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(A.MAP_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(32 * (A.MAP_CROP // 4) ** 2, 128), nn.ReLU())

    def forward(self, m: Tensor) -> Tensor:
        return self.fc(self.convs(m.to(self.fc[0].weight.dtype)).flatten(1))


class DoomPolicy(nn.Module):
    def __init__(self, num_actions: int = NUM_ACTIONS, hidden: int = HIDDEN, use_map: bool = True):
        super().__init__()
        self.num_actions, self.hidden, self.use_map = num_actions, hidden, use_map
        self.rays = RayEncoder()
        self.entities = EntityEncoder()
        self.scalars = nn.Sequential(nn.Linear(A.NUM_SCALARS, 64), nn.ReLU())
        if use_map:
            self.map = MapEncoder()
        self.fusion = nn.Sequential(nn.Linear(128 + 64 + 64 + (128 if use_map else 0), hidden),
                                    nn.LayerNorm(hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden)
        self.pi = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, num_actions))
        self.v = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        ortho_init(self)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.orthogonal_(self.v[-1].weight, 1.0)

    def initial_state(self, batch: int, device=None) -> Tensor:
        return torch.zeros(1, batch, self.hidden, device=device)

    def encode(self, x: dict[str, Tensor]) -> Tensor:
        """Inputs with any leading shape L (flattened inside) -> (L..., hidden)."""
        lead = x["scalars"].shape[:-1]
        f = {k: v.reshape((-1,) + v.shape[len(lead):]) for k, v in x.items() if k in INPUT_KEYS}
        parts = [self.rays(f["rays"]), self.entities(f["ent_type"], f["ent_feat"], f["ent_mask"]),
                 self.scalars(f["scalars"])]
        if self.use_map:
            parts.append(self.map(f["map"]))
        z = torch.cat(parts, -1)
        return self.fusion(z).view(*lead, -1)

    def forward(self, x: dict[str, Tensor], h0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x: (T, B, ...) time-major; h0: (1, B, H). -> logits (T, B, A), values (T, B), hT."""
        e = self.encode(x)
        z, h = self.gru(e, h0)
        return self.pi(z), self.v(z).squeeze(-1), h

    @torch.inference_mode()
    def step(self, x: dict[str, Tensor], h: Tensor) -> tuple[Tensor, Tensor]:
        """One decision for a batch: x (B, ...), h (1, B, H) -> logits (B, A), new h."""
        e = self.encode(x).unsqueeze(0)
        z, h = self.gru(e, h)
        return self.pi(z[0]), h
