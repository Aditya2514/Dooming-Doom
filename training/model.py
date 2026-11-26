"""Training-side wrapper: acting network + auxiliary head + privileged critic.

Total for the baseline: 726,731 parameters = Actor 691,401 + aux 257 + critic 35,073.
The critic sees the actor's hidden state plus 16 privileged inputs (obs["priv"],
training only). Nothing here is imported by submission/.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from submission.doomnet import dist
from submission.doomnet.model import Actor, RecurrentActor, count_params  # noqa: F401

PRIV_DIM = 16


class ActorCritic(nn.Module):
    def __init__(self, actor: RecurrentActor | None = None, priv_dim: int = PRIV_DIM, n_aux: int = 1,
                 critic_hidden: int = 128):
        super().__init__()
        self.actor = actor if actor is not None else Actor()
        H = self.actor.hidden
        self.priv_dim = priv_dim
        self.aux = nn.Linear(H, n_aux) if n_aux else None
        self.critic = nn.Sequential(nn.Linear(H + priv_dim, critic_hidden), nn.ReLU(), nn.Linear(critic_hidden, 1))
        nn.init.orthogonal_(self.critic[0].weight, math.sqrt(2))
        nn.init.zeros_(self.critic[0].bias)
        nn.init.orthogonal_(self.critic[2].weight, 1.0)
        nn.init.zeros_(self.critic[2].bias)
        if self.aux is not None:
            nn.init.orthogonal_(self.aux.weight, 1.0)
            nn.init.zeros_(self.aux.bias)

    @property
    def hidden(self) -> int:
        return self.actor.hidden

    @property
    def n_buttons(self) -> int:
        return self.actor.n_buttons

    def initial_state(self, n: int, device=None) -> Tensor:
        return self.actor.initial_state(n, device)

    def _heads(self, z: Tensor, priv: Tensor | None):
        cin = z if self.priv_dim == 0 else torch.cat([z, priv], -1)
        value = self.critic(cin).squeeze(-1)
        aux = self.aux(z) if self.aux is not None else z.new_zeros(z.shape[:-1] + (0,))
        return value, aux

    def forward(self, obs: dict[str, Tensor], h: Tensor, start: Tensor):
        logits, h_new = self.actor.step(obs, h, start)
        value, aux = self._heads(h_new, obs.get("priv"))
        return logits, value, aux, h_new

    def forward_sequence(self, obs: dict[str, Tensor], h0: Tensor, starts: Tensor):
        logits, z, h = self.actor.sequence(obs, h0, starts)
        value, aux = self._heads(z, obs.get("priv"))
        return logits, value, aux, h

    @torch.no_grad()
    def act(self, obs: dict[str, Tensor], h: Tensor, start: Tensor, greedy: bool = False):
        logits, value, aux, h_new = self.forward(obs, h, start)
        actions = dist.sample(logits, greedy)
        return actions, dist.log_prob(logits, actions), value, aux, h_new, logits

    def acting_parameters(self):
        return list(self.actor.parameters())
