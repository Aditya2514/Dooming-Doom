"""9 independent Bernoulli buttons (contract: MultiBinary(9)), treated as ONE joint action:
log pi(a) = sum of the 9 Bernoulli log-probs; entropy = sum of the 9 entropies (Spec v2.1)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sample(logits: Tensor, greedy: bool = False) -> Tensor:
    """{0,1} float actions, same shape as logits. greedy = threshold at p = 0.5."""
    if greedy:
        return (logits > 0).to(logits.dtype)
    return torch.bernoulli(torch.sigmoid(logits))


def log_prob(logits: Tensor, actions: Tensor) -> Tensor:
    """Joint log-prob over the last dim (the 9 buttons)."""
    return -F.binary_cross_entropy_with_logits(logits, actions.to(logits.dtype), reduction="none").sum(-1)


def entropy(logits: Tensor) -> Tensor:
    """Per-button Bernoulli entropy H = softplus(l) - l * sigmoid(l); summed over buttons."""
    return (F.softplus(logits) - logits * torch.sigmoid(logits)).sum(-1)


def button_entropy(logits: Tensor) -> Tensor:
    """Per-button entropies (..., 9), for logging."""
    return F.softplus(logits) - logits * torch.sigmoid(logits)
