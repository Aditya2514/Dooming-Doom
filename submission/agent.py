"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6. Inference path (Spec v2.1):
    official observation -> shared adapter (doomnet/obs_adapter.py) -> encoders -> GRU
    -> 9 Bernoulli logits -> action (threshold at 0.5, or sample: frozen at selection)
Weights: weights/actor.pt (Git LFS), written by training/export.py. Nothing here imports
training code; the critic and auxiliary head are not part of the submission.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
if str(SUBMISSION_DIR) not in sys.path:          # the scorer adds it too; keep local runs identical
    sys.path.insert(0, str(SUBMISSION_DIR))

from doomnet import obs_adapter as A  # noqa: E402
from doomnet.dist import sample  # noqa: E402
from doomnet.model import Actor  # noqa: E402

WEIGHTS_FILE = WEIGHTS_DIR / "actor.pt"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once. Loads weights onto `device` ("cuda" or "cpu")."""
        self.observation_space = observation_space
        self.action_space = action_space
        torch.set_num_threads(1)                   # evaluation gives each agent one CPU thread
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        self.actor = Actor()
        self.greedy = True
        self.meta: dict = {}
        if WEIGHTS_FILE.is_file():
            ckpt = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=False)
            self.actor.load_state_dict(ckpt["state_dict"])
            self.meta = dict(ckpt.get("meta", {}))
            self.greedy = bool(self.meta.get("greedy", True))
        else:
            print(f"[DoomAgent] no weights at {WEIGHTS_FILE}: using an untrained network", flush=True)
        self.actor.to(self.device).eval()
        self.gen = torch.Generator(device=self.device)
        self.adapter = A.ObsAdapter()
        self.reset()
        # warm-up so the first act() of a map doesn't pay for CUDA/kernel initialisation
        with torch.inference_mode():
            dummy = A.stack([self.adapter.observe(observation_space.sample())])
            self.actor.step(self._tensors(dummy), self.h, self.start)
        self.reset()

    def _tensors(self, batch: dict) -> dict:
        return {k: torch.from_numpy(batch[k]).to(self.device) for k in A.INPUT_KEYS}

    def reset(self) -> None:
        """Before every map: zero the GRU state, clear the adapter's tracked inputs, reseed sampling."""
        self.h = self.actor.initial_state(1, self.device)
        self.start = torch.ones(1, device=self.device)
        self.adapter.reset()
        self.gen.manual_seed(0)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        x = A.stack([self.adapter.observe(observation)])
        logits, self.h = self.actor.step(self._tensors(x), self.h, self.start)
        self.start = torch.zeros(1, device=self.device)
        if self.greedy:
            a = sample(logits, greedy=True)
        else:
            a = torch.bernoulli(torch.sigmoid(logits), generator=self.gen)
        action = a[0].to("cpu").numpy().astype(np.int8)
        self.adapter.record_action(action)
        self.last_logits = logits
        return action
