"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6. Inference path (Spec v2.1):
    official observation -> shared adapter (doomnet/obs_adapter.py) -> encoders
    [-> episodic memory read (memory variant)] -> GRU -> 9 Bernoulli logits
    -> action (threshold at 0.5, or sample: frozen at selection)
The memory variant keeps the same MemoryBook the learner uses; each record's static
encoding is computed once when written (weights are fixed at eval) and its age and its offset
from the current dead-reckoned pose are added fresh at every read (Spec v2.1 F2; pose-tagged
memory: the adapter's PoseTracker, built from the own-velocity vitals and our own buttons).
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
from doomnet.memory import N_SLOTS, MemoryBook  # noqa: E402
from doomnet.model import Actor, MemoryActor  # noqa: E402

WEIGHTS_FILE = WEIGHTS_DIR / "actor.pt"


class DoomAgent:
    # Evaluation-only memory ablations (Spec v2.1 Experiments). None in the submission.
    #   "memory_off"   only the null record is visible (closest to 'B without memory')
    #   "clear_at_key" the memory is cleared whenever the key flags change
    #   "foreign"      records come from the previous episode instead of this one
    #   "no_inv_query" the explicit inventory input to the query is zeroed
    #   "no_pose"      records are read without their positions (pose-tagged memory off)
    ablation: str | None = None

    def __init__(self, observation_space, action_space, device: str):
        """Called once. Loads weights onto `device` ("cuda" or "cpu")."""
        self.observation_space = observation_space
        self.action_space = action_space
        torch.set_num_threads(1)                   # evaluation gives each agent one CPU thread
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        self.greedy = True
        self.meta: dict = {}
        ckpt = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=False) if WEIGHTS_FILE.is_file() else None
        if ckpt is not None:
            self.meta = dict(ckpt.get("meta", {}))
        self.memory = self.meta.get("arch") == "memory"
        self.actor = MemoryActor() if self.memory else Actor()
        if ckpt is not None:
            self.actor.load_state_dict(ckpt["state_dict"])
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
            self._forward(self._tensors(dummy))
        self.reset()

    def _tensors(self, batch: dict) -> dict:
        return {k: torch.from_numpy(batch[k]).to(self.device) for k in A.INPUT_KEYS}

    def reset(self) -> None:
        """Before every map: zero the GRU state, clear the adapter's tracked inputs, reseed sampling."""
        self.h = self.actor.initial_state(1, self.device)
        self.start = torch.ones(1, device=self.device)
        self.adapter.reset()
        self.gen.manual_seed(0)
        if self.memory:
            if self.ablation == "foreign" and getattr(self, "book", None) is not None:
                # freeze the finished episode's records (with their cached encodings) as donor
                self.donor = (self.book, self.cache.clone(), self.t)
            self.book = MemoryBook()
            self.cache = torch.zeros(1, N_SLOTS, MemoryActor.REC_DIM, device=self.device)
            self.t = 0
            self.prev_keys = None
            self.actor.query_keys_off = self.ablation == "no_inv_query"

    def _forward(self, x: dict):
        if not self.memory:
            logits, self.h = self.actor.step(x, self.h, self.start)
            return logits, None
        book, cache, t = self.book, self.cache, self.t
        if self.ablation == "foreign":
            donor = getattr(self, "donor", None)
            book, cache, t = donor if donor is not None else (MemoryBook(), torch.zeros_like(self.cache), 0)
        valid = torch.as_tensor(book.valid(), device=self.device)[None]
        if self.ablation == "memory_off":
            valid = torch.zeros_like(valid)
        mem = {"s": cache * valid[..., None],
               "age": torch.as_tensor(book.ages(t), device=self.device)[None], "valid": valid,
               "rpose": torch.as_tensor(book.poses(), device=self.device)[None]}
        if self.ablation == "no_pose":
            del mem["rpose"]
        logits, self.h, parts = self.actor.step(x, self.h, self.start, mem)
        return logits, parts

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        xn = self.adapter.observe(observation)
        x = self._tensors(A.stack([xn]))
        logits, parts = self._forward(x)
        self.start = torch.zeros(1, device=self.device)
        if self.memory:
            keys = xn["scalars"][8:14]
            if self.ablation == "clear_at_key" and self.prev_keys is not None and (keys != self.prev_keys).any():
                self.book = MemoryBook()                     # key pickup: forget everything so far
                self.cache.zero_()
            self.prev_keys = keys.copy()
            for slot, _rec in self.book.write(xn, self.t):
                self.cache[0, slot] = self.actor.record_static(parts[0], parts[1], x["scalars"][:, 8:14])[0]
            self.t += 1
        if self.greedy:
            a = sample(logits, greedy=True)
        else:
            a = torch.bernoulli(torch.sigmoid(logits), generator=self.gen)
        action = a[0].to("cpu").numpy().astype(np.int8)
        self.adapter.record_action(action)
        self.last_logits = logits
        return action
