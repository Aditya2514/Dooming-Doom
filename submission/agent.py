"""Evaluation entry point (docs/CONTRACT.md section 6). The scorer instantiates DoomAgent from here.

    official observation -> doomnet/obs_adapter.py (shared with training: identical tensors)
    -> doomnet/policy.py (ray conv + entity attention + vitals + spatial-map CNN -> GRU 256)
    -> one macro-action (doomnet/actions.py) -> 9 official buttons

Every decision comes from the trained network. Weights: weights/actor.pt (Git LFS), trained by
training/dagger.py (imitation of a privileged teacher; see training/README.md). Nothing here
imports training code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from doomnet import obs_adapter as A  # noqa: E402
from doomnet.actions import BUTTONS, NUM_ACTIONS  # noqa: E402
from doomnet.policy import INPUT_KEYS, DoomPolicy  # noqa: E402

WEIGHTS_FILE = WEIGHTS_DIR / "actor.pt"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once: build the network and load the weights onto `device`."""
        self.observation_space = observation_space
        self.action_space = action_space
        torch.set_num_threads(1)                      # evaluation gives each agent one CPU thread
        # Always CPU: the network is small (~2.5 ms per decision on one thread), while per-step GPU
        # calls on a GPU shared by many evaluation workers measured 19 ms median / 38 ms max under
        # load, against a 40 ms budget. `device` is accepted for the interface and ignored.
        self.device = torch.device("cpu")
        ckpt = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=False) if WEIGHTS_FILE.is_file() else None
        meta = ckpt.get("meta", {}) if ckpt else {}
        use_map = any(k.startswith("map.") for k in ckpt["state_dict"]) if ckpt else True
        self.policy = DoomPolicy(NUM_ACTIONS, use_map=use_map)
        self.greedy = bool(meta.get("greedy", False))
        self.temperature = float(meta.get("temperature", 1.0))
        if ckpt is not None:
            self.policy.load_state_dict(ckpt["state_dict"])
        else:
            print(f"[DoomAgent] no weights at {WEIGHTS_FILE}: using an untrained network", flush=True)
        self.policy.to(self.device).eval()
        self.adapter = A.ObsAdapter()
        self.gen = torch.Generator(device="cpu")
        self.reset()
        self.act(observation_space.sample())          # warm-up (kernels, allocator) before the first map
        self.reset()

    def reset(self) -> None:
        """Before every map: clear the recurrent memory and the adapter's tracked inputs."""
        self.h = self.policy.initial_state(1, self.device)
        self.adapter.reset()
        self.gen.manual_seed(0)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        x = self.adapter.observe(observation)
        xt = {k: torch.from_numpy(np.asarray(x[k])[None]).to(self.device) for k in INPUT_KEYS}
        logits, self.h = self.policy.step(xt, self.h)
        logits = logits[0].float().cpu()
        if self.greedy:
            a = int(logits.argmax())
        else:
            p = torch.softmax(logits / self.temperature, -1)
            a = int(torch.multinomial(p, 1, generator=self.gen))
        action = BUTTONS[a].astype(np.int8)
        self.adapter.record_action(action)
        self.last_action_index = a
        return action
