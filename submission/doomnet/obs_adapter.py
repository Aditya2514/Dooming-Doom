"""Shared observation adapter (Spec v2.1, "One shared adapter").

The ONLY code that turns the official observation dict into network inputs. Both the
training env workers and DoomAgent.act() call it, so training and evaluation see
identical tensors. A training/eval preprocessing mismatch is otherwise silent.

Per decision, in this order:
    x = adapter.observe(obs)        # network inputs for this decision
    action = policy(x)
    adapter.record_action(action)   # the 9 buttons actually sent
adapter.reset() before every map / episode.

Outputs (numpy, so env workers never import torch):
    rays      float32 (7, 64)   channel 0 = distance (0..1), channels 1..6 = one-hot hit type
                                (none, wall, door, locked door, enemy, item). Ray 0 = ahead, CCW.
    ent_type  int64   (32,)     entity class 0..11 (0 = padding)
    ent_feat  float32 (32, 5)   [distance/1024 (0..4), sin, cos, rel_vel_fwd/16, rel_vel_right/16]
    ent_mask  bool    (32,)
    scalars   float32 (28,)     17 normalised vitals | 9 previous buttons | previous observable
                                reward | elapsed decisions / 1050
"""
from __future__ import annotations

import numpy as np

NUM_RAYS = 64
NUM_HIT_TYPES = 6
RAY_CHANNELS = 1 + NUM_HIT_TYPES            # 7
MAX_ENTITIES = 32
ENT_FEATURES = 5
NUM_ENTITY_TYPES = 12
NUM_VITALS = 17
NUM_BUTTONS = 9
NUM_SCALARS = NUM_VITALS + NUM_BUTTONS + 2  # 28
MAX_DECISIONS = 1050                        # 4,200-tic cap / frame skip 4

# vitals layout (docs/CONTRACT.md §2.1)
HEALTH, ARMOR, WEAPON, WEAPON_AMMO = 0, 1, 2, 3
AMMO = slice(4, 8)
KEYS = slice(8, 14)
VEL = slice(14, 17)

# Normalisation [guessing; re-check against logged ranges]. Spec v2.1, Observations.
HP_SCALE = 100.0        # health, armor (0..200)
WEAPON_SCALE = 7.0      # slot 0..7
AMMO_LOG_SCALE = 5.0    # log(1 + ammo) / 5
VEL_SCALE = 16.0        # units per tic, own velocity and entity relative velocity

# Previous observable reward: a fixed function of vitals changes only (identical at eval).
OBS_REWARD_KEY = 1.0            # per key flag gained
OBS_REWARD_HEALTH = -0.01       # per health point lost

# Scalar layout of the output vector
S_VITALS = slice(0, NUM_VITALS)
S_PREV_ACTION = slice(NUM_VITALS, NUM_VITALS + NUM_BUTTONS)
S_PREV_REWARD = NUM_VITALS + NUM_BUTTONS
S_ELAPSED = NUM_VITALS + NUM_BUTTONS + 1
S_KEYS = slice(KEYS.start, KEYS.stop)       # raw 0/1 key flags inside scalars (memory query input)


def normalise_vitals(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    out = np.empty(NUM_VITALS, np.float32)
    out[HEALTH] = v[HEALTH] / HP_SCALE
    out[ARMOR] = v[ARMOR] / HP_SCALE
    out[WEAPON] = v[WEAPON] / WEAPON_SCALE
    out[WEAPON_AMMO] = np.log1p(max(float(v[WEAPON_AMMO]), 0.0)) / AMMO_LOG_SCALE
    out[AMMO] = np.log1p(np.maximum(v[AMMO], 0.0)) / AMMO_LOG_SCALE
    out[KEYS] = v[KEYS]
    out[VEL] = v[VEL] / VEL_SCALE
    return out


def observable_reward(prev_vitals: np.ndarray | None, vitals: np.ndarray) -> float:
    """Key flags gained and health lost since the previous decision; 0 on the first one."""
    if prev_vitals is None:
        return 0.0
    keys_gained = float(np.sum(np.maximum(vitals[KEYS] - prev_vitals[KEYS], 0.0)))
    health_lost = max(float(prev_vitals[HEALTH] - vitals[HEALTH]), 0.0)
    return OBS_REWARD_KEY * keys_gained + OBS_REWARD_HEALTH * health_lost


def encode(obs: dict, prev_action: np.ndarray, prev_reward: float, elapsed: int) -> dict:
    """Pure function: one official observation + our tracked inputs -> network inputs."""
    rays_in = obs["rays"]
    dist = np.asarray(rays_in["distance"], np.float32).reshape(NUM_RAYS)
    hit = np.clip(np.asarray(rays_in["hit_type"], np.int64).reshape(NUM_RAYS), 0, NUM_HIT_TYPES - 1)
    rays = np.zeros((RAY_CHANNELS, NUM_RAYS), np.float32)
    rays[0] = dist
    rays[1 + hit, np.arange(NUM_RAYS)] = 1.0

    ents = obs["visible_entities"]
    mask = np.asarray(ents["mask"]).reshape(MAX_ENTITIES).astype(bool)
    ent_type = np.clip(np.asarray(ents["entity_type"], np.int64).reshape(MAX_ENTITIES), 0, NUM_ENTITY_TYPES - 1)
    ent_feat = np.asarray(ents["features"], np.float32).reshape(MAX_ENTITIES, ENT_FEATURES).copy()
    ent_feat[:, 3:5] /= VEL_SCALE
    ent_feat[~mask] = 0.0
    ent_type[~mask] = 0

    scalars = np.empty(NUM_SCALARS, np.float32)
    scalars[S_VITALS] = normalise_vitals(obs["vitals"])
    scalars[S_PREV_ACTION] = np.asarray(prev_action, np.float32).reshape(NUM_BUTTONS)
    scalars[S_PREV_REWARD] = prev_reward
    scalars[S_ELAPSED] = elapsed / MAX_DECISIONS
    return {"rays": rays, "ent_type": ent_type, "ent_feat": ent_feat, "ent_mask": mask, "scalars": scalars}


class ObsAdapter:
    """Stateful wrapper around encode(): tracks previous buttons, previous vitals, decision count."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.prev_action = np.zeros(NUM_BUTTONS, np.float32)
        self.prev_vitals: np.ndarray | None = None
        self.elapsed = 0

    def observe(self, obs: dict) -> dict:
        vitals = np.asarray(obs["vitals"], np.float32).reshape(NUM_VITALS)
        r = observable_reward(self.prev_vitals, vitals)
        x = encode(obs, self.prev_action, r, self.elapsed)
        self.prev_vitals = vitals.copy()
        return x

    def record_action(self, action) -> None:
        a = np.asarray(action).reshape(NUM_BUTTONS)
        self.prev_action = (a != 0).astype(np.float32)
        self.elapsed += 1


INPUT_KEYS = ("rays", "ent_type", "ent_feat", "ent_mask", "scalars")


def stack(xs: list[dict]) -> dict:
    """List of adapter outputs -> batched numpy arrays (leading batch dim)."""
    return {k: np.stack([x[k] for x in xs]) for k in INPUT_KEYS}


def to_torch(x: dict, device="cpu") -> dict:
    """Numpy adapter output (batched or not) -> torch tensors on `device`. torch imported lazily."""
    import torch

    return {k: torch.as_tensor(x[k]).to(device) for k in INPUT_KEYS}
