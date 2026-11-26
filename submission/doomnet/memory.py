"""Episodic memory bookkeeping (Spec v2.1, Memory: slots, write rules, eviction).

Shared, unchanged, by DoomAgent (eval) and the PPO learner (training), so both keep exactly
the same records. Pure numpy; everything is decided from the adapter's outputs, i.e. only
from observations the agent is given. These rules decide WHAT IS REMEMBERED, never what
to do: retrieval and every decision are learned (the network in model.MemoryActor).

Every record keeps the adapter's dead-reckoned pose of its decision (x["pose"]); the network
reads each record together with where it is relative to the agent now (pose-tagged memory).

Per decision t (decisions since reset):
    slots = book.read()            # records written at steps < t (16 slots, None = empty)
    ... network reads them, acts ...
    book.write(x_t, t)             # may store x_t (adapter output) in a recent and/or event slot

Slots 0-7 RECENT: x_t is written every 8 decisions (t % 8 == 0) into a ring buffer.
Slots 8-15 EVENT: written when a condition STARTS being true (F6):
    locked_door  a locked-door hit type appears in the central rays
    door         a door hit type appears in the central rays
    key_seen     a key entity comes into view
    inventory    the key flags change
    new_area     the distance profile changes sharply (sorted distances, so pure turning
                 doesn't count; threshold calibrated on real maps: ~1 per 130 decisions)
  At most one event record per decision (first reason in the order above that may write).
  Duplicate suppression: a reason can't write again within 16 decisions.
  Eviction when full: if the reason already holds >= 2 records, replace its oldest; otherwise
  replace the oldest record of whichever reason holds the most.
"""
from __future__ import annotations

import numpy as np

from . import obs_adapter as A

N_RECENT, N_EVENT = 8, 8
N_SLOTS = N_RECENT + N_EVENT
RECENT_EVERY = 8
DEDUP = 16
REASONS = ("locked_door", "door", "key_seen", "inventory", "new_area")
CENTRAL_RAYS = np.array([62, 63, 0, 1, 2])         # ~ +-28 degrees around straight ahead
HIT_DOOR, HIT_LOCKED = 2, 3
ENT_KEY = 9
PROFILE_THRESHOLD = 0.025    # mean |sorted distance change|; ~99th percentile on real rooms maps (calibrated 26 Nov)
CONTENT_KEYS = A.INPUT_KEYS


class Record:
    __slots__ = ("x", "t", "reason", "s", "rid")

    def __init__(self, x: dict, t: int, reason: str):
        self.x = {k: np.array(x[k], copy=True) for k in CONTENT_KEYS}
        self.t, self.reason = t, reason
        self.s = None          # cached static encoding (set by the owner; invalid after weight changes)
        self.rid = -1          # row in the learner's per-rollout record table


def _present(x: dict) -> dict:
    hit = x["rays"][1:].argmax(0)[CENTRAL_RAYS]
    ents = x["ent_type"][x["ent_mask"]]
    return {"locked_door": bool((hit == HIT_LOCKED).any()), "door": bool((hit == HIT_DOOR).any()),
            "key_seen": bool((ents == ENT_KEY).any())}


class MemoryBook:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.slots: list[Record | None] = [None] * N_SLOTS
        self.prev_present = {"locked_door": False, "door": False, "key_seen": False}
        self.prev_keys: np.ndarray | None = None
        self.prev_profile: np.ndarray | None = None
        self.last_write = {r: -10**9 for r in REASONS}

    def read(self) -> list:
        return list(self.slots)

    def triggers(self, x: dict) -> list[str]:
        """Reasons whose condition STARTS at this observation (also updates the 'previous' state)."""
        now = _present(x)
        out = [r for r in ("locked_door", "door", "key_seen") if now[r] and not self.prev_present[r]]
        keys = x["scalars"][A.S_KEYS]
        if self.prev_keys is not None and not np.array_equal(keys, self.prev_keys):
            out.append("inventory")
        profile = np.sort(x["rays"][0])
        if self.prev_profile is not None and float(np.abs(profile - self.prev_profile).mean()) > PROFILE_THRESHOLD:
            out.append("new_area")
        self.prev_present, self.prev_keys, self.prev_profile = now, keys.copy(), profile
        return out

    def write(self, x: dict, t: int) -> list[tuple[int, Record]]:
        """Store x_t where the rules say so; returns [(slot, new record)] (0, 1 or 2 entries)."""
        new = []
        if t % RECENT_EVERY == 0:
            rec, slot = Record(x, t, "recent"), (t // RECENT_EVERY) % N_RECENT
            self.slots[slot] = rec
            new.append((slot, rec))
        for reason in self.triggers(x):
            if t - self.last_write[reason] < DEDUP:
                continue
            self.last_write[reason] = t
            rec, slot = Record(x, t, reason), N_RECENT + self._event_slot(reason)
            self.slots[slot] = rec
            new.append((slot, rec))
            break                                   # at most one event record per decision
        return new

    def _event_slot(self, reason: str) -> int:
        ev = self.slots[N_RECENT:]
        for i, r in enumerate(ev):
            if r is None:
                return i
        same = [i for i, r in enumerate(ev) if r.reason == reason]
        if len(same) >= 2:
            return min(same, key=lambda i: ev[i].t)
        counts: dict[str, int] = {}
        for r in ev:
            counts[r.reason] = counts.get(r.reason, 0) + 1
        top = max(counts.values())
        cand = [i for i, r in enumerate(ev) if counts[r.reason] == top]
        return min(cand, key=lambda i: ev[i].t)

    def ages(self, t: int) -> np.ndarray:
        """Normalised ages (decisions since written / 1050) of the current slots; 0 for empty."""
        return np.array([0.0 if r is None else (t - r.t) / A.MAX_DECISIONS for r in self.slots], np.float32)

    def valid(self) -> np.ndarray:
        return np.array([r is not None for r in self.slots], bool)

    def poses(self) -> np.ndarray:
        """(16, 3) dead-reckoned pose [x, y, yaw] each record was written at; 0 for empty slots.
        The network turns these into offsets from the current pose at every read."""
        return np.stack([np.zeros(3, np.float32) if r is None else r.x["pose"] for r in self.slots])
