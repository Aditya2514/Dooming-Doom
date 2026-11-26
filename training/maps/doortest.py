"""Deterministic door-reward test map (Spec v2.1 Rewards: 'Deterministic test before enabling').

A straight 128-wide corridor along +x; the player spawns at x=64 facing +x:

    room1 | door A (ordinary) | room2 (red key) | door C (open at reset) | room3 |
    door D (ordinary, floor raised 24) | room4 | door B (red-card locked) | room5

MAP01: as above.        Expected door rewards: A +0.5, C 0, D +0.5, B +2.0 (key held).
MAP02: no red key.      B stays shut: 0.
MAP03: only door A, which the script opens, lets close, and opens again: +0.5 once.
"""
from __future__ import annotations

from pathlib import Path

from training.maps import udmf

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "maps" / "generated" / "doortest.wad"
W = 128
DOOR_X = {"A": 256, "C": 544, "D": 832, "B": 1120}


def _map(with_key: bool) -> str:
    R = udmf.Rect
    rects = [
        R(0, 0, 256, W),
        R(256, 0, 288, W, door=True),                       # A
        R(288, 0, 544, W),
        R(544, 0, 576, W, door=True, starts_open=True),     # C: open at reset
        R(576, 0, 832, W),
        R(832, 0, 864, W, floor=24, door=True),             # D: raised floor
        R(864, 0, 1120, W),
        R(1120, 0, 1152, W, door=True, lock=udmf.LOCK_RED_CARD),   # B
        R(1152, 0, 1408, W),
    ]
    things = [udmf.Thing(64, W // 2, udmf.THING_PLAYER1, 0)]
    if with_key:
        things.append(udmf.Thing(416, W // 2, udmf.THING_RED_CARD))
    return udmf.build_textmap(rects, things)


def _map_reopen() -> str:
    R = udmf.Rect
    rects = [R(0, 0, 256, W), R(256, 0, 288, W, door=True), R(288, 0, 544, W)]
    return udmf.build_textmap(rects, [udmf.Thing(64, W // 2, udmf.THING_PLAYER1, 0)])


def write(path: Path = OUT) -> Path:
    return udmf.write_wad(path, {"MAP01": _map(True), "MAP02": _map(False), "MAP03": _map_reopen()})


if __name__ == "__main__":
    print(write())
