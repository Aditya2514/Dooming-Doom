"""Load every map of a generated pool in the real engine and check it plays.

    .venv\\Scripts\\python.exe -m training.maps.validate --family combat --pool train

Per map: the wrapper loads it, the player spawns alive, a short random run doesn't crash,
and the engine's object list holds exactly the generated number of monsters (offline
validation only: engine access here never reaches the agent). Also reports whether an
enemy was visible from the spawn (rays or entities while turning) - informational, since
an enemy behind the pillar is legitimately hidden.
"""
from __future__ import annotations

import argparse

import numpy as np

from training.maps.gen import load_index
from training.maprunner import MapRunner

ENEMY_RAY = 4
ENEMY_TYPES = (1, 2, 3)          # melee, hitscan, projectile enemies
MONSTER_NAMES = {"Zombieman", "ZombieMan", "ShotgunGuy", "DoomImp", "Demon", "ChaingunGuy", "Spectre", "LostSoul",
                 "Cacodemon", "HellKnight", "BaronOfHell", "Arachnotron", "PainElemental", "Revenant",
                 "Fatso", "Archvile", "WolfensteinSS", "SpiderMastermind", "Cyberdemon"}


def check_map(runner: MapRunner, e: dict, rng) -> dict:
    env = runner.load(e)
    obs, _ = env.reset(seed=0)
    ps = env.privileged_state()
    objs = env._game.get_state().objects
    n_mon = sum(o.name in MONSTER_NAMES for o in objs)
    res = {"map": e["label"], "alive": ps["health"] > 0, "monsters": n_mon, "expected": e.get("monsters", 0),
           "enemy_seen": False, "crash": None}
    turn = np.zeros(9, np.int64)
    turn[4] = 1
    try:
        for t in range(60):
            seen = (obs["rays"]["hit_type"] == ENEMY_RAY).any() or np.isin(
                obs["visible_entities"]["entity_type"][obs["visible_entities"]["mask"].astype(bool)], ENEMY_TYPES).any()
            res["enemy_seen"] |= bool(seen)
            a = turn if t < 30 else rng.integers(0, 2, 9)
            obs, _, te, tr, _ = env.step(a)
            if te or tr:
                break
    except Exception as ex:      # noqa: BLE001
        res["crash"] = repr(ex)
    lock = ps["door_lock"]
    res["doors"] = int((lock >= 0).sum()) if lock is not None else 0
    res["locked"] = int((lock > 0).sum()) if lock is not None else 0
    doors_ok = ("n_doors" not in e or res["doors"] == e["n_doors"]) and                ("n_locked" not in e or res["locked"] == e["n_locked"])
    res["ok"] = res["alive"] and res["crash"] is None and n_mon == res["expected"] and doors_ok
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="combat")
    ap.add_argument("--pool", default="train")
    a = ap.parse_args()
    entries = load_index(a.pool, a.family)
    runner = MapRunner()
    rng = np.random.default_rng(0)
    bad, unseen = [], 0
    try:
        for e in entries:
            r = check_map(runner, e, rng)
            if not r["ok"]:
                bad.append(r)
            unseen += r["expected"] > 0 and not r["enemy_seen"]
    finally:
        runner.close()
    print(f"{a.family}/{a.pool}: {len(entries) - len(bad)}/{len(entries)} maps OK "
          f"({unseen} with no enemy visible from the spawn)")
    for r in bad:
        print("  BAD", r)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
