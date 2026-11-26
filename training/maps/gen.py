"""Procedural training maps (Spec v2.1 Phase 2). One WAD per (family, pool); maps M0001...

    .venv\\Scripts\\python.exe -m training.maps.gen --all                 # every family x pool
    .venv\\Scripts\\python.exe -m training.maps.gen --family combat --pool train --n 100

Every family/pool pair writes maps/generated/<pool>/<family>.wad plus index_<family>.json:
one schedule entry per map ({"wad", "map", "ticks", ...}, directly usable by
training/vec_env.py) with the generator version, seed and a layout hash. Pools use
disjoint seed ranges and layouts already used by another pool are rejected, so
validation and test never repeat a training layout. Generation is deterministic:
the 5060 regenerates identical maps from the same command.

Families
    combat  Stage 1: one room (3x3 tiles, varied floor heights, optional centre pillar),
            1-5 enemies away from the spawn, ammo, sometimes a shotgun, health, armour.
            No exit: the episode ends when every enemy is dead (end_on_clear) or at the
            2,100-tic cap. Survival and kill fraction are the Stage 1 gate metrics.
    rooms   Stage 2: 3-8 rooms (training/maps/layout.py) joined by a spanning tree plus a
            few loops, ~half the corridors with an ordinary door, health/ammo/armour
            pickups, no enemies, an exit alcove off the farthest room. Tick cap
            2,100-3,150 by size. Gate: exit rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from training.maps import layout as LY
from training.maps import udmf

GEN_VERSION = "v2-0.1"
REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "maps" / "generated"
POOL_SEED_BASE = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
POOL_SIZE = {"train": 100, "validation": 20, "test": 20}

MONSTERS_COMBAT = [(udmf.THING_ZOMBIEMAN, 0.35), (udmf.THING_SHOTGUNNER, 0.25), (udmf.THING_IMP, 0.30),
                   (udmf.THING_DEMON, 0.10)]


def _weighted(rng: random.Random, table):
    r, acc = rng.random() * sum(w for _, w in table), 0.0
    for item, w in table:
        acc += w
        if r <= acc:
            return item
    return table[-1][0]


def _layout_hash(rects, things) -> str:
    key = json.dumps([sorted([r.x0, r.y0, r.x1, r.y1, r.floor, r.ceil, r.door, r.lock] for r in rects),
                      sorted([int(t.x), int(t.y), t.type] for t in things)])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------ combat
@dataclass
class CombatConfig:
    size_min: int = 640
    size_max: int = 1280
    monsters: tuple = (1, 5)
    pillar_p: float = 0.4
    floors: tuple = (0, 0, 0, 8, 16, 24)
    shotgun_p: float = 0.35
    armor_p: float = 0.2
    ticks: int = 2100
    min_spawn_dist: float = 384.0


def gen_combat(seed: int, cfg: CombatConfig = CombatConfig()) -> dict:
    rng = random.Random(seed)
    W = rng.randrange(cfg.size_min, cfg.size_max + 1, 64)
    H = rng.randrange(cfg.size_min, cfg.size_max + 1, 64)
    xs = [0, int(W * rng.uniform(0.3, 0.4)) // 16 * 16, int(W * rng.uniform(0.6, 0.7)) // 16 * 16, W]
    ys = [0, int(H * rng.uniform(0.3, 0.4)) // 16 * 16, int(H * rng.uniform(0.6, 0.7)) // 16 * 16, H]
    light = rng.randrange(144, 256, 16)
    pillar = rng.random() < cfg.pillar_p
    cells, rects = {}, []
    for i in range(3):
        for j in range(3):
            fl = rng.choice(cfg.floors)
            solid = pillar and (i, j) == (1, 1)
            r = udmf.Rect(xs[i], ys[j], xs[i + 1], ys[j + 1], floor=fl, ceil=fl if solid else fl + 128 + 32,
                          light=light)
            rects.append(r)
            if not solid:
                cells[(i, j)] = r

    def spot(r, margin=40):
        return (rng.randrange(r.x0 + margin, r.x1 - margin + 1, 8), rng.randrange(r.y0 + margin, r.y1 - margin + 1, 8))

    spawn_cell = rng.choice(sorted(cells))
    sx, sy = spot(cells[spawn_cell], 48)
    things = [udmf.Thing(sx, sy, udmf.THING_PLAYER1, rng.randrange(0, 360, 45))]
    n_mon = rng.randint(*cfg.monsters)
    others = [c for c in sorted(cells) if c != spawn_cell]
    for _ in range(n_mon):
        for _try in range(30):
            x, y = spot(cells[rng.choice(others)], 48)
            if math.hypot(x - sx, y - sy) >= cfg.min_spawn_dist or _try == 29:
                break
        things.append(udmf.Thing(x, y, _weighted(rng, MONSTERS_COMBAT), rng.randrange(0, 360, 45)))
    items = [rng.choice([udmf.THING_CLIP, udmf.THING_SHELLS]) for _ in range(rng.randint(1, 3))]
    if rng.random() < cfg.shotgun_p:
        items.append(udmf.THING_SHOTGUN)
    items += [rng.choice([udmf.THING_STIMPACK, udmf.THING_MEDIKIT]) for _ in range(rng.randint(0, 2))]
    if rng.random() < cfg.armor_p:
        items.append(udmf.THING_GREEN_ARMOR)
    for t in items:
        x, y = spot(cells[rng.choice(sorted(cells))], 32)
        things.append(udmf.Thing(x, y, t))
    return {"textmap": udmf.build_textmap(rects, things), "ticks": cfg.ticks, "monsters": n_mon,
            "end_on_clear": True, "layout_hash": _layout_hash(rects, things),
            "meta": {"size": [W, H], "pillar": pillar, "n_items": len(items),
                     "monster_types": sorted({t.type for t in things[1:1 + n_mon]})}}


# ------------------------------------------------------------------------- rooms
PICKUPS = [udmf.THING_STIMPACK, udmf.THING_MEDIKIT, udmf.THING_CLIP, udmf.THING_SHELLS, udmf.THING_GREEN_ARMOR]


def gen_rooms(seed: int) -> dict | None:
    rng = random.Random(seed)
    k = rng.randint(3, 8)
    L = LY.make_layout(rng, k, extra_edge_p=0.15)
    LY.pick_spawn_exit(rng, L)
    for e in sorted(L.edges):
        if rng.random() < 0.5:
            L.doors[e] = 0
    items = []
    for _ in range(rng.randint(1, 4)):
        x, y = LY.spot(rng, L, rng.choice(L.cells), 40)
        items.append(udmf.Thing(x, y, rng.choice(PICKUPS)))
    b = LY.build(L, items)
    if not LY.raster_reachable(b, b["spawn_xy"], b["exit_xy"]):
        return None
    ticks = min(3150, 2100 + 210 * (k - 3))
    return {"textmap": b["textmap"], "ticks": ticks, "monsters": 0, "end_on_clear": False,
            "layout_hash": b["layout_hash"], "exit_xy": b["exit_xy"],
            "meta": {"n_rooms": k, "n_doors": len(L.doors), "n_edges": len(L.edges),
                     "exit_hops": L.hops_from(L.spawn)[L.exit_cell]}}


FAMILIES = {"combat": gen_combat, "rooms": gen_rooms}


# ------------------------------------------------------------------------- pools
def pool_dir(pool: str, root: Path = OUT_ROOT) -> Path:
    return root / pool


def index_path(pool: str, family: str, root: Path = OUT_ROOT) -> Path:
    return pool_dir(pool, root) / f"index_{family}.json"


def load_index(pool: str, family: str, root: Path = OUT_ROOT) -> list[dict]:
    """Schedule entries with absolute WAD paths (index files store repo-relative ones)."""
    entries = json.loads(index_path(pool, family, root).read_text())
    for e in entries:
        p = Path(e["wad"])
        e["wad"] = str(p if p.is_absolute() else REPO / p)
    return entries


def write_pool(family: str, pool: str, n: int, root: Path = OUT_ROOT) -> list[dict]:
    gen = FAMILIES[family]
    out = pool_dir(pool, root)
    out.mkdir(parents=True, exist_ok=True)
    taken = set()
    for other in POOL_SEED_BASE:
        p = index_path(other, family, root)
        if other != pool and p.exists():
            taken |= {e["layout_hash"] for e in json.loads(p.read_text())}
    wad = out / f"{family}.wad"
    try:
        rel = str(wad.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        rel = str(wad)
    index, lumps, seen, seed = [], {}, set(), POOL_SEED_BASE[pool]
    while len(index) < n:
        m = gen(seed)
        seed += 1
        if m is None:                    # failed an offline check (e.g. exit unreachable)
            continue
        h = m["layout_hash"]
        if h in seen or h in taken:
            continue
        seen.add(h)
        name = f"M{len(index) + 1:04d}"
        lumps[name] = m["textmap"]
        index.append({"wad": rel, "map": name, "ticks": m["ticks"], "label": f"{family}/{pool}/{name}",
                      "family": family, "pool": pool, "seed": seed - 1, "generator": GEN_VERSION,
                      "layout_hash": h, "monsters": m.get("monsters", 0), "end_on_clear": m.get("end_on_clear", False),
                      "exit_xy": m.get("exit_xy"),
                      **m["meta"]})
    udmf.write_wad(wad, lumps)
    index_path(pool, family, root).write_text(json.dumps(index, indent=1))
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every family x pool, default sizes")
    ap.add_argument("--family", default="combat", choices=list(FAMILIES))
    ap.add_argument("--pool", default="train", choices=list(POOL_SEED_BASE))
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    jobs = [(f, p) for f in FAMILIES for p in POOL_SEED_BASE] if a.all else [(a.family, a.pool)]
    for fam, pool in jobs:
        idx = write_pool(fam, pool, a.n or POOL_SIZE[pool])
        print(f"{fam:8s} {pool:10s} {len(idx):4d} maps -> {idx[0]['wad']}")


if __name__ == "__main__":
    main()
