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
POOL_SIZE_FAMILY = {"keydoor": {"train": 120, "validation": 30, "test": 30},
                    "mixed": {"train": 160, "validation": 40, "test": 40},
                    "keyeasy": {"train": 60, "validation": 20, "test": 20}}
MIXED_ROTATION = ["rooms"] * 7 + ["lock"] * 8 + ["multilock"] * 5          # 35 / 40 / 25 %   # A/B/C in rotation

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
            "route": {"rects": b["rects"], "keys": [], "exit_xy": b["exit_xy"]},
            "meta": {"n_rooms": k, "n_doors": len(L.doors), "n_edges": len(L.edges),
                     "exit_hops": L.hops_from(L.spawn)[L.exit_cell]}}


# ----------------------------------------------------------------------- keydoor
LOCKS = [udmf.LOCK_RED_CARD, udmf.LOCK_BLUE_CARD, udmf.LOCK_YELLOW_CARD,
         udmf.LOCK_RED_SKULL, udmf.LOCK_BLUE_SKULL, udmf.LOCK_YELLOW_SKULL]


def _attachment(L, c, path_index: dict) -> tuple[int, int]:
    """For a room off the path: (index of the path room its branch hangs from, depth in hops)."""
    prev, q, adj = {c: None}, [c], L.adj()
    while q:
        u = q.pop(0)
        if u in path_index:
            d, v = 0, u
            while prev[v] is not None:
                v, d = prev[v], d + 1
            return path_index[u], d
        for n in adj[u]:
            if n not in prev:
                prev[n] = u
                q.append(n)
    raise ValueError("disconnected")


def gen_keydoor(seed: int, variant: str | None = None) -> dict | None:
    """Stage 3. A tree of 4-8 rooms (no loops, so the lock really gates the exit); one locked
    corridor on the spawn -> exit path; one key (matching card or skull) placed by variant:
        A  key on the way: a room on the path before the locked door
        B  locked door seen first: key down a side branch that hangs off the room in front of
           the locked door (the agent reaches the door before the branch, must come back)
        C  backtrack: key at the dead end of a side branch leaving the path BEFORE the door
    No enemies (they join dependency maps in Stage 4)."""
    rng = random.Random(seed)
    variant = variant or "ABC"[seed % 3]
    k = rng.randint(4, 8)
    L = LY.make_layout(rng, k, extra_edge_p=0.0)
    LY.pick_spawn_exit(rng, L)
    P = L.path(L.spawn, L.exit_cell)
    if len(P) < 3:
        return None
    pidx = {c: i for i, c in enumerate(P)}
    options = []
    for j in range(1, len(P) - 1):                   # lock between P[j] and P[j+1]
        lock_edge = tuple(sorted((P[j], P[j + 1])))
        region = set(L.hops_from(L.spawn, blocked_edges=[lock_edge]))
        off = [c for c in region if c not in pidx]
        if variant == "A":
            cand = [P[i] for i in range(1, j + 1)]
        elif variant == "B":
            cand = [c for c in off if _attachment(L, c, pidx)[0] == j]
        else:
            leaves = [c for c in off if len(L.adj()[c]) == 1 and _attachment(L, c, pidx)[0] < j]
            deepest = max((_attachment(L, c, pidx)[1] for c in leaves), default=0)
            cand = [c for c in leaves if _attachment(L, c, pidx)[1] == deepest]
        options += [(j, lock_edge, c) for c in cand]
    if not options:
        return None
    j, lock_edge, key_room = rng.choice(options)
    lock = rng.choice(LOCKS)
    L.doors[lock_edge] = lock
    for e in sorted(L.edges):
        if e != lock_edge and rng.random() < 0.3:
            L.doors[e] = 0
    kx, ky = LY.spot(rng, L, key_room, 48)
    things = [udmf.Thing(kx, ky, udmf.KEY_FOR_LOCK[lock])]
    for _ in range(rng.randint(0, 3)):
        x, y = LY.spot(rng, L, rng.choice(L.cells), 40)
        things.append(udmf.Thing(x, y, rng.choice(PICKUPS)))
    b = LY.build(L, things)
    if (LY.raster_reachable(b, b["spawn_xy"], b["exit_xy"], locked_open=False)       # lock must gate
            or not LY.raster_reachable(b, b["spawn_xy"], (kx, ky), locked_open=False)  # key reachable
            or not LY.raster_reachable(b, b["spawn_xy"], b["exit_xy"], locked_open=True)):
        return None
    ticks = min(4200, 3150 + 175 * (k - 4))
    return {"textmap": b["textmap"], "ticks": ticks, "monsters": 0, "end_on_clear": False,
            "layout_hash": b["layout_hash"], "exit_xy": b["exit_xy"],
            "route": {"rects": b["rects"], "keys": [[kx, ky, lock]], "exit_xy": b["exit_xy"]},
            "meta": {"variant": variant, "lock": lock, "key_xy": [kx, ky], "n_rooms": k,
                     "n_doors": len(L.doors), "n_locked": 1, "lock_path_index": j, "path_len": len(P),
                     "key_room_on_path": key_room in pidx}}


# ----------------------------------------------------------------------- keyeasy
def gen_keyeasy(seed: int) -> dict | None:
    """Easy key-door warm-up (added after Stage 3 failed twice: the long key -> door -> exit chain
    almost never completed by chance). 2-3 rooms in a line; the key lies in the spawn room; the
    corridor out of the spawn room is locked; the exit is behind it. Tick cap 2,100."""
    rng = random.Random(seed)
    k = rng.choice([2, 2, 3])
    L = LY.make_layout(rng, k, extra_edge_p=0.0)
    LY.pick_spawn_exit(rng, L)
    P = L.path(L.spawn, L.exit_cell)
    if len(P) < 2:
        return None
    lock_edge = tuple(sorted((P[0], P[1])))
    lock = rng.choice(LOCKS)
    L.doors[lock_edge] = lock
    kx, ky = LY.spot(rng, L, L.spawn, 48)
    sx, sy = L.centre(L.spawn)
    if math.hypot(kx - sx, ky - sy) < 64:            # not on top of the player
        return None
    b = LY.build(L, [udmf.Thing(kx, ky, udmf.KEY_FOR_LOCK[lock])])
    if (LY.raster_reachable(b, b["spawn_xy"], b["exit_xy"], locked_open=False)
            or not LY.raster_reachable(b, b["spawn_xy"], b["exit_xy"], locked_open=True)):
        return None
    return {"textmap": b["textmap"], "ticks": 2100, "monsters": 0, "end_on_clear": False,
            "layout_hash": b["layout_hash"], "exit_xy": b["exit_xy"],
            "route": {"rects": b["rects"], "keys": [[kx, ky, lock]], "exit_xy": b["exit_xy"]},
            "meta": {"variant": "E", "lock": lock, "key_xy": [kx, ky], "n_rooms": k, "n_doors": len(L.doors),
                     "n_locked": 1}}


# ------------------------------------------------------------------------- mixed
MONSTERS_MIXED = [(udmf.THING_ZOMBIEMAN, 0.30), (udmf.THING_SHOTGUNNER, 0.25), (udmf.THING_IMP, 0.30),
                  (udmf.THING_DEMON, 0.10), (udmf.THING_CHAINGUNNER, 0.05)]


def gen_mixed(seed: int, variant: str | None = None) -> dict | None:
    """Stage 4 (vanilla-like gauntlet). 4-8 rooms, 2-8 enemies (never in the spawn room), more
    pickups, tick cap 4,200. Variants (by seed): 'rooms' 35% (no lock, a few loops), 'lock' 40%
    (one locked door on the path, key anywhere before it), 'multilock' 25% (two locks of different
    colours in sequence on the path; key 1 before lock 1, key 2 between the locks)."""
    rng = random.Random(seed)
    r = (seed * 0.6180339887) % 1.0
    variant = variant or ("rooms" if r < 0.35 else "lock" if r < 0.75 else "multilock")
    k = rng.randint(4, 8) if variant != "multilock" else rng.randint(6, 8)
    L = LY.make_layout(rng, k, extra_edge_p=0.15 if variant == "rooms" else 0.0)
    LY.pick_spawn_exit(rng, L)
    P = L.path(L.spawn, L.exit_cell)
    n_locks = {"rooms": 0, "lock": 1, "multilock": 2}[variant]
    if len(P) < 2 + n_locks:
        return None
    keys = []
    if n_locks:
        js = sorted(rng.sample(range(len(P) - 1), n_locks))     # lock edges (P[j], P[j+1]) along the path
        if n_locks == 2 and js[1] - js[0] < 1:
            return None
        locks = rng.sample(LOCKS, n_locks)
        edges = [tuple(sorted((P[j], P[j + 1]))) for j in js]
        prev_region: set = set()
        for i, (e, lock) in enumerate(zip(edges, locks)):
            L.doors[e] = lock
            region = set(L.hops_from(L.spawn, blocked_edges=edges[i:]))   # reachable with this and later locks shut
            cand = sorted(region - prev_region) or sorted(region)
            keys.append((lock, rng.choice(cand)))
            prev_region = region
    for e in sorted(L.edges):
        if e not in L.doors and rng.random() < 0.3:
            L.doors[e] = 0
    things, key_xy = [], []
    for lock, room in keys:
        x, y = LY.spot(rng, L, room, 48)
        things.append(udmf.Thing(x, y, udmf.KEY_FOR_LOCK[lock]))
        key_xy.append([x, y, lock])
    others = [c for c in L.cells if c != L.spawn]
    n_mon = rng.randint(2, 8)
    for _ in range(n_mon):
        x, y = LY.spot(rng, L, rng.choice(others), 48)
        things.append(udmf.Thing(x, y, _weighted(rng, MONSTERS_MIXED), rng.randrange(0, 360, 45)))
    items = [rng.choice([udmf.THING_CLIP, udmf.THING_SHELLS]) for _ in range(rng.randint(2, 5))]
    items += [rng.choice([udmf.THING_STIMPACK, udmf.THING_MEDIKIT]) for _ in range(rng.randint(1, 3))]
    if rng.random() < 0.5:
        items.append(udmf.THING_SHOTGUN)
    if rng.random() < 0.3:
        items.append(udmf.THING_GREEN_ARMOR)
    for t in items:
        x, y = LY.spot(rng, L, rng.choice(L.cells), 40)
        things.append(udmf.Thing(x, y, t))
    b = LY.build(L, things)
    if not LY.raster_reachable(b, b["spawn_xy"], b["exit_xy"], locked_open=True):
        return None
    if n_locks and LY.raster_reachable(b, b["spawn_xy"], b["exit_xy"], locked_open=False):
        return None
    return {"textmap": b["textmap"], "ticks": 4200, "monsters": n_mon, "end_on_clear": False,
            "layout_hash": b["layout_hash"], "exit_xy": b["exit_xy"],
            "route": {"rects": b["rects"], "keys": key_xy, "exit_xy": b["exit_xy"]},
            "meta": {"variant": variant, "n_rooms": k, "n_doors": len(L.doors), "n_locked": n_locks,
                     "locks": [lk for lk, _ in keys]}}


# ----------------------------------------------------------------------- freedoom
# Freedoom 2 levels used in Stage 4 (official IWAD, unmodified; tick cap 4,200). Chosen from a
# survey of all 32 (26 Nov): fast enough per env (>= ~110 decisions/s on the dev laptop) and not
# extreme; almost all have several locks. Validation and test levels are never trained on.
FREEDOOM = {"train": ["MAP01", "MAP02", "MAP05", "MAP06", "MAP13", "MAP14", "MAP16", "MAP17", "MAP21",
                      "MAP24", "MAP29", "MAP31", "MAP32"],
            "validation": ["MAP07", "MAP22"], "test": ["MAP04", "MAP19"]}
# monsters present at skill 3 (engine object count, survey 26 Nov)
FREEDOOM_MONSTERS = {"MAP01": 18, "MAP02": 56, "MAP04": 123, "MAP05": 76, "MAP06": 73, "MAP07": 96, "MAP13": 115,
                     "MAP14": 99, "MAP16": 78, "MAP17": 51, "MAP19": 146, "MAP21": 81, "MAP22": 85, "MAP24": 98,
                     "MAP29": 97, "MAP31": 17, "MAP32": 81}


def write_freedoom_pool(pool: str, root: Path = OUT_ROOT) -> list[dict]:
    out = pool_dir(pool, root)
    out.mkdir(parents=True, exist_ok=True)
    index = [{"wad": "freedoom2", "map": m, "ticks": 4200, "label": f"freedoom/{pool}/{m}", "family": "freedoom",
              "pool": pool, "monsters": FREEDOOM_MONSTERS[m], "end_on_clear": False, "exit_xy": None}
             for m in FREEDOOM[pool]]
    index_path(pool, "freedoom", root).write_text(json.dumps(index, indent=1))
    return index


FAMILIES = {"combat": gen_combat, "rooms": gen_rooms, "keydoor": gen_keydoor, "mixed": gen_mixed,
            "keyeasy": gen_keyeasy}


# ------------------------------------------------------------------------- pools
def pool_dir(pool: str, root: Path = OUT_ROOT) -> Path:
    return root / pool


def index_path(pool: str, family: str, root: Path = OUT_ROOT) -> Path:
    return pool_dir(pool, root) / f"index_{family}.json"


def load_index(pool: str, family: str, root: Path = OUT_ROOT) -> list[dict]:
    """Schedule entries with absolute WAD paths (index files store repo-relative ones)."""
    entries = json.loads(index_path(pool, family, root).read_text())
    for e in entries:
        if e["wad"] == "freedoom2":
            from doom_env.wrapper import IWAD
            e["wad"] = IWAD
            continue
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
        if family == "keydoor":
            m = gen(seed, "ABC"[len(index) % 3])
        elif family == "mixed":
            m = gen(seed, MIXED_ROTATION[len(index) % len(MIXED_ROTATION)])
        else:
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
                      "exit_xy": m.get("exit_xy"), "route": m.get("route"),
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
        idx = write_pool(fam, pool, a.n or POOL_SIZE_FAMILY.get(fam, POOL_SIZE)[pool])
        print(f"{fam:8s} {pool:10s} {len(idx):4d} maps -> {idx[0]['wad']}")
    if a.all:
        for pool in POOL_SEED_BASE:
            idx = write_freedoom_pool(pool)
            print(f"{'freedoom':8s} {pool:10s} {len(idx):4d} maps -> Freedoom 2 IWAD {[e['map'] for e in idx]}")


if __name__ == "__main__":
    main()
