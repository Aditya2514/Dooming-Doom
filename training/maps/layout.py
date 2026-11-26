"""Room-graph layouts shared by the multi-room families (Stage 2 rooms, Stage 3 key-door,
Stage 4 mix). Ported and generalised from the v1 generator.

A layout is a connected random set of k cells on a gx x gy grid. Each chosen cell holds one
rectangular room that always covers the cell's centre band, so neighbouring rooms in a row
or column can be joined by a straight corridor along that band. Rooms are joined along a
random spanning tree (dead ends come for free) plus optional extra loops. A corridor may
hold a door (optionally locked). The exit is a walk-over Exit_Normal line at the back of a
lit alcove on a side of the exit room that faces no other room.

Floors are in {0, 8, 16, 24}; a corridor takes the lower floor of its two rooms, so every
step is <= 24 (climbable). Offline checks use a raster BFS over the sectors (allowed: it
never reaches the agent).
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass, field

from training.maps import udmf

DIRS = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}


@dataclass
class Layout:
    gx: int
    gy: int
    cell: int
    corridor_w: int
    cells: list                              # chosen cells, sorted
    rooms: dict                              # cell -> [x0, y0, x1, y1]
    floors: dict                             # cell -> floor height
    edges: set                               # {(a, b)} with a < b
    spawn: tuple = None
    exit_cell: tuple = None
    exit_side: str = None
    doors: dict = field(default_factory=dict)    # edge -> lock (0 = ordinary door); edge absent = open corridor

    # ---------------------------------------------------------------- graph
    def adj(self) -> dict:
        a = {c: [] for c in self.cells}
        for u, v in self.edges:
            a[u].append(v)
            a[v].append(u)
        return a

    def hops_from(self, src, blocked_edges=()) -> dict:
        blocked = {tuple(sorted(e)) for e in blocked_edges}
        dist, q, adj = {src: 0}, deque([src]), self.adj()
        while q:
            c = q.popleft()
            for n in adj[c]:
                if n not in dist and tuple(sorted((c, n))) not in blocked:
                    dist[n] = dist[c] + 1
                    q.append(n)
        return dist

    def path(self, a, b) -> list:
        """Cells on the tree path a -> b (BFS parents)."""
        prev, q, adj = {a: None}, deque([a]), self.adj()
        while q:
            c = q.popleft()
            for n in adj[c]:
                if n not in prev:
                    prev[n] = c
                    q.append(n)
        out, c = [], b
        while c is not None:
            out.append(c)
            c = prev[c]
        return out[::-1]

    def free_sides(self, c) -> list:
        """Sides of cell c with no chosen neighbour (the exit alcove may go there)."""
        return [s for s, (dx, dy) in DIRS.items() if (c[0] + dx, c[1] + dy) not in self.rooms]

    def centre(self, c) -> tuple:
        r = self.rooms[c]
        return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)


# ------------------------------------------------------------------ generation
def grow_cells(rng: random.Random, gx: int, gy: int, k: int) -> list:
    start = (rng.randrange(gx), rng.randrange(gy))
    chosen = {start}
    while len(chosen) < k:
        frontier = sorted({(x + dx, y + dy) for x, y in chosen for dx, dy in DIRS.values()
                           if 0 <= x + dx < gx and 0 <= y + dy < gy} - chosen)
        chosen.add(rng.choice(frontier))
    return sorted(chosen)


def _room(rng, cell, corridor_w, room_min, room_max, cx, cy):
    c = cell
    ox, oy, mid = cx * c, cy * c, c // 2
    band = corridor_w // 2 + 32
    w = rng.randrange(room_min, room_max + 1, 32)
    h = rng.randrange(room_min, room_max + 1, 32)
    lo, hi = max(64 + 128, mid + band - w), min(mid - band, c - 64 - 128 - w)   # 192: room for an alcove
    x0 = rng.randrange(lo, hi + 1, 16) if hi >= lo else max(64, mid - w // 2)
    lo, hi = max(64 + 128, mid + band - h), min(mid - band, c - 64 - 128 - h)
    y0 = rng.randrange(lo, hi + 1, 16) if hi >= lo else max(64, mid - h // 2)
    return [ox + x0, oy + y0, ox + x0 + w, oy + y0 + h]


def make_layout(rng: random.Random, k: int, gx: int = 4, gy: int = 3, cell: int = 896,
                corridor_w: int = 128, room_min: int = 320, room_max: int = 576,
                extra_edge_p: float = 0.15, floors=(0, 8, 16, 24)) -> Layout:
    cells = grow_cells(rng, gx, gy, k)
    rooms = {c: _room(rng, cell, corridor_w, room_min, room_max, *c) for c in cells}
    fl = {c: rng.choice(floors) for c in cells}
    cs = set(cells)

    def nbrs(c):
        return [(c[0] + dx, c[1] + dy) for dx, dy in DIRS.values() if (c[0] + dx, c[1] + dy) in cs]

    start = rng.choice(cells)
    seen, stack, edges = {start}, [start], set()
    while stack:
        c = stack[-1]
        opts = [n for n in nbrs(c) if n not in seen]
        if not opts:
            stack.pop()
            continue
        n = rng.choice(opts)
        seen.add(n)
        edges.add(tuple(sorted((c, n))))
        stack.append(n)
    for c in cells:
        for n in nbrs(c):
            e = tuple(sorted((c, n)))
            if e not in edges and rng.random() < extra_edge_p:
                edges.add(e)
    return Layout(gx, gy, cell, corridor_w, cells, rooms, fl, edges)


def pick_spawn_exit(rng: random.Random, L: Layout, spawn=None) -> None:
    """Spawn: random room (or given). Exit: farthest room (hops) that has a free side."""
    L.spawn = spawn if spawn is not None else rng.choice(L.cells)
    d = L.hops_from(L.spawn)
    cand = [c for c in L.cells if c != L.spawn and L.free_sides(c)]
    far = max(d[c] for c in cand)
    L.exit_cell = rng.choice([c for c in cand if d[c] == far])
    L.exit_side = rng.choice(L.free_sides(L.exit_cell))


# ------------------------------------------------------------------- building
def _split(a0, a1, door):
    if not door:
        return [(a0, a1, False)]
    mid = (a0 + a1) // 2
    return [(a0, mid - 16, False), (mid - 16, mid + 16, True), (mid + 16, a1, False)]


# Share of maps whose exit is a 64x64 walk-over PAD on the exit room's floor (as in the official
# sample map maps/two_rooms.wad) instead of a lit alcove. Set per pool by gen.write_pool; drawn from
# its own random stream, so pools generated with 0 are byte-identical to before.
PAD_EXIT_P = 0.0
PAD = 64


def build(L: Layout, things_extra: list, light_range=(160, 256)) -> dict:
    """Rects, things, exit special and metadata. things_extra: udmf.Thing list (keys, items,
    monsters); the player thing is added here at the spawn room's centre."""
    rng_light = random.Random(hash((tuple(L.cells), L.spawn)) & 0xFFFF)
    rects = [udmf.Rect(*L.rooms[c], floor=L.floors[c], ceil=L.floors[c] + 128,
                       light=rng_light.randrange(*light_range, 16)) for c in L.cells]
    half = L.corridor_w // 2
    for a, b in sorted(L.edges):
        e = (a, b)
        fl = min(L.floors[a], L.floors[b])
        lock = L.doors.get(e)
        ra, rb = L.rooms[a], L.rooms[b]
        if a[1] == b[1]:                      # horizontal neighbours, a left of b
            yc = a[1] * L.cell + L.cell // 2
            for s0, s1, is_door in _split(ra[2], rb[0], lock is not None):
                rects.append(udmf.Rect(s0, yc - half, s1, yc + half, floor=fl, ceil=fl + 128,
                                       door=is_door, lock=(lock or 0) if is_door else 0))
        else:                                  # vertical neighbours, a below b
            xc = a[0] * L.cell + L.cell // 2
            for s0, s1, is_door in _split(ra[3], rb[1], lock is not None):
                rects.append(udmf.Rect(xc - half, s0, xc + half, s1, floor=fl, ceil=fl + 128,
                                       door=is_door, lock=(lock or 0) if is_door else 0))

    r, ef, (x, y) = L.rooms[L.exit_cell], L.floors[L.exit_cell], L.exit_cell
    xc, yc = x * L.cell + L.cell // 2, y * L.cell + L.cell // 2
    ex = {"special": udmf.SPECIAL_EXIT_NORMAL, "playercross": True}
    s = L.exit_side
    rng_pad = random.Random(f"pad|{L.cells}|{L.spawn}|{L.exit_cell}")
    pad = PAD_EXIT_P > 0 and rng_pad.random() < PAD_EXIT_P
    if pad:
        # split the exit room into up to 4 strips around a 64x64 pad; exit lines on the pad's sides
        k = L.cells.index(L.exit_cell)
        room = rects[k]
        x0, y0, x1, y1 = room.x0, room.y0, room.x1, room.y1
        px0 = rng_pad.randrange(x0, x1 - PAD + 1, 16)
        py0 = rng_pad.randrange(y0, y1 - PAD + 1, 16)
        px1, py1 = px0 + PAD, py0 + PAD
        pieces = [(x0, y0, px0, y1), (px1, y0, x1, y1), (px0, y0, px1, py0), (px0, py1, px1, y1), (px0, py0, px1, py1)]
        rects[k:k + 1] = [udmf.Rect(a, b, c, d, floor=room.floor, ceil=room.ceil, light=room.light)
                          for a, b, c, d in pieces if c > a and d > b]
        sides = [((px0, py0), (px1, py0)), ((px1, py0), (px1, py1)), ((px1, py1), (px0, py1)), ((px0, py1), (px0, py0))]
        specials = [udmf.LineSpecial(a, b, ex) for a, b in sides
                    if not ((a[0] == b[0] and a[0] in (x0, x1)) or (a[1] == b[1] and a[1] in (y0, y1)))]
        line, exit_xy = sides[0], (px0 + PAD // 2, py0 + PAD // 2)
    elif s == "E":
        al = [(r[2], yc - half, r[2] + 64, yc + half), (r[2] + 64, yc - half, r[2] + 128, yc + half)]
        line, exit_xy = ((r[2] + 64, yc - half), (r[2] + 64, yc + half)), (r[2] + 96, yc)
    elif s == "W":
        al = [(r[0] - 64, yc - half, r[0], yc + half), (r[0] - 128, yc - half, r[0] - 64, yc + half)]
        line, exit_xy = ((r[0] - 64, yc - half), (r[0] - 64, yc + half)), (r[0] - 96, yc)
    elif s == "N":
        al = [(xc - half, r[3], xc + half, r[3] + 64), (xc - half, r[3] + 64, xc + half, r[3] + 128)]
        line, exit_xy = ((xc - half, r[3] + 64), (xc + half, r[3] + 64)), (xc, r[3] + 96)
    else:
        al = [(xc - half, r[1] - 64, xc + half, r[1]), (xc - half, r[1] - 128, xc + half, r[1] - 64)]
        line, exit_xy = ((xc - half, r[1] - 64), (xc + half, r[1] - 64)), (xc, r[1] - 96)
    if not pad:
        for a in al:
            rects.append(udmf.Rect(*a, floor=ef, ceil=ef + 128, light=255))
        specials = [udmf.LineSpecial(line[0], line[1], ex)]

    sx, sy = L.centre(L.spawn)
    things = [udmf.Thing(sx, sy, udmf.THING_PLAYER1, rng_light.choice([0, 90, 180, 270]))] + list(things_extra)
    meta_rects = [[q.x0, q.y0, q.x1, q.y1, q.floor, q.ceil, q.door, q.lock] for q in rects]
    key = json.dumps([sorted(meta_rects), sorted([int(t.x), int(t.y), t.type] for t in things)])
    return {"textmap": udmf.build_textmap(rects, things, specials),
            "layout_hash": hashlib.sha1(key.encode()).hexdigest()[:16],
            "rects": meta_rects, "exit_line": [list(line[0]), list(line[1])], "exit_xy": list(exit_xy),
            "spawn_xy": [sx, sy], "things": [[t.x, t.y, t.type] for t in things]}


def spot(rng: random.Random, L: Layout, c, margin: int = 48) -> tuple:
    r = L.rooms[c]
    return (rng.randrange(r[0] + margin, r[2] - margin + 1, 8), rng.randrange(r[1] + margin, r[3] - margin + 1, 8))


# ------------------------------------------------------------ offline checks
def raster_reachable(built: dict, start_xy, goal_xy, locked_open=True, res: int = 16) -> bool:
    """Flood fill from start over sectors whose floors differ by <= 24. Ordinary doors count
    as openable; locked doors only if locked_open. Goal: the cell containing goal_xy."""
    rects = built["rects"]
    x_min = min(r[0] for r in rects)
    y_min = min(r[1] for r in rects)
    floor = {}
    for x0, y0, x1, y1, fl, ce, door, lock in rects:
        if ce <= fl and not door:
            continue                           # solid pillar
        if door and lock and not locked_open:
            continue
        for i in range((x0 - x_min) // res, (x1 - x_min) // res):
            for j in range((y0 - y_min) // res, (y1 - y_min) // res):
                floor[(i, j)] = fl
    cell = lambda x, y: (int((x - x_min) // res), int((y - y_min) // res))  # noqa: E731
    s, g = cell(*start_xy), cell(*goal_xy)
    if s not in floor or g not in floor:
        return False
    seen, q = {s}, deque([s])
    while q:
        c = q.popleft()
        if c == g:
            return True
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in floor and n not in seen and abs(floor[n] - floor[c]) <= 24:
                seen.add(n)
                q.append(n)
    return False
