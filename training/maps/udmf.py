"""Minimal UDMF (zdoom namespace) map writer.

Maps are built from axis-aligned rectangular sectors. Edges shared by two
rectangles become two-sided lines automatically; everything else is a
one-sided wall whose front faces the sector interior. ZDoom builds nodes for
UDMF maps itself, so no node builder is needed.

Ported from the v1 repo; v2 adds locked doors (Door_LockedRaise, which the official
wrapper reports as `locked_door` ray hits and door_lock > 0), keys, pickups and
solid pillars. Line specials are ZDoom/Hexen numbers, which the wrapper parses
(env/doom_env/mapinfo.py HEXEN_DOORS: 12 Door_Raise, 13 Door_LockedRaise with the
lock in arg3).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

# Hexen-format line specials (zdoom namespace)
SPECIAL_DOOR_RAISE = 12
SPECIAL_DOOR_LOCKED_RAISE = 13  # Door_LockedRaise(tag, speed, delay, lock, lighttag)
SPECIAL_EXIT_NORMAL = 243

# ZDoom lock numbers (LOCKDEFS, Doom): 1 red card, 2 blue card, 3 yellow card,
# 4 red skull, 5 blue skull, 6 yellow skull
LOCK_RED_CARD, LOCK_BLUE_CARD, LOCK_YELLOW_CARD = 1, 2, 3
LOCK_RED_SKULL, LOCK_BLUE_SKULL, LOCK_YELLOW_SKULL = 4, 5, 6
SPECIAL_TELEPORT = 70          # Teleport(tid): to the thing with that id

# Thing type numbers (Doom 2 / Freedoom 2)
THING_PLAYER1 = 1
THING_ZOMBIEMAN = 3004
THING_SHOTGUNNER = 9
THING_IMP = 3001
THING_DEMON = 3002
THING_CHAINGUNNER = 65
THING_RED_CARD = 13
THING_BLUE_CARD = 5
THING_YELLOW_CARD = 6
THING_RED_SKULL = 38
THING_BLUE_SKULL = 40
THING_YELLOW_SKULL = 39
THING_CLIP = 2007
THING_SHELLS = 2008
THING_SHOTGUN = 2001
THING_STIMPACK = 2011
THING_MEDIKIT = 2012
THING_GREEN_ARMOR = 2018
THING_BARREL = 2035

# key thing -> the lock number it opens
KEY_FOR_LOCK = {LOCK_RED_CARD: THING_RED_CARD, LOCK_BLUE_CARD: THING_BLUE_CARD, LOCK_YELLOW_CARD: THING_YELLOW_CARD,
                LOCK_RED_SKULL: THING_RED_SKULL, LOCK_BLUE_SKULL: THING_BLUE_SKULL,
                LOCK_YELLOW_SKULL: THING_YELLOW_SKULL}
LOCK_DOOR_TEX = {1: "DOORRED", 2: "DOORBLU", 3: "DOORYEL", 4: "DOORRED2", 5: "DOORBLU2", 6: "DOORYEL2"}
THING_TELEPORT_DEST = 14       # give it extra={"id": tid}


# Visual variety for the RGB branch (Doom II texture names; all exist in the Freedoom 2 IWAD used at
# evaluation). Drawn from a separate random stream, so layouts are unchanged. No red floors: the
# exit pad uses EXIT_FLOOR_TEX (GATE3, as in the official sample map maps/two_rooms.wad).
WALL_TEXTURES = ("STARTAN2", "STARTAN3", "STONE2", "STONE3", "BROWN1", "GRAY1", "BRONZE1", "ICKWALL1",
                 "BSTONE1", "TEKGREN1")
FLOOR_TEXTURES = ("FLOOR4_8", "FLOOR0_1", "FLOOR5_1", "FLAT5_4", "FLAT1", "FLOOR7_1", "FLAT14", "FLOOR1_1",
                  "FLAT5_1", "GRNROCK")
EXIT_FLOOR_TEX = "GATE3"


@dataclass
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int
    floor: int = 0
    ceil: int = 128
    light: int = 192
    door: bool = False          # ceiling starts at floor; faces get Door_Raise
    lock: int = 0               # with door=True: Door_LockedRaise with this lock number
    starts_open: bool = False   # with door=True: the door sector starts open (ceiling at `ceil`)
    extra: dict = field(default_factory=dict)  # raw UDMF sector fields
    floor_tex: str = "FLOOR4_8"
    ceil_tex: str = "CEIL3_5"


@dataclass
class Thing:
    x: float
    y: float
    type: int
    angle: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class LineSpecial:
    """Applied to every generated segment lying on segment (a, b)."""
    a: tuple[int, int]
    b: tuple[int, int]
    fields: dict


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _block(kind: str, fields: dict) -> str:
    body = "".join(f"  {k} = {_fmt(v)};\n" for k, v in fields.items())
    return f"{kind}\n{{\n{body}}}\n"


def _on_segment(p, a, b) -> bool:
    (px, py), (ax, ay), (bx, by) = p, a, b
    if (bx - ax) * (py - ay) - (by - ay) * (px - ax) != 0:
        return False
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


def build_textmap(rects: list[Rect], things: list[Thing],
                  specials: list[LineSpecial] | None = None,
                  wall_tex: str = "STARTAN2", door_tex: str = "BIGDOOR2",
                  auto_doors: bool = False) -> str:
    """auto_doors: door faces also open when the player bumps into them
    (playerpush) -- the curriculum's 'auto-doors' assistance variant."""
    specials = specials or []
    corners = set()
    for r in rects:
        assert r.x0 < r.x1 and r.y0 < r.y1, r
        corners |= {(r.x0, r.y0), (r.x0, r.y1), (r.x1, r.y0), (r.x1, r.y1)}

    # segment key (sorted endpoints) -> list of (sector index, v1, v2) with
    # v1->v2 in that sector's clockwise order (right side = interior)
    segs: dict[tuple, list] = {}
    for si, r in enumerate(rects):
        ring = [(r.x0, r.y0), (r.x0, r.y1), (r.x1, r.y1), (r.x1, r.y0)]
        for k in range(4):
            a, b = ring[k], ring[(k + 1) % 4]
            pts = [p for p in corners if _on_segment(p, a, b)]
            pts.sort(key=lambda p: abs(p[0] - a[0]) + abs(p[1] - a[1]))
            for p, q in zip(pts, pts[1:]):
                segs.setdefault(tuple(sorted((p, q))), []).append((si, p, q))

    verts: dict[tuple, int] = {}

    def vid(p):
        if p not in verts:
            verts[p] = len(verts)
        return verts[p]

    sidedefs, linedefs = [], []
    for key in sorted(segs):
        owners = segs[key]
        if len(owners) > 2:
            raise ValueError(f"segment {key} shared by {len(owners)} sectors")
        fields: dict = {}
        if len(owners) == 1:
            si, p, q = owners[0]
            sidedefs.append({"sector": si, "texturemiddle": wall_tex})
            fields.update(v1=vid(p), v2=vid(q), sidefront=len(sidedefs) - 1,
                          blocking=True)
        else:
            (sa, pa, qa), (sb, _, _) = owners
            # door faces: back side must be the door sector
            if rects[sa].door and not rects[sb].door:
                (sa, pa, qa), (sb, _, _) = owners[1], owners[0]
            front, back = sa, sb
            tex = (LOCK_DOOR_TEX.get(rects[back].lock, door_tex) if rects[back].door else wall_tex)
            sidedefs.append({"sector": front, "texturetop": tex,
                             "texturebottom": wall_tex, "texturemiddle": "-"})
            sidedefs.append({"sector": back, "texturetop": wall_tex,
                             "texturebottom": wall_tex, "texturemiddle": "-"})
            fields.update(v1=vid(pa), v2=vid(qa), sidefront=len(sidedefs) - 2,
                          sideback=len(sidedefs) - 1, twosided=True)
            if rects[back].door and not rects[front].door:
                if rects[back].lock:
                    fields.update(special=SPECIAL_DOOR_LOCKED_RAISE, arg0=0, arg1=16, arg2=150,
                                  arg3=rects[back].lock, playeruse=True, repeatspecial=True)
                else:
                    fields.update(special=SPECIAL_DOOR_RAISE, arg0=0, arg1=16,
                                  arg2=150, playeruse=True, repeatspecial=True)
                if auto_doors:
                    fields["playerpush"] = True
        a, b = key
        for s in specials:
            if _on_segment(a, s.a, s.b) and _on_segment(b, s.a, s.b):
                fields.update(s.fields)
        linedefs.append(fields)

    out = ['namespace = "zdoom";\n']
    for t in things:
        f = {"x": float(t.x), "y": float(t.y), "type": t.type, "angle": t.angle,
             "skill1": True, "skill2": True, "skill3": True, "skill4": True,
             "skill5": True, "single": True, "coop": True, "dm": True}
        f.update(t.extra)
        out.append(_block("thing", f))
    for p, _ in sorted(verts.items(), key=lambda kv: kv[1]):
        out.append(_block("vertex", {"x": float(p[0]), "y": float(p[1])}))
    for l in linedefs:
        out.append(_block("linedef", l))
    for s in sidedefs:
        out.append(_block("sidedef", s))
    for r in rects:
        f = {"heightfloor": r.floor,
             "heightceiling": r.floor if (r.door and not r.starts_open) else r.ceil,
             "texturefloor": r.floor_tex, "textureceiling": r.ceil_tex,
             "lightlevel": r.light}
        f.update(r.extra)
        out.append(_block("sector", f))
    return "".join(out)


def write_wad(path: str | Path, maps: dict[str, str]) -> Path:
    """Write a PWAD containing one or more UDMF maps {"MAP01": textmap}."""
    lumps: list[tuple[str, bytes]] = []
    for name, textmap in maps.items():
        lumps += [(name, b""), ("TEXTMAP", textmap.encode("ascii")),
                  ("ENDMAP", b"")]
    data = bytearray()
    directory = []
    offset = 12
    for name, blob in lumps:
        directory.append((offset, len(blob), name))
        data += blob
        offset += len(blob)
    header = struct.pack("<4sii", b"PWAD", len(lumps), offset)
    dir_bytes = b"".join(struct.pack("<ii8s", o, n, nm.encode("ascii"))
                         for o, n, nm in directory)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + bytes(data) + dir_bytes)
    return path


def corridor(sections: list[dict], width: int = 128, start_len: int = 256,
             end_len: int = 256, exit_at_end: bool = False,
             things: list[Thing] | None = None) -> tuple[list[Rect], list[Thing], list[LineSpecial]]:
    """A straight corridor along +x: start room, test sections, end room.

    Each section dict: {"len": int, "floor": int, "ceil": int, **Rect kwargs}.
    The player spawns at x=64, facing +x (angle 0). If exit_at_end, a
    walk-over Exit_Normal line sits 64 units into the end room.
    """
    rects = [Rect(0, 0, start_len, width)]
    x = start_len
    for s in sections:
        s = dict(s)
        ln = s.pop("len")
        rects.append(Rect(x, 0, x + ln, width, **s))
        x += ln
    specials = []
    if exit_at_end:
        rects.append(Rect(x, 0, x + 64, width))
        x += 64
        specials.append(LineSpecial((x, 0), (x, width),
                                    {"special": SPECIAL_EXIT_NORMAL,
                                     "playercross": True}))
    rects.append(Rect(x, 0, x + end_len, width))
    ts = [Thing(64, width // 2, THING_PLAYER1, 0)] + list(things or [])
    return rects, ts, specials
