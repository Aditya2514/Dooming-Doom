"""Generate the animated README SVGs (hero, pipeline, architecture, results, ablations) next to this script.

    python docs/media/make_svgs.py

The gameplay GIFs come from runs/videos/final_v2/*.mp4 via ffmpeg (bundled by imageio-ffmpeg):
    ffmpeg -ss 0 -t 15 -i clip.mp4 -vf "fps=8,scale=256:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=none" -loop 0 out.gif
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

SANS = "'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, Menlo, monospace"

BG = "#0d1117"
CARD = "#11151c"
CARD2 = "#161b22"
LINE = "#30363d"
INK = "#e6edf3"
INK2 = "#9da7b3"
INK3 = "#6e7681"
RED = "#ff4d2e"
ORANGE = "#ff8a3d"
BLUE = "#338ced"
ORANGE_S = "#e76624"
AQUA = "#00a888"
YELLOW = "#f2b134"
VIOLET = "#a68bff"
GREEN = "#3ddc97"
ENEMY = "#ff5252"


def svg(w: int, h: int, body: str, title: str, extra_style: str = "") -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}">
<title>{title}</title>
<style>
  .sans {{ font-family: {SANS}; }}
  .mono {{ font-family: {MONO}; }}
  @keyframes flow {{ to {{ stroke-dashoffset: -24; }} }}
  .flow {{ stroke-dasharray: 6 6; animation: flow 1s linear infinite; }}
  @keyframes fadein {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
  @keyframes blink {{ 0%, 49% {{ opacity: 1; }} 50%, 100% {{ opacity: 0; }} }}
  @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .35; }} }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
{extra_style}
</style>
{body}
</svg>
"""


def card(x, y, w, h, fill=CARD, stroke=LINE, rx=14, extra=""):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" {extra}/>'


def text(x, y, s, size=14, fill=INK, cls="sans", weight=400, anchor="start", extra=""):
    return (f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{s}</text>')


# ---------------------------------------------------------------- hero

def raycast():
    """64 rays from the agent in a small hand-made room; returns [(dist01, type)]. Forward = +y, left = -x."""
    segs = []  # (x1,y1,x2,y2,type)

    def wall(a, b, t="wall"):
        segs.append((*a, *b, t))

    # room x in [-420, 520], y in [-320, 400]; left wall has an opening y in [40, 230] into a long corridor
    wall((-420, -320), (520, -320))
    wall((520, -320), (520, -210))
    wall((520, -210), (520, -70), "locked")
    wall((520, -70), (520, 400))
    wall((520, 400), (260, 400))
    wall((260, 400), (100, 400), "door")
    wall((100, 400), (-420, 400))
    wall((-420, 400), (-420, 230))
    wall((-420, 40), (-420, -320))
    wall((-420, 230), (-1400, 230))
    wall((-420, 40), (-1400, 40))
    # pillar
    for a, b in [((-200, 130), (-110, 130)), ((-110, 130), (-110, 220)), ((-110, 220), (-200, 220)), ((-200, 220), (-200, 130))]:
        wall(a, b)
    circles = [((300, 250), 34, "enemy"), ((-230, -190), 24, "item"), ((330, -150), 30, "enemy")]
    K = 1.7
    segs = [(a * K, b * K, c * K, d * K, t) for a, b, c, d, t in segs]
    circles = [((x * K, y * K), r * K, t) for (x, y), r, t in circles]
    out = []
    for i in range(64):
        th = 2 * math.pi * i / 64
        dx, dy = -math.sin(th), math.cos(th)
        best, typ = 1024.0, "none"
        for x1, y1, x2, y2, t in segs:
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            tt = (x1 * ey - y1 * ex) / den
            uu = (x1 * dy - y1 * dx) / den
            if 0 < tt < best and 0 <= uu <= 1:
                best, typ = tt, t
        for (cx, cy), r, t in circles:
            b = dx * cx + dy * cy
            c = cx * cx + cy * cy - r * r
            disc = b * b - c
            if disc >= 0:
                tt = b - math.sqrt(disc)
                if 0 < tt < best:
                    best, typ = tt, t
        out.append((best / 1024.0, typ))
    return out


def hero():
    W, H = 1280, 420
    cx, cy, R = 1010, 212, 168
    rays = raycast()
    col = {"wall": "#8b96a5", "door": YELLOW, "locked": VIOLET, "enemy": ENEMY, "item": GREEN, "none": None}
    period = 4.0
    ray_svg, dots = [], []
    for i, (d, t) in enumerate(rays):
        th = 2 * math.pi * i / 64
        ex, ey = cx - math.sin(th) * d * R, cy - math.cos(th) * d * R
        L = d * R
        begin = period * i / 64
        stroke = col[t] or "#4a5361"
        ray_svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{stroke}" stroke-width="1.3" '
            f'stroke-linecap="round" stroke-dasharray="{L:.1f}" stroke-dashoffset="{L:.1f}" opacity=".38">'
            f'<animate attributeName="stroke-dashoffset" from="{L:.1f}" to="0" dur=".5s" begin="{0.6 + i * 0.02:.2f}s" fill="freeze"/>'
            f'<animate attributeName="opacity" values="1;.38;.38" keyTimes="0;.35;1" dur="{period}s" '
            f'begin="{1.9 + begin:.3f}s" repeatCount="indefinite"/></line>')
        if t != "none":
            r = 3.6 if t in ("enemy", "item", "door", "locked") else 2.2
            anim = ('<animate attributeName="r" values="3.6;5.4;3.6" dur="1.2s" repeatCount="indefinite"/>'
                    if t == "enemy" else "")
            dots.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="{r}" fill="{col[t]}" opacity="0">'
                        f'<animate attributeName="opacity" from="0" to="{1 if t != "wall" else .75}" dur=".3s" '
                        f'begin="{1.0 + i * 0.02:.2f}s" fill="freeze"/>{anim}</circle>')

    # sweep wedge (leading edge at angle 0 = up; trail on the clockwise side because rays advance counter-clockwise)
    wedges = []
    n = 8
    for k in range(n):
        a0, a1 = math.radians(k * 4), math.radians((k + 1) * 4)
        x0, y0 = cx + math.sin(a0) * R, cy - math.cos(a0) * R
        x1, y1 = cx + math.sin(a1) * R, cy - math.cos(a1) * R
        op = 0.20 * (1 - k / n)
        wedges.append(f'<path d="M{cx},{cy} L{x0:.1f},{y0:.1f} A{R},{R} 0 0 1 {x1:.1f},{y1:.1f} Z" fill="{ORANGE}" opacity="{op:.3f}"/>')
    sweep = (f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur=".6s" begin="1.9s" fill="freeze"/>'
             f'<g>{"".join(wedges)}<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R}" stroke="{ORANGE}" stroke-width="1.6" opacity=".9"/>'
             f'<animateTransform attributeName="transform" type="rotate" from="0 {cx} {cy}" to="-360 {cx} {cy}" '
             f'dur="{period}s" begin="1.9s" repeatCount="indefinite"/></g></g>')

    rings = "".join(f'<circle cx="{cx}" cy="{cy}" r="{R * f:.1f}" fill="none" stroke="{LINE}" stroke-width="1" '
                    f'stroke-dasharray="{"2 4" if f < 1 else "none"}"/>' for f in (0.25, 0.5, 0.75, 1.0))
    agent = (f'<path d="M{cx},{cy - 11} L{cx + 8},{cy + 8} L{cx},{cy + 4} L{cx - 8},{cy + 8} Z" fill="{INK}"/>'
             f'<circle cx="{cx}" cy="{cy}" r="18" fill="none" stroke="{ORANGE}" stroke-width="1.2" opacity=".6">'
             f'<animate attributeName="r" values="12;26;12" dur="2.4s" repeatCount="indefinite"/>'
             f'<animate attributeName="opacity" values=".7;0;.7" dur="2.4s" repeatCount="indefinite"/></circle>')

    legend_items = [("wall", "#8b96a5"), ("door", YELLOW), ("locked door", VIOLET), ("enemy", ENEMY), ("item", GREEN)]
    lx = cx - 182
    legend = []
    for name, c in legend_items:
        legend.append(f'<circle cx="{lx}" cy="{cy + R + 26}" r="4" fill="{c}"/>'
                      + text(lx + 9, cy + R + 30, name, 12, INK2, "mono"))
        lx += 28 + len(name) * 7.6
    cap = text(cx, 30, "64 RAYS × 360° · WHAT THE AGENT ACTUALLY SEES", 11, INK3, "mono", 600, "middle",
               'letter-spacing="1.5"')

    grid = (f'<defs><pattern id="g" width="32" height="32" patternUnits="userSpaceOnUse">'
            f'<path d="M32 0H0V32" fill="none" stroke="#ffffff" stroke-opacity=".035"/></pattern>'
            f'<radialGradient id="glow" cx="78%" cy="50%" r="55%"><stop offset="0" stop-color="{RED}" stop-opacity=".16"/>'
            f'<stop offset="1" stop-color="{RED}" stop-opacity="0"/></radialGradient>'
            f'<linearGradient id="title" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffc078"/>'
            f'<stop offset=".45" stop-color="{RED}"/><stop offset="1" stop-color="#b3121b"/></linearGradient>'
            f'<linearGradient id="fade" x1="0" x2="1"><stop offset="0" stop-color="{RED}"/><stop offset="1" stop-color="{RED}" stop-opacity="0"/></linearGradient>'
            f'<clipPath id="clip"><rect width="{W}" height="{H}" rx="18"/></clipPath></defs>')

    left = [
        f'<g style="animation: fadein .8s ease-out both">',
        f'<rect x="64" y="70" width="318" height="28" rx="14" fill="none" stroke="{RED}" stroke-opacity=".55"/>',
        text(223, 89, "YUGAANTAR 2026 · DEVFORGE · TRACK 1", 12, ORANGE, "mono", 600, "middle", 'letter-spacing="1.2"'),
        text(60, 186, "DOOMING DOOM", 84, "url(#title)", "sans", 900, extra='letter-spacing="1"'),
        f'<rect x="66" y="206" width="420" height="3" fill="url(#fade)"/>',
        text(64, 250, "A sensor-only DOOM agent, distilled from", 25, INK, "sans", 500),
        text(64, 282, "a privileged teacher with DAgger.", 25, INK, "sans", 500),
        "</g>",
        f'<g style="animation: fadein .8s .5s ease-out both">',
        text(64, 334, f'rays · memory · RGB exit detector · 993k params · 4.7 ms / decision <tspan fill="{ORANGE}" style="animation: blink 1.1s steps(1) infinite">▋</tspan>', 15, INK2, "mono"),
        "</g>",
    ]
    body = (grid + f'<g clip-path="url(#clip)"><rect width="{W}" height="{H}" fill="{BG}"/>'
            f'<rect width="{W}" height="{H}" fill="url(#g)"/><rect width="{W}" height="{H}" fill="url(#glow)"/>'
            + "".join(left) + cap + rings + "".join(ray_svg) + sweep + "".join(dots) + agent + "".join(legend)
            + f'</g><rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="18" fill="none" stroke="{LINE}"/>')
    return svg(W, H, body, "Dooming Doom: a sensor-only DOOM agent distilled from a privileged teacher")


# ---------------------------------------------------------------- pipeline

def bullet_list(x, y, items, size=14, gap=24, fill=INK2, dot=ORANGE):
    out = []
    for k, s in enumerate(items):
        out.append(f'<circle cx="{x + 3}" cy="{y + k * gap - 5}" r="2.5" fill="{dot}"/>' + text(x + 14, y + k * gap, s, size, fill))
    return "".join(out)


def pipeline():
    W, H = 1280, 560
    parts = [f'<rect width="{W}" height="{H}" rx="18" fill="{BG}" stroke="{LINE}"/>']
    # zones
    parts.append(f'<rect x="24" y="22" width="828" height="34" rx="8" fill="{RED}" fill-opacity=".07"/>')
    parts.append(text(40, 44, "TRAINING ONLY · ground truth allowed", 12, ORANGE, "mono", 600, extra='letter-spacing="1.3"'))
    parts.append(f'<rect x="872" y="22" width="384" height="34" rx="8" fill="{BLUE}" fill-opacity=".08"/>')
    parts.append(text(888, 44, "EVALUATION · official sensors only", 12, BLUE, "mono", 600, extra='letter-spacing="1.3"'))
    parts.append(f'<line x1="862" y1="22" x2="862" y2="400" stroke="{LINE}" stroke-dasharray="4 6"/>')

    # card 1: teacher
    x, y, w, h = 24, 76, 340, 300
    parts.append(card(x, y, w, h))
    parts.append(text(x + 22, y + 38, "01", 13, ORANGE, "mono", 700))
    parts.append(text(x + 52, y + 38, "Privileged teacher", 20, INK, "sans", 700))
    parts.append(text(x + 22, y + 62, "training/expert.py", 12, INK3, "mono"))
    parts.append(bullet_list(x + 22, y + 102, [
        "sees the floor plan + its true pose",
        "BFS planner on a 16-unit grid",
        "but decisions stay imitable:",
        "  explores where its rays haven't swept",
        "  keys once seen, exit once on camera",
        "  fights &amp; dodges what is visible",
    ], 14, 26))
    parts.append(f'<rect x="{x + 22}" y="{y + 252}" width="{w - 44}" height="30" rx="8" fill="{CARD2}"/>')
    parts.append(text(x + w / 2, y + 272, "100% exits on validation maps", 13, INK, "mono", 600, "middle"))

    # card 2: DAgger loop
    x2, w2 = 404, 432
    parts.append(card(x2, y, w2, h))
    parts.append(text(x2 + 22, y + 38, "02", 13, ORANGE, "mono", 700))
    parts.append(text(x2 + 52, y + 38, "DAgger: learn while playing", 20, INK, "sans", 700))
    parts.append(text(x2 + 22, y + 62, "training/dagger.py", 12, INK3, "mono"))
    lcx, lcy, lr = x2 + w2 / 2, y + 176, 74
    parts.append(f'<circle cx="{lcx}" cy="{lcy}" r="{lr}" fill="none" stroke="{LINE}" stroke-width="2"/>')
    parts.append(f'<circle cx="{lcx}" cy="{lcy}" r="{lr}" fill="none" stroke="{ORANGE}" stroke-width="2" class="flow" opacity=".8"/>')
    for k in range(3):
        parts.append(f'<circle r="5" fill="{ORANGE}"><animateMotion dur="6s" repeatCount="indefinite" begin="{-2 * k}s" '
                     f'path="M{lcx},{lcy - lr} a{lr},{lr} 0 1,1 0,{2 * lr} a{lr},{lr} 0 1,1 0,{-2 * lr}"/></circle>')
    nodes = [(lcx, lcy - lr, "student drives", "middle", 0, -14),
             (lcx + lr, lcy, "teacher labels", "start", 14, -2),
             (lcx + lr, lcy, "every state", "start", 14, 15),
             (lcx, lcy + lr, "1.5M-state dataset", "middle", 0, 26),
             (lcx - lr, lcy, "train GRU on", "end", -14, -2),
             (lcx - lr, lcy, "whole episodes", "end", -14, 15)]
    seen = set()
    for nx, ny, s, anc, ox, oy in nodes:
        if (nx, ny) not in seen:
            parts.append(f'<circle cx="{nx}" cy="{ny}" r="7" fill="{BG}" stroke="{ORANGE}" stroke-width="2"/>')
            seen.add((nx, ny))
        parts.append(text(nx + ox, ny + oy, s, 13, INK2, "sans", 500, anc))
    parts.append(text(lcx, lcy - 4, "β: 0.5 → 0.1", 13, INK, "mono", 600, "middle"))
    parts.append(text(x2 + w2 / 2, y + 284, "+ RGB exit detector pre-trained on teacher frames", 12, INK3, "mono", 500, "middle"))
    parts.append(text(lcx, lcy + 15, "teacher share", 11, INK3, "mono", 400, "middle"))

    # card 3: student
    x3, w3 = 884, 372
    parts.append(card(x3, y, w3, h, stroke=BLUE, extra='stroke-opacity=".6"'))
    parts.append(text(x3 + 22, y + 38, "03", 13, BLUE, "mono", 700))
    parts.append(text(x3 + 52, y + 38, "Sensor-only student", 20, INK, "sans", 700))
    parts.append(text(x3 + 22, y + 62, "submission/agent.py · ships", 12, INK3, "mono"))
    parts.append(bullet_list(x3 + 22, y + 102, [
        "64 rays · ≤32 visible entities · vitals",
        "+ its own dead-reckoned spatial map",
        "+ RGB exit detector (30×40 frame)",
        "GRU 256 → 16 macro-actions · 993k params",
        "~4.7 ms per decision on 1 CPU thread",
        "no coordinates, no ground truth",
    ], 14, 26, dot=BLUE))
    parts.append(f'<rect x="{x3 + 22}" y="{y + 252}" width="{w3 - 44}" height="30" rx="8" fill="{BLUE}" fill-opacity=".14"/>')
    parts.append(text(x3 + w3 / 2, y + 272, "every action comes from the network", 13, INK, "mono", 600, "middle"))

    # arrows between cards
    def arrow(xa, xb, yy, label, color=ORANGE):
        return (f'<line x1="{xa}" y1="{yy}" x2="{xb - 8}" y2="{yy}" stroke="{color}" stroke-width="2.2" class="flow"/>'
                f'<path d="M{xb - 10},{yy - 6} L{xb},{yy} L{xb - 10},{yy + 6} Z" fill="{color}"/>'
                + text((xa + xb) / 2, yy - 10, label, 11, INK3, "mono", 500, "middle"))
    parts.append(arrow(364, 404, y + 150, "labels"))
    parts.append(arrow(836, 884, y + 150, "weights", BLUE))

    # PPO branch
    py = 416
    parts.append(f'<path d="M{x3 + w3 / 2},{y + h} V{py + 44} H{x2 + w2 + 20}" fill="none" stroke="{INK3}" stroke-width="1.6" stroke-dasharray="3 5"/>')
    parts.append(card(24 + 380, py, 452, 120, fill=CARD, stroke=LINE, extra='stroke-dasharray="5 5"'))
    parts.append(text(x2 + 22, py + 34, "04", 13, INK3, "mono", 700))
    parts.append(text(x2 + 52, py + 34, "Recurrent PPO fine-tuning", 18, INK2, "sans", 700))
    parts.append(text(x2 + 22, py + 62, "scorer-aligned rewards + count-based curiosity on its own map", 13, INK3))
    parts.append(text(x2 + 22, py + 92, "built and measured: held-out exits 0.58 → 0.38", 13, INK2, "mono", 500))
    parts.append(f'<rect x="{x2 + 330}" y="{py + 14}" width="100" height="24" rx="12" fill="none" stroke="{INK3}"/>')
    parts.append(text(x2 + 380, py + 31, "NOT SHIPPED", 11, INK3, "mono", 700, "middle", 'letter-spacing="1"'))
    # footnote
    parts.append(text(24, py + 40, "Why not RL from scratch?", 15, INK, "sans", 700))
    parts.append(text(24, py + 66, "It spends most of its budget learning", 13, INK2))
    parts.append(text(24, py + 86, "that walking forward is useful. Imitation", 13, INK2))
    parts.append(text(24, py + 106, "starts from competent play instead.", 13, INK2))
    return svg(W, H, "".join(parts), "Training pipeline: privileged teacher, DAgger, sensor-only student")


# ---------------------------------------------------------------- architecture

def architecture():
    W, H = 1280, 620
    p = [f'<rect width="{W}" height="{H}" rx="18" fill="{BG}" stroke="{LINE}"/>']
    p.append(text(32, 46, "Network", 22, INK, "sans", 700))
    p.append(text(130, 46, "submission/doomnet/policy.py · 992,998 parameters", 13, INK3, "mono"))

    rows = [
        ("rays", "64 × 360°, distance + hit type", "circular Conv1d ×3", "128"),
        ("entities", "≤ 32 visible, masked", "MLP → attention + null token", "64"),
        ("vitals", "17 vitals + last buttons + time", "MLP", "64"),
        ("spatial map", "4 × 24 × 24 heading-up crop", "Conv2d ×3", "128"),
        ("RGB", "160×120 frame pooled to 30×40", "Conv2d ×3 · aux exit head", "64"),
    ]
    y0, rh = 82, 100
    ix, iw = 32, 360
    ex, ew = 424, 236
    fx = 712
    for k, (name, desc, enc, dim) in enumerate(rows):
        y = y0 + k * rh
        p.append(card(ix, y, iw, rh - 16, rx=12))
        # icon area
        icx, icy = ix + 46, y + (rh - 16) / 2
        if name == "rays":
            for i in range(0, 64, 4):
                a = 2 * math.pi * i / 64
                L = 24 + 8 * math.sin(i * 0.7)
                p.append(f'<line x1="{icx}" y1="{icy}" x2="{icx - math.sin(a) * L:.1f}" y2="{icy - math.cos(a) * L:.1f}" '
                         f'stroke="{ORANGE}" stroke-width="1.3" opacity=".8"/>')
        elif name == "entities":
            for (dx, dy, c) in [(-18, -12, ENEMY), (14, -16, ENEMY), (-6, 14, GREEN), (20, 10, YELLOW), (0, -2, INK2)]:
                p.append(f'<circle cx="{icx + dx}" cy="{icy + dy}" r="5" fill="{c}" style="animation: pulse 2s {abs(dx) / 20:.1f}s infinite"/>')
        elif name == "vitals":
            for j, (hgt, c) in enumerate([(30, ENEMY), (22, BLUE), (14, YELLOW), (26, GREEN)]):
                p.append(f'<rect x="{icx - 22 + j * 12}" y="{icy + 16 - hgt}" width="8" height="{hgt}" rx="2" fill="{c}" opacity=".85"/>')
        elif name == "RGB":  # pooled frame: grey pixels, a red exit pad that the detector flags
            cs = 9
            gx0, gy0 = icx - 3 * cs, icy - 2 * cs
            for gx in range(6):
                for gy in range(4):
                    shade = ["#2a313b", "#343c47", "#3e4652"][(gx * 7 + gy * 3) % 3]
                    p.append(f'<rect x="{gx0 + gx * cs + .5}" y="{gy0 + gy * cs + .5}" width="{cs - 1}" height="{cs - 1}" fill="{shade}"/>')
            for (gx, gy) in [(3, 3), (4, 3)]:
                p.append(f'<rect x="{gx0 + gx * cs + .5}" y="{gy0 + gy * cs + .5}" width="{cs - 1}" height="{cs - 1}" fill="{ENEMY}" '
                         f'style="animation: pulse 1.6s infinite"/>')
            p.append(f'<rect x="{gx0 + 3 * cs - 2}" y="{gy0 + 3 * cs - 2}" width="{2 * cs + 3}" height="{cs + 3}" rx="2" fill="none" '
                     f'stroke="{YELLOW}" stroke-width="1.5" stroke-dasharray="3 2"/>')
        else:  # spatial map: 7x7 grid, a path of visited cells lights up
            cs = 8
            gx0, gy0 = icx - 3.5 * cs, icy - 3.5 * cs
            path = [(3, 6), (3, 5), (3, 4), (2, 4), (1, 4), (1, 3), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (5, 1)]
            for gx in range(7):
                for gy in range(7):
                    p.append(f'<rect x="{gx0 + gx * cs + .5}" y="{gy0 + gy * cs + .5}" width="{cs - 1}" height="{cs - 1}" '
                             f'fill="#1f2630"/>')
            for n_, (gx, gy) in enumerate(path):
                p.append(f'<rect x="{gx0 + gx * cs + .5}" y="{gy0 + gy * cs + .5}" width="{cs - 1}" height="{cs - 1}" fill="{AQUA}" opacity="0">'
                         f'<animate attributeName="opacity" values="0;0;1;1;0" keyTimes="0;{n_ / 16:.3f};{(n_ + 1) / 16:.3f};.94;1" '
                         f'dur="5s" repeatCount="indefinite"/></rect>')
            p.append(f'<rect x="{gx0 + 6 * cs + .5}" y="{gy0 + 0 * cs + .5}" width="{cs - 1}" height="{cs - 1}" fill="{YELLOW}"/>')
        p.append(text(ix + 96, y + 36, name, 17, INK, "sans", 700))
        p.append(text(ix + 96, y + 60, desc, 13, INK2))
        # encoder chip
        p.append(card(ex, y + 14, ew, rh - 44, fill=CARD2, rx=10))
        p.append(text(ex + ew / 2, y + 47, enc, 13, INK, "mono", 500, "middle"))
        # wires input → encoder → fusion
        yy = y + (rh - 16) / 2
        p.append(f'<line x1="{ix + iw}" y1="{yy}" x2="{ex}" y2="{yy}" stroke="{LINE}" stroke-width="2"/>')
        wire = f"M{ex + ew},{yy} C{ex + ew + 30},{yy} {fx - 30},{y0 + 242} {fx},{y0 + 242}"
        p.append(f'<path d="{wire}" fill="none" stroke="{LINE}" stroke-width="2"/>')
        p.append(f'<path d="{wire}" fill="none" stroke="{ORANGE}" stroke-width="2" class="flow" opacity=".7"/>')
        p.append(f'<circle r="4" fill="{ORANGE}"><animateMotion dur="1.6s" begin="{k * .3}s" repeatCount="indefinite" '
                 f'path="M{ix + iw},{yy} H{ex} M{ex + ew},{yy} C{ex + ew + 30},{yy} {fx - 30},{y0 + 242} {fx},{y0 + 242}"/></circle>')
        p.append(text(ex + ew + 14, yy - 8, dim, 11, INK3, "mono", 500))

    # fusion + GRU + heads
    cy = y0 + 242
    p.append(card(fx, cy - 62, 128, 124, fill=CARD2, rx=12))
    p.append(text(fx + 64, cy - 28, "concat 448", 13, INK2, "mono", 500, "middle"))
    p.append(text(fx + 64, cy - 4, "Linear 256", 14, INK, "mono", 600, "middle"))
    p.append(text(fx + 64, cy + 18, "LayerNorm", 13, INK2, "mono", 500, "middle"))
    p.append(text(fx + 64, cy + 40, "ReLU", 13, INK2, "mono", 500, "middle"))
    gx = fx + 160
    p.append(f'<line x1="{fx + 128}" y1="{cy}" x2="{gx}" y2="{cy}" stroke="{ORANGE}" stroke-width="2" class="flow"/>')
    p.append(card(gx, cy - 48, 108, 96, fill=RED, stroke=RED, rx=12, extra='fill-opacity=".14"'))
    p.append(text(gx + 54, cy - 4, "GRU", 22, INK, "sans", 800, "middle"))
    p.append(text(gx + 54, cy + 20, "256", 14, INK2, "mono", 600, "middle"))
    # recurrence loop
    p.append(f'<path d="M{gx + 78},{cy - 48} C{gx + 78},{cy - 96} {gx + 30},{cy - 96} {gx + 30},{cy - 52}" fill="none" stroke="{ORANGE}" stroke-width="2" class="flow"/>'
             f'<path d="M{gx + 24},{cy - 60} L{gx + 30},{cy - 49} L{gx + 36},{cy - 60} Z" fill="{ORANGE}"/>')
    p.append(text(gx + 54, cy - 100, "memory", 11, INK3, "mono", 500, "middle"))
    hx = gx + 136
    p.append(f'<line x1="{gx + 108}" y1="{cy}" x2="{hx}" y2="{cy}" stroke="{ORANGE}" stroke-width="2" class="flow"/>')
    p.append(text(hx + 4, cy - 150, "π · 16 macro-actions", 14, INK, "sans", 700))
    p.append(text(hx + 4, cy - 130, "sampled, T = 1.0", 12, INK3, "mono"))
    names = ["NOOP", "FWD", "FWD_L", "FWD_R", "TURN_L", "TURN_R", "BACK", "STRAF_L", "STRAF_R", "ATK", "ATK_L", "ATK_R",
             "FWD_ATK", "USE", "FWD_USE", "BK_ATK"]
    seq = [1, 1, 2, 1, 3, 13, 1, 1, 10, 9, 11, 1, 14, 2, 1, 5]  # a plausible sampled action sequence
    T = len(seq)
    cw, ch, gap = 58, 30, 5
    for i, nm in enumerate(names):
        r, c = divmod(i, 4)
        x = hx + c * (cw + gap)
        y = cy - 112 + r * (ch + gap)
        p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="7" fill="{CARD2}" stroke="{LINE}"/>')
        on = [t for t, a in enumerate(seq) if a == i]
        if on:
            vals = ";".join("1" if t in on else "0" for t in range(T)) + (";1" if (T - 1) in on else ";0")
            p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="7" fill="{ORANGE}" opacity="0">'
                     f'<animate attributeName="opacity" values="{vals}" calcMode="discrete" dur="{T * .45:.2f}s" repeatCount="indefinite"/></rect>')
        p.append(text(x + cw / 2, y + 19, nm, 10.5, INK, "mono", 600, "middle"))
    p.append(text(hx + 4, cy + 54 + 18 + 22, "V · value head", 13, INK2, "sans", 700))
    p.append(text(hx + 112, cy + 54 + 18 + 22, "(PPO only)", 12, INK3, "mono"))
    p.append(text(hx + 4, cy + 54 + 18 + 48, "no contradictory button combos,", 12, INK3))
    p.append(text(hx + 4, cy + 54 + 18 + 66, "one USE per press: doors don't bounce", 12, INK3))
    return svg(W, H, "".join(p), "Network architecture: five encoders (incl. RGB exit detector), GRU memory, 16 macro-actions")


# ---------------------------------------------------------------- results

def results():
    W = 1280
    # (family, metric, shipped, previous rays-only submission, label) -- fresh maps (confirm + confirmpad)
    rows = [
        ("All 200 fresh maps", "exit rate", 0.77, 0.61, None),
        ("Red floor-pad exits (official style)", "exit rate", 0.78, 0.48, None),
        ("Combat arenas: fight, then leave", "exit rate", 0.85, 0.55, None),
        ("Key-door with backtracking", "exit rate", 0.80, 0.70, None),
        ("Easy key-door", "exit rate", 1.00, 1.00, None),
        ("Multi-room exploration", "exit rate", 0.75, 0.90, None),
        ("Mixed gauntlet: enemies + locks", "exit rate", 0.60, 0.63, None),
        ("Enemies killed (maps with monsters)", "kill fraction", 0.93, 0.86, None),
        ("Official two_rooms.wad", "20 runs", 1.00, 1.00, "20/20 · 71 steps"),
    ]
    top, rh = 120, 50
    H = top + len(rows) * rh + 72
    lx, bx, bw = 32, 392, 620
    p = [f'<rect width="{W}" height="{H}" rx="18" fill="{BG}" stroke="{LINE}"/>']
    p.append(text(32, 48, "Results on fresh maps", 22, INK, "sans", 700))
    p.append(text(32, 74, "200 maps never used for training or for choosing the checkpoint · submitted inference path, 1 CPU thread", 13, INK2))
    p.append(f'<rect x="{bx + bw - 330}" y="62" width="14" height="10" rx="3" fill="{BLUE}"/>')
    p.append(text(bx + bw - 310, 72, "shipped agent", 12, INK2))
    p.append(f'<line x1="{bx + bw - 200}" y1="60" x2="{bx + bw - 200}" y2="76" stroke="{ORANGE}" stroke-width="3"/>')
    p.append(text(bx + bw - 190, 72, "previous submission (rays only)", 12, INK2))
    for f in (0, .25, .5, .75, 1):
        x = bx + f * bw
        p.append(f'<line x1="{x}" y1="{top - 14}" x2="{x}" y2="{top + len(rows) * rh - 10}" stroke="{"#262c35" if f else "#3a414b"}"/>')
        p.append(text(x, top + len(rows) * rh + 10, f"{int(f * 100)}%", 11, INK3, "mono", 400, "middle"))
    for k, (name, metric, v, prev, label) in enumerate(rows):
        y = top + k * rh
        p.append(text(lx, y + 14, name, 15, INK, "sans", 600 if k else 800))
        p.append(text(lx, y + 32, metric, 12, INK3, "mono"))
        p.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="24" rx="4" fill="#161b22"/>')
        w = v * bw
        d = 0.15 + k * 0.10
        p.append(f'<rect x="{bx}" y="{y}" width="0" height="24" rx="4" fill="{BLUE}">'
                 f'<animate attributeName="width" from="0" to="{w:.1f}" dur="1.1s" begin="{d:.2f}s" fill="freeze" '
                 f'calcMode="spline" keySplines=".2 .8 .2 1" keyTimes="0;1"/></rect>')
        px = bx + prev * bw
        p.append(f'<line x1="{px}" y1="{y - 4}" x2="{px}" y2="{y + 28}" stroke="{ORANGE}" stroke-width="3" opacity=".9"/>')
        delta = v - prev
        dtxt = "same" if abs(delta) < 0.005 else f"{delta * 100:+.0f} pts"
        dcol = INK3 if abs(delta) < 0.005 else (GREEN if delta > 0 else ENEMY)
        p.append(f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur=".4s" begin="{d + .9:.2f}s" fill="freeze"/>'
                 + text(bx + bw + 24, y + 18, label or f"{v:.2f}", 18, INK, "mono", 700)
                 + ("" if label else text(bx + bw + 92, y + 18, f"was {prev:.2f}", 12, INK3, "mono"))
                 + ("" if label else text(bx + bw + 172, y + 18, dtxt, 12, dcol, "mono", 700))
                 + "</g>")
    p.append(text(32, H - 22, "Paired over the same 200 maps: +16 ± 8 % exits vs the previous submission (95% CI). Kill fraction = share of enemies killed.",
                  12, INK3))
    return svg(W, H, "".join(p), "Fresh-map results of the shipped agent versus the previous submission")


# ---------------------------------------------------------------- ablations

def ablations():
    W, H = 1280, 520
    fams = ["Multi-room", "Easy key-door", "Key-door A/B/C", "Mixed gauntlet", "Floor-pad exits"]
    series = [("DAgger + spatial map", BLUE, [.95, .95, .67, .65, .41]),
              ("DAgger, no spatial map", ORANGE_S, [.35, .85, .50, .38, .29]),
              ("Behavioural cloning only", AQUA, [.45, .35, .10, .22, .25])]
    p = [f'<rect width="{W}" height="{H}" rx="18" fill="{BG}" stroke="{LINE}"/>']
    p.append(text(32, 48, "Ablations: what actually matters", 22, INK, "sans", 700))
    p.append(text(32, 74, "Held-out exit rate · same seed, same teacher data · paired over 600 episodes", 13, INK2))
    lx = 32
    for name, c, _ in series:
        p.append(f'<rect x="{lx}" y="96" width="14" height="10" rx="3" fill="{c}"/>' + text(lx + 20, 106, name, 13, INK2))
        lx += 44 + len(name) * 7.4
    cx0, cy0, cw, chh = 70, 140, 800, 300
    base = cy0 + chh
    for f in (0, .25, .5, .75, 1):
        y = base - f * chh
        p.append(f'<line x1="{cx0}" y1="{y}" x2="{cx0 + cw}" y2="{y}" stroke="{"#262c35" if f else "#3a414b"}"/>')
        p.append(text(cx0 - 10, y + 4, f"{int(f * 100)}%", 11, INK3, "mono", 400, "end"))
    gw = cw / len(fams)
    bw, gap = 38, 3
    for i, fam in enumerate(fams):
        gx = cx0 + i * gw + (gw - (3 * bw + 2 * gap)) / 2
        for s, (name, c, vals) in enumerate(series):
            v = vals[i]
            h = v * chh
            x = gx + s * (bw + gap)
            d = 0.2 + i * 0.12 + s * 0.06
            # rounded top only: bar = rect with rx clipped by a base rect
            p.append(f'<g><rect x="{x}" y="{base}" width="{bw}" height="0" rx="4" fill="{c}">'
                     f'<animate attributeName="height" from="0" to="{h:.1f}" dur="1s" begin="{d:.2f}s" fill="freeze" calcMode="spline" keySplines=".2 .8 .2 1" keyTimes="0;1"/>'
                     f'<animate attributeName="y" from="{base}" to="{base - h:.1f}" dur="1s" begin="{d:.2f}s" fill="freeze" calcMode="spline" keySplines=".2 .8 .2 1" keyTimes="0;1"/></rect>'
                     f'<rect x="{x}" y="{base - 4}" width="{bw}" height="4" fill="{c}" opacity="0"><set attributeName="opacity" to="1" begin="{d:.2f}s" fill="freeze"/></rect></g>')
            p.append(f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" dur=".3s" begin="{d + .8:.2f}s" fill="freeze"/>'
                     + text(x + bw / 2, base - h - 7, f"{v:.2f}".lstrip("0"), 11, INK2, "mono", 600, "middle") + "</g>")
        p.append(text(cx0 + i * gw + gw / 2, base + 24, fam, 13, INK, "sans", 600, "middle"))

    # stat tiles
    tx, tw = 912, 336
    tiles = [("+19.2 %", "exits from the spatial map", "±4.8 · +23.9 proxy pts", BLUE),
             ("+34.0 %", "exits from DAgger vs cloning", "±4.7 · +45.0 proxy pts", AQUA),
             ("−5.5 %", "exits if actions are greedy", "argmax loops; sampling escapes", ORANGE_S)]
    for k, (big, what, sub, c) in enumerate(tiles):
        y = 128 + k * 112
        p.append(card(tx, y, tw, 96, rx=12))
        p.append(f'<rect x="{tx}" y="{y + 16}" width="4" height="64" rx="2" fill="{c}"/>')
        p.append(text(tx + 24, y + 46, big, 30, INK, "sans", 800))
        p.append(text(tx + 24, y + 72, what, 14, INK2))
        p.append(text(tx + 24, y + 89, sub, 11, INK3, "mono"))
    p.append(text(32, H - 22, "Paired differences: mean ± 95% CI half-width over the same 600 held-out episodes (docs/ablations.md).", 12, INK3))
    return svg(W, H, "".join(p), "Ablations: spatial map, DAgger versus behavioural cloning, action selection")


for name, fn in [("hero", hero), ("pipeline", pipeline), ("architecture", architecture), ("results", results), ("ablations", ablations)]:
    (OUT / f"{name}.svg").write_text(fn(), encoding="utf-8")
    print("wrote", OUT / f"{name}.svg")
