"""Memory demo (Spec v2.1, Demo): play a key-door map with the episodic-memory attention overlaid.

    .venv\\Scripts\\python.exe -m demo.memory_demo runs/mem/latest.pt                 # first variant-B validation map
    .venv\\Scripts\\python.exe -m demo.memory_demo runs/mem/latest.pt --index 4 --ablation clear_at_key

Video: the agent's view (x3) + button bar + a side panel with one bar per memory read target
(the learned null record + 16 slots). Bar length = attention weight this decision; colour = why
the record was written:
    white  null record ("nothing in memory helps")   grey   recent (every 8 decisions)
    red    locked door ahead      orange door ahead    yellow key in view
    green  inventory changed      blue   new area
A thin yellow stripe on the panel's left marks decisions after a key pickup.

It also prints the attention mass on locked-door records in the 40 decisions before vs after
the key pickup (the spec's expectation: it rises after the pickup). Reported as measured.
Instrumentation only: MemoryActor.debug_attention is off in submission/ (nothing sets it there).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from training.record import SCALE, draw_text, frame

REPO = Path(__file__).resolve().parents[1]
COLORS = {"null": (240, 240, 240), "recent": (120, 120, 120), "locked_door": (230, 40, 40),
          "door": (240, 150, 30), "key_seen": (250, 230, 40), "inventory": (60, 210, 80),
          "new_area": (60, 130, 240), "empty": (35, 35, 35)}
PANEL_W = 220


def panel(height: int, weights: np.ndarray, reasons: list[str], after_key: bool) -> np.ndarray:
    p = np.full((height, PANEL_W, 3), 18, np.uint8)
    if after_key:
        p[:, :4] = (250, 230, 40)
    rows = len(weights)
    rh = (height - 20) // rows
    draw_text(p, 10, 4, "17", (200, 200, 200), px=2)        # 17 read targets
    for i, (w, r) in enumerate(zip(weights, reasons)):
        y = 20 + i * rh
        p[y + 2:y + rh - 2, 10:PANEL_W - 10] = (40, 40, 40)
        L = int((PANEL_W - 20) * float(np.clip(w, 0, 1)))
        p[y + 2:y + rh - 2, 10:10 + L] = COLORS.get(r, COLORS["empty"])
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="memory-model checkpoint (training ckpt or exported actor.pt)")
    ap.add_argument("--family", default="keydoor")
    ap.add_argument("--pool", default="validation")
    ap.add_argument("--variant", default="B")
    ap.add_argument("--index", type=int, default=0, help="n-th map of that variant")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--threshold", action="store_true", help="threshold mode (default: sample)")
    ap.add_argument("--ablation", default=None, choices=["memory_off", "clear_at_key", "foreign", "no_inv_query"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import imageio_ffmpeg
    from doom_env import make_env

    from training.evaluate import _W, _init, actor_weights
    from training.maps.gen import load_index

    entries = [e for e in load_index(a.pool, a.family) if not a.variant or e.get("variant") == a.variant]
    entry = entries[a.index]
    _init(actor_weights(a.ckpt, greedy=a.threshold), "cpu", a.ablation)
    agent = _W["agent"]
    _W["runner"].close()
    if not agent.memory:
        raise SystemExit("not a memory-model checkpoint")
    agent.actor.debug_attention = True

    env = make_env(entry["wad"], entry["map"], episode_ticks=int(entry["ticks"]), render_buffers=True)
    tag = f"{entry['map']}_{entry.get('variant') or ''}{'_' + a.ablation if a.ablation else ''}"
    out = Path(a.out or REPO / "runs" / "videos" / f"memory_demo_{tag}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=a.seed)
    agent.reset()
    h, w = obs["rgb"].shape[:2]
    H, W = h * SCALE + 40, w * SCALE + PANEL_W
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "16", "-i", "-",
                             "-pix_fmt", "yuv420p", "-vcodec", "libx264", str(out)], stdin=subprocess.PIPE)
    step, key_step, locked_mass, ps = 0, None, [], env.privileged_state()
    try:
        while True:
            act = agent.act(obs)
            att = agent.actor.last_attention[0].cpu().numpy()            # (17,): null + 16 slots
            slots = agent.book.slots if a.ablation != "foreign" else (getattr(agent, "donor", (agent.book,))[0].slots)
            reasons = ["null"] + [r.reason if r is not None else "empty" for r in slots]
            locked_mass.append(float(sum(wt for wt, r in zip(att, reasons) if r == "locked_door")))
            if key_step is None and ps["keys"].sum() > 0:
                key_step = step
            img = frame(obs["rgb"], act, step, ps["health"])
            img = np.concatenate([img, panel(img.shape[0], att, reasons, key_step is not None)], 1)
            proc.stdin.write(img.tobytes())
            obs, _, te, tr, _ = env.step(act)
            ps = env.privileged_state()
            step += 1
            if te or tr:
                break
    finally:
        proc.stdin.close()
        proc.wait()
        env.close()
    m = np.array(locked_mass)
    print(f"{entry['label']} (variant {entry.get('variant')}), ablation={a.ablation}: {step} decisions, "
          f"exit={ps['exit_reached']} key_at={key_step} -> {out}")
    if key_step is not None and key_step >= 5:
        before = m[max(0, key_step - 40):key_step]
        after = m[key_step:key_step + 40]
        print(f"attention on locked-door records: {before.mean():.3f} in the {len(before)} decisions before the "
              f"key pickup, {after.mean():.3f} in the {len(after)} after (measured, single episode)")
    else:
        print("no key pickup in this episode: no before/after comparison")


if __name__ == "__main__":
    main()
