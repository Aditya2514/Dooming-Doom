"""Record a video of the agent playing one map (diagnostics and demo).

    .venv\\Scripts\\python.exe -m training.record submission/weights/actor.pt --family rooms --index 0
    .venv\\Scripts\\python.exe -m training.record runs/il/best.pt --family keydoor --index 3 --out kd3.mp4

Plays the map through the SUBMITTED DoomAgent (same path as evaluation) with RGB rendering on,
and writes an MP4 via ffmpeg (bundled by the imageio-ffmpeg training dependency): the agent's view upscaled
3x, plus a bar showing the 9 buttons pressed each decision, the step count and health.
Prints the episode outcome. Rendering is for humans only: the agent never uses pixels.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BUTTONS = ["FWD", "BACK", "STR-L", "STR-R", "TURN-L", "TURN-R", "FIRE", "USE", "WPN"]
SCALE = 3

# 3x5 pixel digits for the overlay
DIGITS = {"0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
          "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
          "8": "111101111101111", "9": "111101111001111", " ": "000000000000000"}


def draw_text(img: np.ndarray, x: int, y: int, text: str, color, px: int = 3):
    for ch in text:
        pat = DIGITS.get(ch, DIGITS[" "])
        for i, bit in enumerate(pat):
            if bit == "1":
                r, c = divmod(i, 3)
                img[y + r * px:y + (r + 1) * px, x + c * px:x + (c + 1) * px] = color
        x += 4 * px


def frame(rgb: np.ndarray, action: np.ndarray, step: int, health: float) -> np.ndarray:
    big = np.repeat(np.repeat(rgb, SCALE, 0), SCALE, 1)
    W = big.shape[1]
    bar = np.zeros((40, W, 3), np.uint8)
    cell = W // 9
    for i, on in enumerate(action):
        col = (60, 220, 90) if on else (45, 45, 45)
        bar[6:34, i * cell + 3:(i + 1) * cell - 3] = col
    out = np.concatenate([big, bar], 0)
    draw_text(out, 6, 6, f"{step}", (255, 255, 0))
    draw_text(out, W - 60, 6, f"{int(max(health, 0))}", (255, 80, 80))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--family", default="rooms")
    ap.add_argument("--pool", default="validation")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--greedy", action="store_true", help="argmax actions (default: sample, as submitted)")
    ap.add_argument("--wad", default=None, help="record any WAD instead of a pool map, e.g. maps/two_rooms.wad")
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--ticks", type=int, default=2100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=35 // 4 * 2, help="video fps (decisions per second shown)")
    a = ap.parse_args()

    try:
        import imageio_ffmpeg                       # training dependency: bundled ffmpeg binary
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("no ffmpeg: run 'python -m uv sync' (installs imageio-ffmpeg)")
    from doom_env import action_space, make_env, observation_space

    from training.evaluate import _init, _W, agent_weights
    from training.maps.gen import load_index

    if a.wad:
        entry = {"wad": a.wad, "map": a.map, "ticks": a.ticks, "label": Path(a.wad).stem, "family": "custom"}
    else:
        entry = load_index(a.pool, a.family)[a.index]
    _init(agent_weights(a.ckpt, greedy=a.greedy), "cpu")
    agent = _W["agent"]
    _W["runner"].close()
    env = make_env(entry["wad"], entry["map"], episode_ticks=int(entry["ticks"]), render_buffers=True)
    out = Path(a.out or f"runs/videos/{entry['label'].replace('/', '_')}_{'greedy' if a.greedy else 'sample'}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=a.seed)
    agent.reset()
    h, w = obs["rgb"].shape[:2]
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w * SCALE}x{h * SCALE + 40}", "-r", str(a.fps), "-i", "-",
                             "-pix_fmt", "yuv420p", "-vcodec", "libx264", str(out)], stdin=subprocess.PIPE)
    step, ps = 0, env.privileged_state()
    try:
        while True:
            act = agent.act(obs)
            proc.stdin.write(frame(obs["rgb"], act, step, ps["health"]).tobytes())
            obs, _, te, tr, _ = env.step(act)
            ps = env.privileged_state()
            step += 1
            cleared = entry.get("end_on_clear") and entry.get("monsters") and ps["kills"] >= entry["monsters"]
            if te or tr or cleared:
                break
    finally:
        proc.stdin.close()
        proc.wait()
        env.close()
    print(f"{entry.get('label', entry['map'])} ({entry.get('variant') or entry['family']}): {step} decisions, "
          f"exit={ps['exit_reached']} death={ps['dead']} kills={ps['kills']}/{entry.get('monsters', 0)} "
          f"keys={int(ps['keys'].sum())} health={ps['health']:.0f} -> {out}")


if __name__ == "__main__":
    main()
