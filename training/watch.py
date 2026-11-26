"""Live viewer: watch the latest checkpoint of a training run play held-out maps (training only).

    .venv\\Scripts\\python.exe -m training.watch --run il4
    .venv\\Scripts\\python.exe -m training.watch --ckpt submission/weights/actor.pt --family keydoor --speed 2
    .venv\\Scripts\\python.exe -m training.watch --teacher          # the privileged teacher instead

Plays validation maps (never trained on) through the SUBMITTED DoomAgent, reloading the weights
whenever the run saves a new checkpoint (between episodes). With --teacher, the training-time
expert (training/expert.py: floor plan + true position) plays instead, for comparison; its
spatial-map panel is what the student would see in the same situation. The window shows:
  * the game view (RGB is rendered for humans only; the agent never sees pixels),
  * the agent's own spatial-map crop (heading up): grey = swept by rays, white = visited,
    orange = door seen, magenta = locked door seen,
  * a top-down floor plan with the agent's true track (privileged; viewer only),
  * the macro-action it chose, vitals, and recent episode outcomes.
Uses one CPU core; close the window to stop.
"""
from __future__ import annotations

import argparse
import math
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VIEW_SCALE = 4                     # 160x120 -> 640x480
PANEL = 240


def ppm(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    return f"P6 {w} {h} 255 ".encode() + np.ascontiguousarray(img, np.uint8).tobytes()


def map_panel(crop: np.ndarray) -> np.ndarray:
    """(4, 24, 24) spatial-map crop -> (240, 240, 3) image, heading up."""
    c = crop.shape[1]
    img = np.full((c, c, 3), 18, np.uint8)
    img[crop[1] > 0.5] = (70, 70, 80)
    img[crop[0] > 0.5] = (235, 235, 235)
    img[crop[2] > 0.5] = (255, 150, 30)
    img[crop[3] > 0.5] = (230, 40, 230)
    k = PANEL // c
    big = np.repeat(np.repeat(img, k, 0), k, 1)
    m = PANEL // 2
    big[m - 4:m + 4, m - 4:m + 4] = (60, 200, 90)            # the agent (centre), facing up
    big[m - 14:m - 4, m - 1:m + 1] = (60, 200, 90)
    return big


class Overview:
    """Top-down floor plan of a generated map from its route metadata (viewer only)."""
    LOCK_RGB = {1: (220, 40, 40), 2: (40, 80, 230), 3: (230, 210, 40), 4: (220, 40, 40), 5: (40, 80, 230),
                6: (230, 210, 40)}

    def __init__(self, entry: dict):
        route = entry.get("route") or {"rects": []}
        rects = route["rects"]
        self.ok = bool(rects)
        if not self.ok:
            return
        x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
        x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
        self.s = (PANEL - 10) / max(x1 - x0, y1 - y0)
        self.x0, self.y0 = x0, y0
        base = np.full((PANEL, PANEL, 3), 18, np.uint8)
        for rx0, ry0, rx1, ry1, fl, ce, door, lock in rects:
            if ce <= fl and not door:
                continue
            col = self.LOCK_RGB.get(lock, (255, 150, 30)) if door and lock else (150, 100, 50) if door else (85, 85, 95)
            (a, b), (c, d) = self.px(rx0, ry1), self.px(rx1, ry0)
            base[b:d + 1, a:c + 1] = col
        if route.get("exit_xy"):
            self.dot(base, *route["exit_xy"], (40, 230, 80), 4)
        self.keys = route.get("keys", [])
        self.base = base
        self.trail: list = []

    def px(self, x, y):
        return int(5 + (x - self.x0) * self.s), int(PANEL - 5 - (y - self.y0) * self.s)

    def dot(self, img, x, y, col, r=2):
        i, j = self.px(x, y)
        img[max(0, j - r):j + r + 1, max(0, i - r):i + r + 1] = col

    def render(self, ps: dict) -> np.ndarray:
        if not self.ok:
            return np.full((PANEL, PANEL, 3), 18, np.uint8)
        img = self.base.copy()
        held = np.asarray(ps["keys"])
        for kx, ky, lk in self.keys:
            flag = {1: 2, 2: 0, 3: 1, 4: 5, 5: 3, 6: 4}.get(lk, 0)
            if held[flag] < 0.5:
                self.dot(img, kx, ky, self.LOCK_RGB.get(lk, (255, 255, 255)), 3)
        self.trail.append((ps["x"], ps["y"]))
        for x, y in self.trail[::2]:
            self.dot(img, x, y, (200, 200, 200), 0)
        a = math.radians(ps["angle"])
        for t in range(0, 14, 2):
            self.dot(img, ps["x"] + t / self.s * math.cos(a), ps["y"] + t / self.s * math.sin(a), (60, 200, 90), 1)
        self.dot(img, ps["x"], ps["y"], (60, 200, 90), 3)
        return img


class TeacherAgent:
    """The privileged expert behind the DoomAgent interface (viewer only)."""

    def __init__(self, entry: dict, env):
        from submission.doomnet import obs_adapter as A
        from training.expert import Expert

        self.env, self.expert = env, Expert(entry["route"])
        self.adapter = A.ObsAdapter()

    def reset(self):
        self.expert.reset()
        self.adapter.reset()

    def act(self, obs):
        from submission.doomnet.actions import to_buttons

        self.adapter.observe(obs)                     # keeps the student's-eye spatial map for display
        self.last_action_index = self.expert.act(obs, self.env.privileged_state())
        b = to_buttons(self.last_action_index)
        self.adapter.record_action(b)
        return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="il2", help="training run to follow (runs/<run>/latest.pt)")
    ap.add_argument("--ckpt", default=None, help="fixed checkpoint instead of following a run")
    ap.add_argument("--family", nargs="+", default=["rooms", "keydoor", "mixed", "combat"])
    ap.add_argument("--pool", default="validation")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time (8.75 decisions/s); 0 = as fast as possible")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--teacher", action="store_true", help="watch the privileged teacher instead of the student")
    a = ap.parse_args()

    import importlib.util
    import sys

    from doom_env import action_space, make_env, observation_space
    from submission.doomnet.actions import NAMES
    from training.evaluate import agent_weights
    from training.maps.gen import load_index

    ckpt = Path(a.ckpt) if a.ckpt else REPO / "runs" / a.run / "latest.pt"
    spec = importlib.util.spec_from_file_location("submission_agent_watch", REPO / "submission" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submission_agent_watch"] = mod
    spec.loader.exec_module(mod)

    maps = []
    for fam in a.family:
        maps += load_index(a.pool, fam)
    rng = np.random.default_rng()
    st = {"agent": None, "mtime": None, "iter": None, "env": None, "wad": None, "results": deque(maxlen=8)}

    root = tk.Tk()
    root.title("DOOM TEACHER (privileged expert, training only)" if a.teacher else f"DOOM agent live (student): {ckpt}")
    root.configure(bg="#111")
    top = tk.Frame(root, bg="#111")
    top.pack()
    view = tk.Label(top, bg="#111")
    view.grid(row=0, column=0, rowspan=2)
    mem = tk.Label(top, bg="#111")
    mem.grid(row=0, column=1, padx=6, pady=2)
    ov = tk.Label(top, bg="#111")
    ov.grid(row=1, column=1, padx=6, pady=2)
    font = ("Consolas", 11)
    info = tk.Label(root, bg="#111", fg="#ddd", font=font, justify="left", anchor="w")
    info.pack(fill="x", padx=8)
    hist = tk.Label(root, bg="#111", fg="#9c9", font=font, justify="left", anchor="w")
    hist.pack(fill="x", padx=8, pady=(0, 6))
    tk.Label(top, text="agent's spatial map (heading up)", bg="#111", fg="#888", font=("Consolas", 9)).grid(row=2, column=1)
    imgs = {}

    def load_agent():
        if not ckpt.is_file():
            return False
        m = ckpt.stat().st_mtime
        if m != st["mtime"]:
            for _ in range(5):                       # the trainer may be mid-write: retry
                try:
                    mod.WEIGHTS_FILE = Path(agent_weights(str(ckpt), greedy=a.greedy))
                    break
                except Exception:
                    time.sleep(1.0)
            st["agent"] = mod.DoomAgent(observation_space(), action_space(), "cpu")
            st["mtime"] = m
            import torch
            st["iter"] = torch.load(ckpt, map_location="cpu", weights_only=False).get("meta", {}).get("iter")
        return True

    def new_episode():
        if not a.teacher and not load_agent():
            info.config(text=f"waiting for {ckpt} ...")
            root.after(3000, new_episode)
            return
        e = maps[rng.integers(len(maps))]
        if st["env"] is not None and (st["wad"], st["map"]) != (e["wad"], e["map"]):
            st["env"].close()
            st["env"] = None
        if st["env"] is None:
            st["env"] = make_env(e["wad"], e["map"], episode_ticks=int(e["ticks"]), render_buffers=True)
            st["wad"], st["map"] = e["wad"], e["map"]
        st.update(entry=e, overview=Overview(e), step=0, t_next=time.perf_counter())
        st["obs"], _ = st["env"].reset(seed=int(rng.integers(1 << 30)))
        if a.teacher:
            st["agent"], st["iter"] = TeacherAgent(e, st["env"]), "TEACHER"
        st["agent"].reset()
        root.after(1, tick)

    def tick():
        agent, env, e = st["agent"], st["env"], st["entry"]
        act = agent.act(st["obs"])
        name = NAMES[agent.last_action_index]
        pose = agent.adapter.pose                        # the pose of the observation just acted on
        crop = agent.adapter.map.crop(np.array([pose.x, pose.y, pose.yaw], np.float32))
        frame = np.repeat(np.repeat(st["obs"]["rgb"], VIEW_SCALE, 0), VIEW_SCALE, 1)
        st["obs"], _, te, tr, _ = env.step(act)
        ps = env.privileged_state()
        st["step"] += 1
        for key, img in (("view", frame), ("mem", map_panel(crop)), ("ov", st["overview"].render(ps))):
            imgs[key] = tk.PhotoImage(data=ppm(img), format="PPM")
        view.config(image=imgs["view"])
        mem.config(image=imgs["mem"])
        ov.config(image=imgs["ov"])
        v = st["obs"]["vitals"]
        mon = int(e.get("monsters", 0))
        info.config(text=(f"{e['label']} ({e.get('variant') or e['family']})   checkpoint iter {st['iter']}\n"
                          f"step {st['step']:4d}/{int(e['ticks']) // 4}   action {name:9s}   health {v[0]:4.0f}  "
                          f"armor {v[1]:3.0f}  ammo {v[3]:3.0f}   kills {ps['kills']}/{mon}   "
                          f"keys {int(np.sum(ps['keys']))}/{len((e.get('route') or {}).get('keys', []))}"))
        cleared = bool(mon and ps["kills"] >= mon)
        if te or tr:
            res = "EXIT" if ps["exit_reached"] else "DIED" if ps["dead"] else "CLEARED" if cleared else "timeout"
            st["results"].appendleft(f"{res:8s} {e['label']:24s} {st['step']:4d} steps  kills {ps['kills']}/{mon}  "
                                     f"keys {int(np.sum(ps['keys']))}  (iter {st['iter']})")
            hist.config(text="recent episodes:\n" + "\n".join(st["results"]))
            root.after(800, new_episode)
            return
        if a.speed > 0:
            st["t_next"] += (4 / 35) / a.speed
            delay = max(1, int(1000 * (st["t_next"] - time.perf_counter())))
        else:
            delay = 1
        root.after(delay, tick)

    root.after(10, new_episode)
    try:
        root.mainloop()
    finally:
        if st["env"] is not None:
            st["env"].close()


if __name__ == "__main__":
    main()
