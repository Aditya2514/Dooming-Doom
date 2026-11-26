"""Live view of a training run: watch one training worker's game in real time.

    # on the training laptop, in a second window, while training runs:
    .venv\\Scripts\\python.exe -m training.live --run mem

One worker (worker 0 by default) copies its screen and a few numbers into a small named shared
memory block after every decision (~57 KB). ViZDoom renders every frame anyway, so the cost is one
copy per decision for one of the envs (~1-2% of training speed). The viewer is a separate process:
open and close it at any time; training never waits for it. Keys: Q / Esc quit, +/- zoom.
These are the real training games (sampling actions, current weights), not a replay.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import shared_memory

import numpy as np

H, W = 120, 160
HEADER = 256
SIZE = HEADER + H * W * 3
BUTTONS = ["FWD", "BACK", "STR-L", "STR-R", "TURN-L", "TURN-R", "FIRE", "USE", "WPN"]


def shm_name(run: str) -> str:
    return f"doomlive_{run}"


class LiveWriter:
    """Writer side, used inside one env worker. Never raises into training."""

    def __init__(self, name: str):
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=SIZE)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name)
        b = self.shm.buf
        self.seq = np.ndarray((1,), np.uint64, buffer=b, offset=0)
        self.ints = np.ndarray((8,), np.int32, buffer=b, offset=8)       # ep_len, episode, kills, monsters, exits, deaths, keys, unused
        self.floats = np.ndarray((2,), np.float32, buffer=b, offset=40)  # health, episode return
        self.buttons = np.ndarray((9,), np.uint8, buffer=b, offset=48)
        self.label = np.ndarray((64,), np.uint8, buffer=b, offset=64)
        self.frame = np.ndarray((H, W, 3), np.uint8, buffer=b, offset=HEADER)
        self.episode = self.exits = self.deaths = 0

    def write(self, rgb, action, label: str, ep_len: int, ps: dict, monsters: int, ret: float):
        try:
            self.frame[:] = rgb
            self.buttons[:] = np.asarray(action, np.uint8)[:9]
            lab = label.encode("ascii", "replace")[:63]
            self.label[:] = 0
            self.label[:len(lab)] = np.frombuffer(lab, np.uint8)
            self.ints[:7] = (ep_len, self.episode, int(ps["kills"]), monsters, self.exits, self.deaths,
                             int(np.sum(ps["keys"])))
            self.floats[:] = (float(ps["health"]), float(ret))
            self.seq[0] += 1
        except Exception:                 # noqa: BLE001 - a viewer problem must never stop training
            pass

    def episode_end(self, exited: bool, died: bool):
        self.episode += 1
        self.exits += int(exited)
        self.deaths += int(died)

    def close(self):
        try:
            self.shm.close()
        except Exception:                 # noqa: BLE001
            pass


# ------------------------------------------------------------------------ viewer
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="mem")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()

    import os
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.init()
    scale = a.scale
    bar = 70
    screen = pygame.display.set_mode((W * scale, H * scale + bar), pygame.RESIZABLE)
    pygame.display.set_caption(f"Dooming Doom - live training view ({a.run})")
    font = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 12)
    clock = pygame.time.Clock()
    shm, last_seq, last_change = None, -1, time.time()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE)):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                scale = min(8, scale + 1)
                screen = pygame.display.set_mode((W * scale, H * scale + bar), pygame.RESIZABLE)
            elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                scale = max(2, scale - 1)
                screen = pygame.display.set_mode((W * scale, H * scale + bar), pygame.RESIZABLE)
        screen.fill((12, 12, 14))
        if shm is None:
            try:
                shm = shared_memory.SharedMemory(name=shm_name(a.run))
            except FileNotFoundError:
                screen.blit(font.render(f"waiting for training run '{a.run}' (live view starts with the run)...",
                                        True, (220, 220, 220)), (10, 10))
                pygame.display.flip()
                clock.tick(2)
                continue
        b = shm.buf
        seq = int(np.ndarray((1,), np.uint64, buffer=b, offset=0)[0])
        ints = np.ndarray((8,), np.int32, buffer=b, offset=8).copy()
        floats = np.ndarray((2,), np.float32, buffer=b, offset=40).copy()
        buttons = np.ndarray((9,), np.uint8, buffer=b, offset=48).copy()
        label = bytes(np.ndarray((64,), np.uint8, buffer=b, offset=64)).split(b"\0")[0].decode("ascii", "replace")
        frame = np.ndarray((H, W, 3), np.uint8, buffer=b, offset=HEADER).copy()
        if seq != last_seq:
            last_seq, last_change = seq, time.time()
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.blit(pygame.transform.scale(surf, (W * scale, H * scale)), (0, 0))
        y = H * scale + 6
        cell = (W * scale) // 9
        for i, (name, on) in enumerate(zip(BUTTONS, buttons)):
            r = pygame.Rect(i * cell + 2, y, cell - 4, 22)
            pygame.draw.rect(screen, (60, 200, 90) if on else (45, 45, 50), r, border_radius=4)
            t = small.render(name, True, (10, 10, 10) if on else (170, 170, 170))
            screen.blit(t, t.get_rect(center=r.center))
        ep_len, episode, kills, monsters, exits, deaths, keys = ints[:7]
        stale = time.time() - last_change > 5
        info = (f"{label}   step {ep_len}   health {floats[0]:.0f}   kills {kills}/{monsters}   keys {keys}   "
                f"return {floats[1]:+.2f}   | worker episodes {episode}: exits {exits}, deaths {deaths}"
                + ("   [paused: gate eval / update / restart]" if stale else ""))
        screen.blit(font.render(info, True, (230, 230, 120) if not stale else (230, 140, 120)), (6, y + 30))
        pygame.display.flip()
        clock.tick(a.fps)
    if shm is not None:
        shm.close()
    pygame.quit()


if __name__ == "__main__":
    main()
