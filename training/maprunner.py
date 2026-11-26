"""MapRunner: one official DoomEnv moved between maps (training / local evaluation only).

Maps of the same WAD are switched inside the running engine (~0.07 s instead of ~0.54 s for a
new engine). On Windows it also applies training/win_ipc.py (3x faster engine round trips).
"""
from __future__ import annotations

from pathlib import Path


class MapRunner:
    def __init__(self, fix_ipc: bool = True, render: bool = False):
        self.env = None
        self.fix_ipc = fix_ipc
        self.render = render

    def load(self, entry: dict):
        """entry: {"wad", "map", "ticks", ...} (a map index entry). Returns a DoomEnv ready to reset()."""
        from doom_env import make_env
        from doom_env.mapinfo import load_mapinfo

        wad = str(Path(entry["wad"]).resolve())          # DoomEnv stores the resolved path
        name, ticks = entry["map"].upper(), int(entry["ticks"])
        env = self.env
        if env is not None and env.wad == wad:
            if (env.map_name, env.episode_ticks) != (name, ticks):
                # same WAD: mirror DoomEnv.__init__'s per-map state; reset() rebuilds geometry/sensors
                g = env._game
                if env.map_name != name:
                    g.set_doom_map(name)
                    env.map_name = name
                    env.mapinfo = load_mapinfo(env.wad, name)
                    env._geometry = None
                    env._sensors = None
                g.set_episode_timeout(ticks)
                env.episode_ticks = ticks
            return env
        if env is not None:
            env.close()
        self.env = make_env(wad, name, episode_ticks=ticks, render_buffers=self.render)
        if self.fix_ipc:
            from training.win_ipc import windows_fast_ipc
            windows_fast_ipc()
        return self.env

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
