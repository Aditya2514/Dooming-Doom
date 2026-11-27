"""Teacher -> student distillation with DAgger (Ross, Gordon & Bagnell 2011). Training only.

    .venv\\Scripts\\python.exe -m training.dagger --run il --workers 8

Teacher: training/expert.py (privileged: true pose + floor plan; decisions kept imitable).
Student: submission/doomnet/policy.py, which sees ONLY the official observation (via the shared
adapter), exactly as at evaluation.

Loop (one "iteration"):
  1. collect: W engines in lockstep. Every visited state is labelled with the expert's action.
     Which action is EXECUTED is decided per episode: with prob p_student the student drives
     alone (on-policy states, and an honest running score of the student); otherwise each step
     executes the expert's action with prob beta, else the student's. Iteration 0 is pure expert.
  2. aggregate: episodes go into a FIFO store (DAgger's growing dataset, capped by steps).
  3. train: cross-entropy to the expert label on whole episodes through the GRU (exactly the
     acting computation), for a fixed number of updates.
Checkpoints: runs/<run>/latest.pt (+ best by student score) in the DoomAgent weight format.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAP_SHAPE = (4, 24, 24)          # obs_adapter.MAP_CHANNELS, MAP_CROP, MAP_CROP (packed with np.packbits)
FAMILIES = {"rooms": 0.2, "keyeasy": 0.1, "keydoor": 0.2, "mixed": 0.35, "combat": 0.15}


# ------------------------------------------------------------------ episode store
class EpisodeStore:
    """Compact per-episode arrays; FIFO by total steps."""

    def __init__(self, cap_steps: int):
        self.cap = cap_steps
        self.eps: deque = deque()
        self.steps = 0

    @staticmethod
    def pack(xs: list[dict], labels: list[int]) -> dict:
        rays = np.stack([x["rays"] for x in xs])                    # (T, 7, 64)
        return {
            "dist": rays[:, 0].astype(np.float16),
            "hit": rays[:, 1:].argmax(1).astype(np.uint8),
            "ent_type": np.stack([x["ent_type"] for x in xs]).astype(np.uint8),
            "ent_feat": np.stack([x["ent_feat"] for x in xs]).astype(np.float16),
            "ent_mask": np.stack([x["ent_mask"] for x in xs]).astype(bool),
            "scalars": np.stack([x["scalars"] for x in xs]).astype(np.float32),
            "map": np.stack([x["map"] for x in xs]).astype(np.uint8),
            "label": np.asarray(labels, np.int64),
        }

    def save(self, path) -> None:
        """Persist the dataset (DAgger's aggregate) so a resumed run continues from it."""
        import pickle

        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"cap": self.cap, "eps": list(self.eps)}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def load(self, path) -> None:
        import pickle

        with open(path, "rb") as f:
            d = pickle.load(f)
        for e in d["eps"]:
            self.add(e)

    def add(self, ep: dict) -> None:
        self.eps.append(ep)
        self.steps += len(ep["label"])
        while self.steps > self.cap and len(self.eps) > 1:
            self.steps -= len(self.eps.popleft()["label"])

    def sample(self, rng: random.Random, max_steps: int) -> list[dict]:
        """Episodes drawn with probability ~ length until the batch holds ~max_steps steps."""
        lens = [len(e["label"]) for e in self.eps]
        out, n = [], 0
        while n < max_steps:
            e = rng.choices(self.eps, weights=lens)[0]
            out.append(e)
            n += len(e["label"])
        return out


def batch_tensors(eps: list[dict], device):
    """List of packed episodes -> time-major padded tensors (T, B, ...) + labels + mask.

    T is rounded up to a multiple of 128 and B to a multiple of 8 (padding is masked out of the loss):
    with free-varying shapes the CUDA caching allocator keeps a block per shape and memory grows
    without bound (measured: 14 GB private after ~1,000 updates on Windows)."""
    import torch

    from submission.doomnet import obs_adapter as A

    T = -(-max(len(e["label"]) for e in eps) // 128) * 128
    B = -(-len(eps) // 8) * 8

    def pad(k, dtype):
        first = eps[0][k]
        arr = np.zeros((T, B) + first.shape[1:], dtype)
        for b, e in enumerate(eps):
            arr[:len(e[k]), b] = e[k]
        return torch.from_numpy(arr).to(device)

    dist = pad("dist", np.float32)
    hit = pad("hit", np.int64)
    rays = torch.zeros(T, B, A.RAY_CHANNELS, A.NUM_RAYS, device=device)
    rays[:, :, 0] = dist
    rays[:, :, 1:].scatter_(2, hit.unsqueeze(2), 1.0)
    x = {"rays": rays, "ent_type": pad("ent_type", np.int64), "ent_feat": pad("ent_feat", np.float32),
         "ent_mask": pad("ent_mask", bool), "scalars": pad("scalars", np.float32),
         "map": unpack_map(pad("map", np.uint8))}
    label = pad("label", np.int64)
    mask = torch.zeros(T, B, device=device)
    for b, e in enumerate(eps):
        mask[:len(e["label"]), b] = 1.0
    return x, label, mask


def unpack_map(packed):
    """(..., 288) uint8 tensor/array of packed bits -> (..., 4, 24, 24) uint8 tensor of 0/1
    (the network's MapEncoder converts to float)."""
    import torch

    arr = packed.cpu().numpy() if torch.is_tensor(packed) else packed
    bits = np.unpackbits(arr, axis=-1)[..., :int(np.prod(MAP_SHAPE))]
    t = torch.from_numpy(bits.reshape(arr.shape[:-1] + MAP_SHAPE))
    return t.to(packed.device if torch.is_tensor(packed) else "cpu")


# ------------------------------------------------------------------ collection
class Collector:
    def __init__(self, n: int, cfg: dict):
        from training.collect import worker

        ctx = mp.get_context("spawn")
        self.n = n
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(i, b, cfg), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.cur = [self._recv(i, ("reset", None)) for i in range(n)]
        self.bufs = [([], []) for _ in range(n)]
        self.mode = [None] * n
        self.h = None                       # student GRU state, kept across iterations (episodes in flight)

    def _recv(self, i, msg):
        self.conns[i].send(msg)
        r = self.conns[i].recv()
        if r[0] == "error":
            raise RuntimeError(f"worker {i} failed:\n{r[1]}")
        return r

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def run(self, policy, device, steps: int, beta: float, p_student: float, rng: random.Random,
            sample: bool = True) -> tuple[list, list, int]:
        """Collect >= `steps` labelled states. Returns (packed episodes, stats, n)."""
        import torch

        from submission.doomnet import obs_adapter as A
        from submission.doomnet.policy import INPUT_KEYS

        n = self.n
        h = None
        if policy is not None:
            h = self.h if self.h is not None else policy.initial_state(n, device)
        for i in range(n):              # episodes in flight are restarted cleanly below
            if self.mode[i] is None:
                self.mode[i] = self._new_mode(beta, p_student, rng)
        done_eps, stats, count = [], [], 0
        while count < steps:
            xs = [c[0] for c in self.cur]
            labels = [c[1] for c in self.cur]
            need_student = policy is not None and any(m != "expert" for m in self.mode)
            if need_student:
                batch = A.stack(xs)
                xt = {k: (unpack_map(batch[k]) if k == "map" else torch.from_numpy(batch[k])).to(device)
                      for k in INPUT_KEYS}
                logits, h = policy.step(xt, h)
                if sample:
                    st_a = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
                else:
                    st_a = logits.argmax(-1).cpu().numpy()
            acts = []
            for i in range(n):
                self.bufs[i][0].append(xs[i])
                self.bufs[i][1].append(labels[i])
                m = self.mode[i]
                if m == "expert" or not need_student:
                    acts.append(labels[i])
                elif m == "student":
                    acts.append(int(st_a[i]))
                else:
                    acts.append(labels[i] if rng.random() < beta else int(st_a[i]))
            for i in range(n):
                self.conns[i].send(("step", acts[i]))
            for i in range(n):
                r = self.conns[i].recv()
                if r[0] == "error":
                    raise RuntimeError(f"worker {i} failed:\n{r[1]}")
                self.cur[i] = r
                if r[2] is not None:                                # episode ended
                    xs_i, ls_i = self.bufs[i]
                    if len(ls_i) >= 8:
                        done_eps.append(EpisodeStore.pack(xs_i, ls_i))
                    stats.append({**r[2], "mode": self.mode[i]})
                    self.bufs[i] = ([], [])
                    self.mode[i] = self._new_mode(beta, p_student, rng)
                    if h is not None:
                        with torch.inference_mode():
                            h[:, i] = 0.0
            count += n
        self.h = h
        return done_eps, stats, count

    @staticmethod
    def _new_mode(beta, p_student, rng):
        if beta >= 1.0:
            return "expert"
        return "student" if rng.random() < p_student else "mix"


# ------------------------------------------------------------------ reporting
def summarize(stats: list[dict]) -> dict:
    out = {}
    groups = defaultdict(list)
    for s in stats:
        groups[(s["mode"], s["family"])].append(s)
    for (mode, fam), ss in sorted(groups.items()):
        n = len(ss)
        mon = [s for s in ss if s["monsters"]]
        keyed = [s for s in ss if s["n_keys"]]
        out[f"{mode}/{fam}"] = {
            "n": n,
            "exit": sum(s["exit"] for s in ss) / n,
            "dead": sum(s["dead"] for s in ss) / n,
            "stalled": sum(s["stalled"] for s in ss) / n,
            "kill_frac": (sum(s["kills"] / s["monsters"] for s in mon) / len(mon)) if mon else None,
            "key_frac": (sum(min(1.0, s["keys"] / s["n_keys"]) for s in keyed) / len(keyed)) if keyed else None,
            "cells": sum(s["cells"] for s in ss) / n,
            "steps": sum(s["steps"] for s in ss) / n,
        }
    return out


def student_score(summary: dict) -> float | None:
    """One number for checkpoint selection: mean over families of exit (or kill fraction on combat)."""
    vals = []
    for k, v in summary.items():
        if not k.startswith("student/"):
            continue
        vals.append(v["kill_frac"] if k.endswith("/combat") and v["kill_frac"] is not None else v["exit"])
    return sum(vals) / len(vals) if vals else None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="il")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--steps0", type=int, default=150_000, help="labelled states in iteration 0 (pure expert)")
    ap.add_argument("--steps", type=int, default=60_000, help="labelled states per later iteration")
    ap.add_argument("--updates0", type=int, default=1500)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--batch-steps", type=int, default=8192, help="states per gradient step (whole episodes)")
    ap.add_argument("--chunk", type=int, default=256,
                    help="backprop chunk: episodes run in order with the GRU state carried across chunks "
                         "(forward = acting computation), gradients truncated at chunk borders; bounds GPU memory")
    ap.add_argument("--cap", type=int, default=1_500_000, help="dataset size (states), FIFO")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, nargs=2, default=[0.5, 0.1], help="expert prob in 'mix' episodes: first, last")
    ap.add_argument("--p-student", type=float, default=0.4, help="share of pure-student episodes after iteration 0")
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--resume", default=None, help="checkpoint to start from (skips the pure-expert warm-up)")
    ap.add_argument("--save-store", action="store_true",
                    help="write the aggregated dataset to runs/<run>/store.pkl after every iteration (~1 KB/state)")
    ap.add_argument("--load-store", default=None, help="start from a saved dataset (store.pkl) instead of an empty one")
    ap.add_argument("--families", default=None, help="json dict family -> weight")
    ap.add_argument("--pools", nargs="+", default=["train"],
                    help="map pools to train on (e.g. train train2: +1,600 layouts, half with floor-pad exits)")
    ap.add_argument("--no-map", action="store_true", help="ablation: network without the spatial map")
    ap.add_argument("--fp32", action="store_true", help="disable mixed precision")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from submission.doomnet.actions import NAMES, NUM_ACTIONS
    from submission.doomnet.policy import DoomPolicy

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    run = REPO / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    fams = json.loads(a.families) if a.families else FAMILIES
    (run / "config.json").write_text(json.dumps({**vars(a), "families": fams}, indent=1))
    log = open(run / "log.jsonl", "a", buffering=1)

    policy = DoomPolicy(use_map=not a.no_map).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr, eps=1e-5)
    amp = device.type == "cuda" and not a.fp32     # fp16 autocast: ~13x faster updates on the dev GPU
    scaler = torch.amp.GradScaler(enabled=amp)
    start_iter = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["state_dict"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_iter = 1
    print(f"[dagger] {sum(p.numel() for p in policy.parameters()):,} params on {device}; run {run}", flush=True)

    store = EpisodeStore(a.cap)
    if a.load_store:
        store.load(a.load_store)
        print(f"[dagger] loaded {len(store.eps)} episodes / {store.steps} states from {a.load_store}", flush=True)
    col = Collector(a.workers, {"families": fams, "pool": a.pools, "seed": a.seed})
    best = -1.0

    def save(path, it, extra=None):
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
                    "opt": opt.state_dict(),
                    "meta": {"arch": "doompolicy-v2", "use_map": policy.use_map, "num_actions": NUM_ACTIONS, "actions": list(NAMES),
                             "iter": it, "greedy": False, **(extra or {})}}, path)

    try:
        for it in range(start_iter, a.iters):
            frac = (it - 1) / max(1, a.iters - 2) if it > 0 else 0.0
            beta = 1.0 if it == 0 else a.beta[0] + (a.beta[1] - a.beta[0]) * min(1.0, max(0.0, frac))
            t0 = time.time()
            policy.eval()
            eps, stats, n = col.run(None if it == 0 else policy, device, a.steps0 if it == 0 else a.steps,
                                    beta, a.p_student, rng)
            for e in eps:
                store.add(e)
            t_col = time.time() - t0
            summ = summarize(stats)
            score = student_score(summ)
            if score is not None and score > best:      # these weights produced the score: save before training
                best = score
                save(run / "best.pt", it - 1, {"student_score": score, "scored_in_iter": it})
                print(f"    new best (score {score:.3f}, weights after iter {it - 1}) -> {run / 'best.pt'}", flush=True)

            policy.train()
            t1 = time.time()
            n_upd = a.updates0 if it == 0 else a.updates
            losses, accs = [], []
            for _ in range(n_upd):
                x, y, m = batch_tensors(store.sample(rng, a.batch_steps), device)
                n_real = m.sum()
                h = policy.initial_state(y.shape[1], device)
                opt.zero_grad(set_to_none=True)
                tot_loss = tot_hit = 0.0
                for s0 in range(0, y.shape[0], a.chunk):
                    mc = m[s0:s0 + a.chunk]
                    if mc.sum() == 0:
                        break                                    # only padding left
                    xc = {k: v[s0:s0 + a.chunk] for k, v in x.items()}
                    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                        logits, _, h = policy(xc, h)
                    logits = logits.float()
                    yc = y[s0:s0 + a.chunk]
                    ce = F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), yc.reshape(-1), reduction="none",
                                         label_smoothing=a.smoothing).view_as(mc)
                    loss = (ce * mc).sum() / n_real
                    scaler.scale(loss).backward()
                    h = h.detach()
                    tot_loss += loss.item()
                    tot_hit += ((logits.argmax(-1) == yc).float() * mc).sum().item()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                losses.append(tot_loss)
                accs.append(tot_hit / n_real.item())
            t_tr = time.time() - t1

            row = {"iter": it, "beta": round(beta, 3), "collected": n, "episodes": len(stats),
                   "store_steps": store.steps, "loss": float(np.mean(losses[-50:])), "acc": float(np.mean(accs[-50:])),
                   "t_collect": round(t_col, 1), "t_train": round(t_tr, 1), "sps": round(n / max(t_col, 1e-6)),
                   "student_score": score, "groups": summ, "time": time.time()}
            log.write(json.dumps(row) + "\n")
            print(f"\n[iter {it:2d}] beta {beta:.2f}  states {n}  store {store.steps}  "
                  f"loss {row['loss']:.3f}  acc {row['acc']:.3f}  ({row['sps']} st/s, collect {t_col:.0f}s, train {t_tr:.0f}s)",
                  flush=True)
            for k, v in summ.items():
                kf = "" if v["kill_frac"] is None else f"  kills {v['kill_frac']:.2f}"
                kk = "" if v["key_frac"] is None else f"  keys {v['key_frac']:.2f}"
                print(f"    {k:20s} n {v['n']:3d}  exit {v['exit']:.2f}  dead {v['dead']:.2f}  stalled {v['stalled']:.2f}"
                      f"{kf}{kk}  cells {v['cells']:.0f}  len {v['steps']:.0f}", flush=True)
            if score is not None:
                print(f"    student score {score:.3f}", flush=True)
            save(run / "latest.pt", it)
            if a.save_store:
                store.save(run / "store.pkl")
            save(run / f"iter{it:02d}.pt", it)              # every iteration, for offline selection
    finally:
        col.close()
        log.close()


if __name__ == "__main__":
    main()
