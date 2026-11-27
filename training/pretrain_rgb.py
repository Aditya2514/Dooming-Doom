"""Supervised pre-training of the student's RGB branch as an exit detector (training only).

Rays and entities have no exit class; exits are only visible in the RGB frame (the official sample
map's exit is a red GATE3 floor pad). Learning to recognise them through the imitation loss alone is
slow (an exit is on screen in ~3% of states), so this script first trains the RGB encoder with the
auxiliary head (policy.rgb_aux, never used to act) to answer "is the exit on screen, and in which
quarter of the view?", from frames of the privileged teacher playing generated maps (labels from the
generator's exit position + line of sight along the official rays, training/expert.py
exit_view_class). The trained branch is then written into a policy checkpoint; DAgger continues from it.

    python -m training.pretrain_rgb --init submission/weights/actor.pt --out runs/rgbpre.pt
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time

import numpy as np


def _collect(args):
    """One process: teacher episodes (with some random actions for varied views) -> (rgb, label)."""
    pool, fams, n_frames, seed, eps = args
    from submission.doomnet import obs_adapter as A
    from submission.doomnet.actions import NUM_ACTIONS, to_buttons
    from training.expert import Expert, exit_view_class
    from training.maprunner import MapRunner
    from training.maps.gen import load_index

    rng = random.Random(seed)
    idx = [e for f in fams for e in load_index(pool, f)]
    runner, rgbs, labs = MapRunner(), [], []
    try:
        while len(labs) < n_frames:
            e = rng.choice(idx)
            env = runner.load(e)
            obs, _ = env.reset(seed=rng.randrange(2**31))
            ex = Expert(e["route"])
            ex.reset()
            exit_xy = e["route"].get("exit_xy")
            for _ in range(1050):
                ps = env.privileged_state()
                rgbs.append(A.downsample_rgb(obs["rgb"]))
                labs.append(exit_view_class(obs, ps, exit_xy))
                a = ex.act(obs, ps)
                if rng.random() < eps:
                    a = rng.randrange(NUM_ACTIONS)
                    ex.executed(a)
                obs, _, te, tr, _ = env.step(to_buttons(a))
                if te or tr or len(labs) >= n_frames:
                    break
    finally:
        runner.close()
    return np.stack(rgbs), np.asarray(labs, np.int64)


def collect(pool, fams, n, workers, seed, eps=0.15):
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as p:
        parts = p.map(_collect, [(pool, fams, n // workers, seed * 100 + i, eps) for i in range(workers)])
    return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="policy checkpoint (widened with an RGB branch if it has none)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120_000)
    ap.add_argument("--val-frames", type=int, default=20_000)
    ap.add_argument("--pools", nargs="+", default=["train", "train2"])
    ap.add_argument("--families", nargs="+", default=["rooms", "keyeasy", "keydoor", "mixed", "combat"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None, help="npz file: reuse collected frames if it exists, else write it")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from submission.doomnet.policy import EXIT_BINS, policy_from_state_dict

    torch.manual_seed(a.seed)
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    pol = policy_from_state_dict(ck["state_dict"], use_rgb=True)

    t0 = time.time()
    from pathlib import Path
    if a.cache and Path(a.cache).is_file():
        d = np.load(a.cache)
        X, Y, Xv, Yv = (torch.from_numpy(d[k]) for k in ("X", "Y", "Xv", "Yv"))
    else:
        per_pool = a.frames // len(a.pools)
        parts = [collect(p, a.families, per_pool, a.workers, a.seed + i) for i, p in enumerate(a.pools)]
        X = torch.from_numpy(np.concatenate([p[0] for p in parts]))
        Y = torch.from_numpy(np.concatenate([p[1] for p in parts]))
        Xv, Yv = (torch.from_numpy(v) for v in collect("validation", a.families, a.val_frames, a.workers, a.seed + 99))
        if a.cache:
            np.savez(a.cache, X=X.numpy(), Y=Y.numpy(), Xv=Xv.numpy(), Yv=Yv.numpy())
    print(f"[pretrain_rgb] {len(Y)} train frames ({(Y > 0).float().mean():.3f} exit on screen), "
          f"{len(Yv)} validation frames ({(Yv > 0).float().mean():.3f}); collected in {time.time() - t0:.0f}s", flush=True)

    w = torch.ones(EXIT_BINS + 1)
    w[1:] = a.pos_weight
    params = list(pol.rgb.parameters()) + list(pol.rgb_aux.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)

    def evaluate():
        pol.eval()
        with torch.no_grad():
            pr = torch.cat([pol.exit_logits(Xv[i:i + 2048]).argmax(-1) for i in range(0, len(Yv), 2048)])
        vis, not_vis = Yv > 0, Yv == 0
        det = ((pr > 0) & vis).sum().item() / max(1, vis.sum().item())               # recall: exit on screen
        fa = ((pr > 0) & not_vis).sum().item() / max(1, not_vis.sum().item())        # false alarms
        acc_bin = ((pr == Yv) & vis).sum().item() / max(1, vis.sum().item())          # right quarter of the view
        prec = ((pr > 0) & vis).sum().item() / max(1, (pr > 0).sum().item())
        pol.train()
        return {"recall": round(det, 3), "precision": round(prec, 3), "false_alarm": round(fa, 4),
                "bin_acc": round(acc_bin, 3)}

    print("  before:", evaluate(), flush=True)
    for ep in range(a.epochs):
        perm = torch.randperm(len(Y))
        tot = 0.0
        for i in range(0, len(Y), a.batch):
            b = perm[i:i + a.batch]
            xb = X[b]
            if random.random() < 0.5:                         # mirror: left/right bins swap
                xb = xb.flip(-1)
                yb = torch.where(Y[b] > 0, EXIT_BINS + 1 - Y[b], Y[b])
            else:
                yb = Y[b]
            loss = F.cross_entropy(pol.exit_logits(xb), yb, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        print(f"  epoch {ep + 1}: loss {tot / len(Y):.4f}  validation {evaluate()}", flush=True)

    meta = {**ck.get("meta", {}), "rgb_pretrain": {"frames": len(Y), "pools": a.pools, "val": evaluate()},
            "use_rgb": True}
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in pol.state_dict().items()}, "meta": meta}, a.out)
    print(f"[pretrain_rgb] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
