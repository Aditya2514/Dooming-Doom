"""Export a training checkpoint's acting network to submission/weights/actor.pt.

    .venv\\Scripts\\python.exe -m training.export runs/base/latest.pt            # threshold at 0.5
    .venv\\Scripts\\python.exe -m training.export runs/base/latest.pt --sample   # sampling mode

Only the actor (encoders, GRU, policy) is exported; critic and aux head stay in training.
The file's SHA-256 is printed so the frozen candidate can be recorded (Phase 5).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "submission" / "weights" / "actor.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--sample", action="store_true", help="sample actions instead of thresholding at 0.5")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    import torch

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    actor = {k[len("actor."):]: v for k, v in d["model"].items() if k.startswith("actor.")}
    meta = {"greedy": not a.sample, "source": str(a.ckpt), "global_step": d.get("global_step"),
            "arch": "baseline", **{f"run_{k}": v for k, v in d.get("meta", {}).items()}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": actor, "meta": meta}, out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB), mode {'sample' if a.sample else 'threshold'}, "
          f"step {meta['global_step']}, sha256 {h[:16]}")


if __name__ == "__main__":
    main()
