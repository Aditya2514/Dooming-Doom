"""Curriculum controller (Spec v2.1, Curriculum mapped to the tiers; plan deadlines).

    Stage 1 Combat        combat maps      gamma 0.99   gate: survival >= 0.70 and kill fraction >= 0.60
    Stage 2 Multi-room    rooms maps       gamma 0.995  gate: exit >= 0.60
    Stage 3 Dependencies  keydoor A-C      gamma 0.995  gate: exit >= 0.50 overall, every variant >= 0.30
    Stage 4 Mixed         mixed maps + Freedoom 2 levels (~1/6 of workers)   gamma 0.995  final stage
Earlier stages stay in the schedule at 20% so skills don't fade.

Gates: every GATE_EVERY decisions of stage-local training, on predefined checkpoints only,
the current stage's VALIDATION pool is played through the submitted DoomAgent (matched
episodes; training/evaluate.py). A gate batch is never re-run until it happens to pass.
Wall-clock deadlines (plan note; the spec gives none): a stage advances anyway when the
run passes T + its deadline hours. T is stored in the state file on first use, so restarts
(also after midnight) keep the same anchor.

The state file (runs/<run>/curriculum.json) records every gate and decision.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

GATE_EVERY = int(os.environ.get("DOOM_GATE_EVERY", 1_000_000))     # env override: smoke tests only
REPLAY = 0.20
STAGES = {
    "1": {"family": "combat", "gamma": 0.99, "deadline_h": 6.0, "next": "2", "seeds": 2},
    "2": {"family": "rooms", "gamma": 0.995, "deadline_h": 10.0, "next": "3", "seeds": 2},
    # Stage 3 fix (26 Nov, measured on the 5060): with 20% replay, exiting collapsed within 360k
    # decisions (training exit 0.56 -> 0.00; validation rooms exit 0.80 -> 0.10), because on
    # key-door maps the exit rarely pays. So: half the workers replay earlier stages (rooms-heavy),
    # variant A (key on the way) dominates early key-door play, and gates also watch rooms exits.
    "3": {"family": "keydoor", "gamma": 0.995, "deadline_h": 15.0, "next": "4", "seeds": 2,
          "replay": 0.5, "replay_weights": {"rooms": 3, "combat": 1},
          "variant_weights": {"A": 4, "C": 2, "B": 1}, "monitor": ["rooms"]},
    "4": {"family": "mixed", "gamma": 0.995, "deadline_h": None, "next": None, "seeds": 1,
          "extra": {"freedoom": 1 / 6}},      # Freedoom levels are slow (~100-250 decisions/s per env)
}
ORDER = ["1", "2", "3", "4"]


def gate_met(stage: str, groups: dict) -> tuple[bool, str]:
    fam = STAGES[stage]["family"]
    g = groups.get(fam)
    if g is None:
        return False, "no results"
    if stage == "1":
        ok = g["survival"] >= 0.70 and (g["kill_frac"] or 0) >= 0.60
        return ok, f"survival {g['survival']:.2f} (>=0.70), kill fraction {(g['kill_frac'] or 0):.2f} (>=0.60)"
    if stage == "2":
        return g["exit"]["rate"] >= 0.60, f"exit {g['exit']['rate']:.2f} (>=0.60)"
    if stage == "3":
        var = {k.split("/")[1]: v["exit"]["rate"] for k, v in groups.items() if k.startswith(f"{fam}/")}
        ok = g["exit"]["rate"] >= 0.50 and var and min(var.values()) >= 0.30
        return bool(ok), f"exit {g['exit']['rate']:.2f} (>=0.50), variants {var} (each >=0.30)"
    return False, "final stage"


def stage_score(stage: str, groups: dict) -> float:
    """Selection metric within a stage: the stage family's mean proxy score on validation."""
    g = groups.get(STAGES[stage]["family"])
    return float(g["proxy"]["mean"]) if g else float("-inf")


def _weighted_variants(entries: list[dict], weights: dict | None) -> list[dict]:
    """Repeat each map by its variant's weight (e.g. A x4, C x2, B x1); unlisted variants x1."""
    if not weights:
        return entries
    out = []
    for e in entries:
        out += [e] * int(weights.get(e.get("variant"), 1))
    random.Random(1234).shuffle(out)              # interleave: no map repeated back-to-back by design
    return out


def _replay_pools(stage: str, earlier: list[str]) -> list[list[dict]]:
    """Replay families in proportion to the stage's replay_weights (default: equal)."""
    from training.maps.gen import index_path, load_index

    weights = STAGES[stage].get("replay_weights", {})
    pools = []
    for f in earlier:
        if index_path("train", f).exists():
            pools += [load_index("train", f)] * int(weights.get(f, 1))
    return pools


def build_worker_schedules(stage: str, n_envs: int, seed: int = 0) -> list[list[dict]]:
    """Per-worker schedules, each inside ONE WAD (no engine restarts): round(replay x N) workers
    replay earlier families (in proportion to replay_weights), the rest play the current stage's
    maps (variants weighted by variant_weights)."""
    from training.maps.gen import index_path, load_index

    fam = STAGES[stage]["family"]
    earlier = [STAGES[s]["family"] for s in ORDER[:ORDER.index(stage)]]
    if not index_path("train", fam).exists():
        if stage != "4":
            raise SystemExit(f"no training maps for stage {stage} ({fam}): run python -m training.maps.gen --all")
        print("WARNING: no Stage 4 'mixed' maps yet: Stage 4 trains on every earlier family instead")
        fams = [f for f in earlier if index_path("train", f).exists()]
        return [load_index("train", fams[i % len(fams)]) for i in range(n_envs)]
    main = _weighted_variants(load_index("train", fam), STAGES[stage].get("variant_weights"))
    pools = _replay_pools(stage, earlier)
    n_rep = round(n_envs * STAGES[stage].get("replay", REPLAY)) if pools else 0
    extra = []
    for f, share in STAGES[stage].get("extra", {}).items():
        if index_path("train", f).exists():
            extra += [load_index("train", f)] * max(1, round(n_envs * share))
    rng = random.Random(seed * 7919 + int(stage))
    workers = [pools[i % len(pools)] for i in range(n_rep)] + extra + [main] * (n_envs - n_rep - len(extra))
    rng.shuffle(workers)
    return workers


def build_schedule(stage: str, seed: int = 0) -> list[dict]:
    """Current stage's training maps + earlier stages at REPLAY share, interleaved (fixed shuffle).
    One shared list that mixes WADs: training uses build_worker_schedules."""
    from training.maps.gen import index_path, load_index

    fam = STAGES[stage]["family"]
    if not index_path("train", fam).exists():
        if stage != "4":
            raise SystemExit(f"no training maps for stage {stage} ({fam}): run python -m training.maps.gen --all")
        print("WARNING: no Stage 4 'mixed' maps yet: Stage 4 trains on every earlier family instead")
        out = [e for s in ORDER[:3] for e in load_index("train", STAGES[s]["family"])]
        random.Random(seed).shuffle(out)
        return out
    main = load_index("train", fam)
    earlier = [STAGES[s]["family"] for s in ORDER[:ORDER.index(stage)]]
    rng = random.Random(seed * 7919 + int(stage))
    extra = []
    if earlier:
        n_extra = round(len(main) * REPLAY / (1 - REPLAY))
        pools = [load_index("train", f) for f in earlier if index_path("train", f).exists()]
        for i in range(n_extra):
            p = pools[i % len(pools)]
            extra.append(p[rng.randrange(len(p))])
    sched = main + extra
    rng.shuffle(sched)
    return sched


class Curriculum:
    def __init__(self, path: Path, t0: float | None = None, start_stage: str = "1"):
        self.path = Path(path)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"stage": start_stage, "t0": t0 if t0 is not None else time.time(),
                          "stage_start_step": None, "best": {}, "history": []}
            self.save()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.path)

    @property
    def stage(self) -> str:
        return self.state["stage"]

    def begin(self, global_step: int):
        if self.state["stage_start_step"] is None:
            self.state["stage_start_step"] = global_step
            self.log("stage_start", global_step, stage=self.stage)

    def log(self, event: str, step: int, **kw):
        self.state["history"].append({"event": event, "step": step, "t": time.time(), **kw})
        self.save()

    def hours(self, now: float | None = None) -> float:
        return ((now or time.time()) - self.state["t0"]) / 3600

    def gate_due(self, prev_step: int, step: int) -> bool:
        s0 = self.state["stage_start_step"] or 0
        return STAGES[self.stage]["next"] is not None and (step - s0) // GATE_EVERY > (prev_step - s0) // GATE_EVERY

    def deadline_passed(self, now: float | None = None) -> bool:
        d = STAGES[self.stage]["deadline_h"]
        return d is not None and self.hours(now) >= d

    def decide(self, step: int, groups: dict | None, now: float | None = None) -> dict:
        """After a gate eval (groups) or a deadline check (groups=None): advance or continue."""
        st = self.stage
        met, why = gate_met(st, groups) if groups is not None else (False, "no gate eval")
        if groups is not None:
            score = stage_score(st, groups)
            best = self.state["best"].get(st)
            improved = best is None or score > best["score"]
            if improved:
                self.state["best"][st] = {"score": score, "step": step}
            self.log("gate", step, stage=st, met=met, why=why, score=score, improved=improved)
        else:
            improved = False
        if met or self.deadline_passed(now):
            nxt = STAGES[st]["next"]
            reason = f"gate met: {why}" if met else f"deadline T+{STAGES[st]['deadline_h']}h passed ({why})"
            self.state["stage"] = nxt
            self.state["stage_start_step"] = step
            self.log("advance", step, frm=st, to=nxt, why=reason)
            return {"action": "advance", "from": st, "to": nxt, "why": reason, "improved": improved}
        return {"action": "continue", "why": why, "improved": improved}
