"""Curriculum controller: gates, deadlines, replay share, persistence."""
import collections

import pytest

from training import curriculum as C


def groups(fam, exit_=0.0, survival=1.0, kill_frac=None, proxy=0.0, variants=None):
    g = {fam: {"exit": {"rate": exit_}, "survival": survival, "kill_frac": kill_frac, "proxy": {"mean": proxy}}}
    for v, r in (variants or {}).items():
        g[f"{fam}/{v}"] = {"exit": {"rate": r}, "survival": 1.0, "kill_frac": None, "proxy": {"mean": 0.0}}
    return g


def test_gate_rules():
    assert C.gate_met("1", groups("combat", survival=0.8, kill_frac=0.65))[0]
    assert not C.gate_met("1", groups("combat", survival=0.8, kill_frac=0.55))[0]
    assert not C.gate_met("1", groups("combat", survival=0.6, kill_frac=0.9))[0]
    assert C.gate_met("2", groups("rooms", exit_=0.6))[0] and not C.gate_met("2", groups("rooms", exit_=0.59))[0]
    assert C.gate_met("3", groups("keydoor", exit_=0.55, variants={"A": 0.8, "B": 0.3, "C": 0.5}))[0]
    assert not C.gate_met("3", groups("keydoor", exit_=0.55, variants={"A": 0.9, "B": 0.2, "C": 0.5}))[0]
    assert not C.gate_met("4", groups("mixed", exit_=1.0))[0]


def test_gate_schedule_and_advance(tmp_path):
    cur = C.Curriculum(tmp_path / "c.json", t0=1000.0)
    cur.begin(5_000)
    assert not cur.gate_due(5_000, 5_000 + C.GATE_EVERY - 1)
    assert cur.gate_due(5_000 + C.GATE_EVERY - 1, 5_000 + C.GATE_EVERY + 10)
    d = cur.decide(10**6, groups("combat", survival=0.9, kill_frac=0.3, proxy=5), now=1000.0)
    assert d["action"] == "continue" and d["improved"]
    d = cur.decide(2 * 10**6, groups("combat", survival=0.9, kill_frac=0.7, proxy=4), now=1000.0)
    assert d["action"] == "advance" and d["to"] == "2" and not d["improved"]
    assert cur.stage == "2" and cur.state["stage_start_step"] == 2 * 10**6
    # persisted
    again = C.Curriculum(tmp_path / "c.json", t0=0.0)
    assert again.stage == "2" and again.state["t0"] == 1000.0


def test_deadline_advances_without_gate(tmp_path):
    cur = C.Curriculum(tmp_path / "c.json", t0=0.0, start_stage="2")
    cur.begin(0)
    assert not cur.deadline_passed(now=9.9 * 3600)
    assert cur.deadline_passed(now=10.1 * 3600)
    d = cur.decide(123, None, now=10.1 * 3600)
    assert d["action"] == "advance" and d["to"] == "3" and "deadline" in d["why"]


def test_final_stage_has_no_gates(tmp_path):
    cur = C.Curriculum(tmp_path / "c.json", t0=0.0, start_stage="4")
    cur.begin(0)
    assert not cur.gate_due(0, 10 * C.GATE_EVERY) and not cur.deadline_passed(now=10**9)


def test_schedule_replays_earlier_stages_at_20_percent():
    sched = C.build_schedule("3")
    fams = collections.Counter(e["family"] for e in sched)
    share = (fams["combat"] + fams["rooms"]) / len(sched)
    assert fams["keydoor"] == 120 and share == pytest.approx(0.20, abs=0.01)
    assert C.build_schedule("3") == sched                       # deterministic
    assert collections.Counter(e["family"] for e in C.build_schedule("1")) == {"combat": 100}


def test_worker_schedules_one_wad_each_and_20_percent_replay():
    ws = C.build_worker_schedules("3", 24)
    assert len(ws) == 24
    assert all(len({e["wad"] for e in w}) == 1 for w in ws)          # no WAD switching inside a worker
    fams = collections.Counter(w[0]["family"] for w in ws)
    assert fams["keydoor"] == 19 and fams["combat"] + fams["rooms"] == 5   # round(0.2 * 24) = 5
    assert C.build_worker_schedules("1", 24) == [ws_ for ws_ in C.build_worker_schedules("1", 24)]
    assert {w[0]["family"] for w in C.build_worker_schedules("1", 24)} == {"combat"}


def test_stage4_split_mixed_freedoom_replay():
    ws = C.build_worker_schedules("4", 24)
    fams = collections.Counter(w[0]["family"] for w in ws)
    assert fams["freedoom"] == 4 and fams["mixed"] == 15
    assert fams["combat"] + fams["rooms"] + fams["keydoor"] == 5
    assert all(len({e["wad"] for e in w}) == 1 for w in ws)
    from training.maps.gen import load_index
    val = {e["map"] for e in load_index("validation", "freedoom")} | {e["map"] for e in load_index("test", "freedoom")}
    assert not val & {e["map"] for e in load_index("train", "freedoom")}        # unseen levels stay unseen
