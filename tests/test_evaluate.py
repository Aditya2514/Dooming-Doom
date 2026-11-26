"""Eval runner helpers: Wilson intervals and the proxy score."""
import pytest

from training.evaluate import proxy_score, summarize, wilson


def test_wilson_known_values():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)


def ep(**kw):
    base = dict(map="m", family="rooms", variant=None, seed=0, ticks=2100, tic=1050, steps=262, exit=False,
                death=False, tick_cap=False, cleared=False, health=100.0, kills=0, monsters=0, kill_frac=0.0,
                damage=0.0, keys=0, doors=0, locked_doors=0, cells=0, act_ms=1.0)
    base.update(kw)
    return base


def test_proxy_score_categories():
    assert proxy_score(ep(exit=True, health=50.0)) == pytest.approx(100 + 5 + 5)
    assert proxy_score(ep(keys=1, doors=2, locked_doors=1)) == pytest.approx(10 + 6 + 10)
    assert proxy_score(ep(cells=100)) == pytest.approx(30)                       # capped
    assert proxy_score(ep(kills=2, damage=40, death=True, tic=525)) == pytest.approx(10 - 2 - 20 * 0.75)


def test_summarize_groups():
    eps = [ep(exit=True, proxy=110.0), ep(proxy=0.0), ep(death=True, monsters=2, kills=1, kill_frac=0.5, proxy=-5.0)]
    s = summarize(eps)
    assert s["n"] == 3 and s["exit"]["count"] == 1 and s["survival"] == pytest.approx(2 / 3)
    assert s["kill_frac"] == pytest.approx(0.5) and s["median_time_to_exit_s"] == pytest.approx(30.0)
