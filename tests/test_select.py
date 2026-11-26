"""Phase 5 selection rule (frozen before results): tier-weighted proxy, ties -> variant B."""
import pytest

from training.select import TIER_WEIGHT, variant_b, weighted_score


def g(**proxy):
    return {f: {"proxy": {"mean": v}} for f, v in proxy.items()}


def test_tier_weighted_score():
    s = weighted_score(g(combat=10, rooms=20, keydoor=30, mixed=40, freedoom=0))
    w = TIER_WEIGHT
    assert s == pytest.approx((10 * w["combat"] + 20 * w["rooms"] + 30 * w["keydoor"] + 40 * w["mixed"])
                              / sum(w.values()))
    assert weighted_score(g(rooms=5)) == pytest.approx(5)      # missing families are skipped


def test_harder_tiers_weigh_more():
    assert weighted_score(g(combat=0, keydoor=10)) > weighted_score(g(combat=10, keydoor=0))


def test_variant_b_lookup():
    assert variant_b({"keydoor/B": {"exit": {"rate": 0.4}}}) == 0.4 and variant_b({}) == 0.0
