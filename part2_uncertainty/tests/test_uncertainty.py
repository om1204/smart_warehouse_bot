# part2_uncertainty/tests/test_uncertainty.py
# Unit tests for Module B — all 6 uncertainty methods.

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from part2_uncertainty.default_logic import (monotonic_verdict,
                                               default_logic_verdict)
from part2_uncertainty.fuzzy_logic import fuzzy_logic_score
from part2_uncertainty.ds_theory import ds_belief_interval
from part2_uncertainty.cf_engine import cf_score
from part2_uncertainty.bayes_net import bn_posterior
from part2_uncertainty.bayes import bayes_posterior


# ---------------------------------------------------------------------------
# Test cases (§4.3)
# ---------------------------------------------------------------------------
TC = [
    {"id": "TC-B1",  "weight_deviation_pct": 5,
        "crack_severity_pct": 0,  "tilt_triggered": False},
    {"id": "TC-B2",  "weight_deviation_pct": 90,
        "crack_severity_pct": 95, "tilt_triggered": True},
    {"id": "TC-B3",  "weight_deviation_pct": 70,
        "crack_severity_pct": 5,  "tilt_triggered": False},
    {"id": "TC-B4",  "weight_deviation_pct": 10,
        "crack_severity_pct": 80, "tilt_triggered": False},
    {"id": "TC-B5",  "weight_deviation_pct": 20,
        "crack_severity_pct": 10, "tilt_triggered": True},
    {"id": "TC-B6",  "weight_deviation_pct": 45,
        "crack_severity_pct": 20, "tilt_triggered": False},
    {"id": "TC-B7",  "weight_deviation_pct": 0,
        "crack_severity_pct": 0,  "tilt_triggered": False},
    {"id": "TC-B8",  "weight_deviation_pct": 100,
        "crack_severity_pct": 100, "tilt_triggered": True},
    {"id": "TC-B9",  "weight_deviation_pct": 30,
        "crack_severity_pct": 55, "tilt_triggered": True},
    {"id": "TC-B10", "weight_deviation_pct": 60,
        "crack_severity_pct": 40, "tilt_triggered": False},
]


def reading(tc_dict):
    return {k: v for k, v in tc_dict.items() if k != "id"}


# ---------------------------------------------------------------------------
# Nonmonotonic
# ---------------------------------------------------------------------------
class TestNonmonotonic:
    def test_no_damage_facts(self):
        assert default_logic_verdict(set()) == "NotDamaged"

    def test_tilt_triggers_damaged(self):
        assert default_logic_verdict({"tilt_triggered"}) == "Damaged"

    def test_crack_triggers_damaged(self):
        assert default_logic_verdict({"crack_visible"}) == "Damaged"

    def test_retraction(self):
        """Adding then removing tilt_triggered flips the verdict (default_logicity)."""
        facts = {"tilt_triggered"}
        assert default_logic_verdict(facts) == "Damaged"
        facts.discard("tilt_triggered")
        assert default_logic_verdict(
            facts) == "NotDamaged"  # verdict retracted!

    def test_monotonic_does_not_retract(self):
        """monotonic_verdict does NOT retract even after fact removal."""
        kb = ["crack_detected"]
        assert monotonic_verdict(kb) == "Damaged"
        # Simulate removing the fact from the working set but NOT from kb
        facts = set()  # empty — fact removed
        # default_logic correctly returns NotDamaged
        assert default_logic_verdict(facts) == "NotDamaged"
        # But monotonic still returns Damaged (stale conclusion)
        assert monotonic_verdict(kb) == "Damaged"


# ---------------------------------------------------------------------------
# Bayes
# ---------------------------------------------------------------------------
class TestBayes:
    @pytest.mark.parametrize("tc", TC)
    def test_returns_probability(self, tc):
        p = bayes_posterior(reading(tc))
        assert 0.0 <= p <= 1.0, f"{tc['id']}: out of range {p}"

    def test_tc_b1_approx(self):
        # Expected ≈ 0.00238 (see bayes.py manual verification comment)
        p = bayes_posterior(
            {"weight_deviation_pct": 5, "crack_severity_pct": 0, "tilt_triggered": False})
        assert p < 0.01, f"TC-B1 expected near 0, got {p}"

    def test_tc_b2_approx(self):
        # Expected ≈ 0.9975
        p = bayes_posterior({"weight_deviation_pct": 90,
                            "crack_severity_pct": 95, "tilt_triggered": True})
        assert p > 0.99, f"TC-B2 expected near 1, got {p}"

    def test_high_evidence_increases_posterior(self):
        low_p = bayes_posterior(
            {"weight_deviation_pct": 5, "crack_severity_pct": 0, "tilt_triggered": False})
        high_p = bayes_posterior(
            {"weight_deviation_pct": 90, "crack_severity_pct": 95, "tilt_triggered": True})
        assert low_p < high_p


# ---------------------------------------------------------------------------
# Certainty Factors
# ---------------------------------------------------------------------------
class TestCF:
    @pytest.mark.parametrize("tc", TC)
    def test_in_range(self, tc):
        cf = cf_score(reading(tc))
        assert -1.0 <= cf <= 1.0, f"{tc['id']}: CF out of [-1,1]: {cf}"

    def test_all_false_negative(self):
        # w=5, c=0, t=False → all CFs negative
        cf = cf_score({"weight_deviation_pct": 5,
                      "crack_severity_pct": 0, "tilt_triggered": False})
        assert cf < 0

    def test_all_true_positive(self):
        cf = cf_score({"weight_deviation_pct": 90,
                      "crack_severity_pct": 95, "tilt_triggered": True})
        assert cf > 0


# ---------------------------------------------------------------------------
# Bayesian Network
# ---------------------------------------------------------------------------
class TestBN:
    @pytest.mark.parametrize("tc", TC)
    def test_returns_probability(self, tc):
        p = bn_posterior(reading(tc))
        assert 0.0 <= p <= 1.0, f"{tc['id']}: BN out of range {p}"

    def test_high_crack_increases_p(self):
        low_p = bn_posterior(
            {"weight_deviation_pct": 0, "crack_severity_pct": 0,  "tilt_triggered": False})
        high_p = bn_posterior(
            {"weight_deviation_pct": 0, "crack_severity_pct": 90, "tilt_triggered": True})
        assert low_p < high_p


# ---------------------------------------------------------------------------
# Dempster-Shafer
# ---------------------------------------------------------------------------
class TestDS:
    @pytest.mark.parametrize("tc", TC)
    def test_bel_le_pl(self, tc):
        bel, pl = ds_belief_interval(reading(tc))
        assert bel <= pl, f"{tc['id']}: Bel({bel}) > Pl({pl})"
        assert 0.0 <= bel <= 1.0
        assert 0.0 <= pl <= 1.0


# ---------------------------------------------------------------------------
# Fuzzy
# ---------------------------------------------------------------------------
class TestFuzzy:
    @pytest.mark.parametrize("tc", TC)
    def test_in_range(self, tc):
        score = fuzzy_logic_score(reading(tc))
        assert 0.0 <= score <= 100.0, f"{tc['id']}: fuzzy_logic score out of [0,100]: {score}"

    def test_high_inputs_give_high_score(self):
        low_score = fuzzy_logic_score(
            {"weight_deviation_pct": 0,   "crack_severity_pct": 0,   "tilt_triggered": False})
        high_score = fuzzy_logic_score(
            {"weight_deviation_pct": 100, "crack_severity_pct": 100, "tilt_triggered": True})
        assert low_score < high_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
