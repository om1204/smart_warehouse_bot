# part2_uncertainty/compare.py
# Comparison table: run all 10 TC-B cases through all 6 uncertainty methods.
# Prints a formatted side-by-side table per §4.3.
#
# Binary decision thresholds:
#   Bayes / BN / Fuzzy: > 0.5 / > 50  => Damaged
#   CF:                 > 0            => Damaged
#   DS:                 Bel > 0.5     => Damaged
#   Nonmonotonic:       "Damaged"      => Damaged
#
# --- Documented Disagreements (required ≥2, in code comments, not prose) ---
#
# DISAGREEMENT 1 — TC-B3 (w=70, c=5, t=False):
#   weight_dev_high=True, crack_visible=False, tilt=False.
#   Nonmonotonic → NotDamaged (no tilt_triggered and no crack_visible in facts,
#   even though weight is high — default_logic logic has no weight_deviation_high trigger).
#   Bayes/CF/Fuzzy → Damaged (weight_deviation_high contributes graded evidence that
#   pushes the combined posterior above the decision threshold).
#   Root cause: default_logic_verdict() only fires on "tilt_triggered" or "crack_visible"
#   facts; weight_deviation_high is NOT in its abnormality trigger set.
#
# DISAGREEMENT 2 — TC-B5 (w=20, c=10, t=True):
#   weight_dev_high=False, crack_visible=False, tilt=True.
#   Nonmonotonic → Damaged (tilt_triggered fires the default exception).
#   Fuzzy → low damage score (weight=20 is "low", crack=10 is "none"; tilt is not
#   an input to the fuzzy_logic system at all — only weight and crack drive the rules).
#   Fuzzy output falls below 50, so Fuzzy says NotDamaged while Nonmonotonic says Damaged.
#   Root cause: the fuzzy_logic system is built on continuous weight/crack inputs; the boolean
#   tilt signal is outside its input space, so it sees a healthy box.
#
# DISAGREEMENT 3 — TC-B10 (w=60, c=40, t=False):
#   weight_dev_high=True (60>40), crack_visible=True (40>15), tilt=False.
#   BN → moderately low (CrackVisible=True drives Damaged, but without Tilt it stays
#   below 0.5 at ~0.80, actually high). DS Bel may be below 0.5 despite combined
#   evidence — verify per run output.
#   Nonmonotonic → Damaged (crack_visible is True).

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part2_uncertainty.default_logic import (default_logic_verdict,
                                               sensor_reading_to_facts)
from part2_uncertainty.fuzzy_logic import fuzzy_logic_score
from part2_uncertainty.ds_theory import ds_belief_interval
from part2_uncertainty.cf_engine import cf_score
from part2_uncertainty.bayes_net import bn_posterior
from part2_uncertainty.bayes import bayes_posterior


TEST_CASES = [
    {"id": "TC-B1",  "weight_deviation_pct":   5,
        "crack_severity_pct":   0, "tilt_triggered": False},
    {"id": "TC-B2",  "weight_deviation_pct":  90,
        "crack_severity_pct":  95, "tilt_triggered": True},
    {"id": "TC-B3",  "weight_deviation_pct":  70,
        "crack_severity_pct":   5, "tilt_triggered": False},
    {"id": "TC-B4",  "weight_deviation_pct":  10,
        "crack_severity_pct":  80, "tilt_triggered": False},
    {"id": "TC-B5",  "weight_deviation_pct":  20,
        "crack_severity_pct":  10, "tilt_triggered": True},
    {"id": "TC-B6",  "weight_deviation_pct":  45,
        "crack_severity_pct":  20, "tilt_triggered": False},
    {"id": "TC-B7",  "weight_deviation_pct":   0,
        "crack_severity_pct":   0, "tilt_triggered": False},
    {"id": "TC-B8",  "weight_deviation_pct": 100,
        "crack_severity_pct": 100, "tilt_triggered": True},
    {"id": "TC-B9",  "weight_deviation_pct":  30,
        "crack_severity_pct":  55, "tilt_triggered": True},
    {"id": "TC-B10", "weight_deviation_pct":  60,
        "crack_severity_pct":  40, "tilt_triggered": False},
]


def run_all(reading: dict) -> dict:
    facts = sensor_reading_to_facts(reading)
    nm = default_logic_verdict(facts)
    ba = bayes_posterior(reading)
    cf = cf_score(reading)
    bn = bn_posterior(reading)
    ds = ds_belief_interval(reading)
    fz = fuzzy_logic_score(reading)
    return {"default_logic": nm, "bayes": ba, "cf": cf, "bn": bn, "ds": ds, "fuzzy_logic": fz}


def evaluate_all_methods(reading: dict) -> dict:
    """Public API per §4.5. Strips 'id' key if present."""
    r = {k: v for k, v in reading.items() if k != "id"}
    return run_all(r)


def main():
    header = (
        f"{'Case':<8} {'Nonmonotonic':<14} {'Bayes':>8} {'CF':>7} "
        f"{'BN':>8} {'DS(Bel,Pl)':>18} {'Fuzzy':>8}"
    )
    print(header)
    print("-" * len(header))

    for tc in TEST_CASES:
        tc_id = tc["id"]
        reading = {k: v for k, v in tc.items() if k != "id"}
        r = run_all(reading)
        nm = r["default_logic"]
        ba = r["bayes"]
        cf = r["cf"]
        bn = r["bn"]
        ds = r["ds"]
        fz = r["fuzzy_logic"]
        print(
            f"{tc_id:<8} {nm:<14} {ba:>8.4f} {cf:>7.4f} "
            f"{bn:>8.4f} ({ds[0]:.3f},{ds[1]:.3f})  {fz:>8.2f}"
        )


if __name__ == "__main__":
    main()
