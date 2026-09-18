# part2_uncertainty/bayes.py
# Naive Bayes posterior P(Damaged | evidence) per §4.2.2.
#
# Prior: P(Damaged) = 0.10
# Likelihood table (fixed constants):
#   Evidence             P(True|Damaged)  P(True|NotDamaged)
#   weight_deviation_hi  0.70             0.05
#   crack_visible        0.85             0.02
#   tilt_triggered       0.60             0.10
#
# Conditional independence assumed (naive Bayes).
#
# Manual verification TC-B1 (w=5, c=0, t=False):
#   Derived booleans: weight_dev_high=F, crack_visible=F, tilt_triggered=F
#   P(evidence|Damaged)    = (1-0.70)*(1-0.85)*(1-0.60) = 0.30*0.15*0.40 = 0.018
#   P(evidence|NotDamaged) = (1-0.05)*(1-0.02)*(1-0.10) = 0.95*0.98*0.90 = 0.8379
#   Numerator_D   = 0.10 * 0.018  = 0.0018
#   Numerator_ND  = 0.90 * 0.8379 = 0.75411
#   P(Damaged|e)  = 0.0018 / (0.0018 + 0.75411) ≈ 0.00238  (very low ✓)
#
# Manual verification TC-B2 (w=90, c=95, t=True):
#   Derived booleans: weight_dev_high=T, crack_visible=T, tilt_triggered=T
#   P(evidence|Damaged)    = 0.70*0.85*0.60 = 0.357
#   P(evidence|NotDamaged) = 0.05*0.02*0.10 = 0.0001
#   Numerator_D   = 0.10 * 0.357  = 0.0357
#   Numerator_ND  = 0.90 * 0.0001 = 0.00009
#   P(Damaged|e)  = 0.0357 / (0.0357 + 0.00009) ≈ 0.9975  (very high ✓)

from __future__ import annotations

PRIOR_DAMAGED = 0.10
PRIOR_NOT_DAMAGED = 0.90

# Likelihood table: key -> (P(True|D), P(True|ND))
LIKELIHOOD = {
    "weight_deviation_high": (0.70, 0.05),
    "crack_visible":         (0.85, 0.02),
    "tilt_triggered":        (0.60, 0.10),
}


def _derive_booleans(reading: dict) -> dict:
    """Derive the three boolean evidence variables from a raw reading."""
    return {
        "weight_deviation_high": reading.get("weight_deviation_pct", 0) > 40,
        "crack_visible":         reading.get("crack_severity_pct", 0) > 15,
        "tilt_triggered":        bool(reading.get("tilt_triggered", False)),
    }


def bayes_posterior(evidence: dict) -> float:
    """
    Compute P(Damaged | evidence) via naive Bayes product of likelihoods.
    `evidence` is a raw sensor reading dict (§4.1 schema).
    Returns a float in [0, 1].
    """
    bools = _derive_booleans(evidence)

    p_e_given_d = 1.0
    p_e_given_nd = 1.0

    for feature, is_true in bools.items():
        p_true_d, p_true_nd = LIKELIHOOD[feature]
        if is_true:
            p_e_given_d *= p_true_d
            p_e_given_nd *= p_true_nd
        else:
            p_e_given_d *= (1.0 - p_true_d)
            p_e_given_nd *= (1.0 - p_true_nd)

    numerator_d = PRIOR_DAMAGED * p_e_given_d
    numerator_nd = PRIOR_NOT_DAMAGED * p_e_given_nd
    denom = numerator_d + numerator_nd

    if denom == 0:
        return 0.0
    return numerator_d / denom
