# part2_uncertainty/cf_engine.py
# Certainty Factor (CF) model per §4.2.3.
#
# Per-evidence CFs (True / False):
#   crack_visible:         +0.9 / -0.3
#   tilt_triggered:        +0.6 / -0.2
#   weight_deviation_high: +0.5 / -0.1
#
# Serial attenuation: CF_after_rule = CF_evidence * 0.9  (rule confidence = 0.9)
#
# Parallel combination (pairwise, left-to-right):
#   both >= 0: CF1 + CF2*(1 - CF1)
#   both <  0: CF1 + CF2*(1 + CF1)
#   opposite : (CF1 + CF2) / (1 - min(|CF1|, |CF2|))

from __future__ import annotations

# (CF_if_true, CF_if_false) per evidence variable
EVIDENCE_CF = {
    "crack_visible":         (0.9, -0.3),
    "tilt_triggered":        (0.6, -0.2),
    "weight_deviation_high": (0.5, -0.1),
}

RULE_CONFIDENCE = 0.9   # serial attenuation factor for all three rules


def _combine(cf1: float, cf2: float) -> float:
    """Parallel combination of two CFs."""
    if cf1 >= 0 and cf2 >= 0:
        return cf1 + cf2 * (1.0 - cf1)
    elif cf1 < 0 and cf2 < 0:
        return cf1 + cf2 * (1.0 + cf1)
    else:
        return (cf1 + cf2) / (1.0 - min(abs(cf1), abs(cf2)))


def _derive_booleans(reading: dict) -> dict:
    return {
        "weight_deviation_high": reading.get("weight_deviation_pct", 0) > 40,
        "crack_visible":         reading.get("crack_severity_pct", 0) > 15,
        "tilt_triggered":        bool(reading.get("tilt_triggered", False)),
    }


def cf_score(evidence: dict) -> float:
    """
    Compute combined Certainty Factor for 'Damaged' given `evidence`.
    Returns a float in [-1, 1].

    Order of combination: crack_visible → tilt_triggered → weight_deviation_high.
    Each is first attenuated by RULE_CONFIDENCE (serial combination), then
    combined pairwise (parallel combination).
    """
    bools = _derive_booleans(evidence)
    order = ["crack_visible", "tilt_triggered", "weight_deviation_high"]

    attenuated_cfs = []
    for feat in order:
        cf_true, cf_false = EVIDENCE_CF[feat]
        raw_cf = cf_true if bools[feat] else cf_false
        attenuated = raw_cf * RULE_CONFIDENCE
        attenuated_cfs.append(attenuated)

    # Pairwise combination left-to-right
    combined = attenuated_cfs[0]
    for i in range(1, len(attenuated_cfs)):
        combined = _combine(combined, attenuated_cfs[i])

    # Clamp to [-1, 1] as a safety guard
    return max(-1.0, min(1.0, combined))
