# part2_uncertainty/ds_theory.py
# Dempster-Shafer evidence theory per §4.2.5.
#
# Frame Θ = {D (Damaged), N (NotDamaged)}
# Focal elements: {"D"}, {"N"}, {"D","N"} (the full frame Θ = "Uncertain")
#
# BPAs per sensor (True / False):
#   weight (True):  m(D)=0.5, m(N)=0.2, m(Θ)=0.3
#   weight (False): m(D)=0.1, m(N)=0.6, m(Θ)=0.3
#   crack  (True):  m(D)=0.8, m(N)=0.05,m(Θ)=0.15
#   crack  (False): m(D)=0.05,m(N)=0.7, m(Θ)=0.25
#   tilt   (True):  m(D)=0.6, m(N)=0.1, m(Θ)=0.3
#   tilt   (False): m(D)=0.1, m(N)=0.5, m(Θ)=0.4
#
# Dempster's combination rule:
#   m12(A) = (1/(1-K)) * sum_{B∩C=A} m1(B)*m2(C)
#   K = sum_{B∩C=∅} m1(B)*m2(C)
#
# Combined sequentially: weight⊕crack, then ⊕tilt.
#
# Returns (Bel(D), Pl(D)) where Bel(D) = m({D}), Pl(D) = m({D}) + m(Θ).

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# Focal elements represented as frozensets
D = frozenset(["D"])
N = frozenset(["N"])
TH = frozenset(["D", "N"])  # Θ (uncertain)
EMPTY = frozenset()

BPA: Dict[str, Dict[bool, Dict[FrozenSet, float]]] = {
    "weight": {
        True:  {D: 0.5, N: 0.2, TH: 0.3},
        False: {D: 0.1, N: 0.6, TH: 0.3},
    },
    "crack": {
        True:  {D: 0.8, N: 0.05, TH: 0.15},
        False: {D: 0.05, N: 0.7, TH: 0.25},
    },
    "tilt": {
        True:  {D: 0.6, N: 0.1, TH: 0.3},
        False: {D: 0.1, N: 0.5, TH: 0.4},
    },
}


def _dempster_combine(
    m1: Dict[FrozenSet, float],
    m2: Dict[FrozenSet, float],
) -> Dict[FrozenSet, float]:
    """
    Dempster's rule of combination for two BPAs.
    Returns the combined BPA (normalised, conflict mass K removed).
    """
    # Compute K (conflict mass: mass on empty set before normalisation)
    K = 0.0
    unnorm: Dict[FrozenSet, float] = {}

    for A, mA in m1.items():
        for B, mB in m2.items():
            intersection = A & B
            if intersection == EMPTY:
                K += mA * mB
            else:
                unnorm[intersection] = unnorm.get(intersection, 0.0) + mA * mB

    if K >= 1.0:
        raise ValueError(
            "Complete conflict (K=1): Dempster combination undefined.")

    norm_factor = 1.0 / (1.0 - K)
    return {key: val * norm_factor for key, val in unnorm.items()}


def ds_belief_interval(evidence: dict) -> Tuple[float, float]:
    """
    Compute (Bel(D), Pl(D)) by combining three sensor BPAs.
    `evidence` is a raw sensor reading dict (§4.1 schema).
    Returns (Bel(D), Pl(D)) with Bel ≤ Pl guaranteed.
    """
    weight_high = evidence.get("weight_deviation_pct", 0) > 40
    crack_visible = evidence.get("crack_severity_pct", 0) > 15
    tilt = bool(evidence.get("tilt_triggered", False))

    m_weight = BPA["weight"][weight_high]
    m_crack = BPA["crack"][crack_visible]
    m_tilt = BPA["tilt"][tilt]

    # Combine sequentially: weight ⊕ crack, then ⊕ tilt
    m12 = _dempster_combine(m_weight, m_crack)
    m123 = _dempster_combine(m12, m_tilt)

    bel_D = m123.get(D,  0.0)
    m_th = m123.get(TH, 0.0)
    pl_D = bel_D + m_th   # Pl(D) = Bel(D) + mass on Θ

    assert bel_D <= pl_D, f"Bel > Pl invariant violated: {bel_D} > {pl_D}"
    return (bel_D, pl_D)
