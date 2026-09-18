# part2_uncertainty/fuzzy_logic.py
# Fuzzy Logic damage scoring using scikit-fuzzy per §4.2.6.
#
# Universes:
#   weight_deviation: [0,100] — low=trimf(0,0,40), medium=trimf(20,50,80), high=trimf(60,100,100)
#   crack_severity:   [0,100] — none=trimf(0,0,20), minor=trimf(10,40,60), major=trimf(50,100,100)
#   damage_score:     [0,100] — low=trimf(0,0,40), medium=trimf(30,50,70), high=trimf(60,100,100)
#
# 5 rules (verbatim from §4.2.6):
#   R1: IF weight is high   AND crack is major  THEN damage is high
#   R2: IF crack is none    AND weight is low   THEN damage is low
#   R3: IF weight is medium                     THEN damage is medium
#   R4: IF crack is minor                       THEN damage is medium
#   R5: IF weight is high   OR  crack is major  THEN damage is high
#
# Defuzzification: centroid method.
# Returns crisp damage_score ∈ [0, 100].

from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------------------------------------------------------
# Build the fuzzy_logic control system once at module load
# ---------------------------------------------------------------------------


def _build_system():
    # Antecedents
    weight_dev = ctrl.Antecedent(np.arange(0, 101, 1), "weight_deviation")
    crack_sev = ctrl.Antecedent(np.arange(0, 101, 1), "crack_severity")

    # Consequent
    damage_score = ctrl.Consequent(np.arange(0, 101, 1), "damage_score",
                                   defuzzify_method="centroid")

    # Membership functions
    weight_dev["low"] = fuzz.trimf(weight_dev.universe, [0,   0,  40])
    weight_dev["medium"] = fuzz.trimf(weight_dev.universe, [20, 50,  80])
    weight_dev["high"] = fuzz.trimf(weight_dev.universe, [60, 100, 100])

    crack_sev["none"] = fuzz.trimf(crack_sev.universe, [0,   0,  20])
    crack_sev["minor"] = fuzz.trimf(crack_sev.universe, [10, 40,  60])
    crack_sev["major"] = fuzz.trimf(crack_sev.universe, [50, 100, 100])

    damage_score["low"] = fuzz.trimf(damage_score.universe, [0,   0,  40])
    damage_score["medium"] = fuzz.trimf(damage_score.universe, [30, 50,  70])
    damage_score["high"] = fuzz.trimf(damage_score.universe, [60, 100, 100])

    # Rules (exactly 5)
    r1 = ctrl.Rule(weight_dev["high"] &
                   crack_sev["major"],  damage_score["high"])
    r2 = ctrl.Rule(crack_sev["none"] &
                   weight_dev["low"],   damage_score["low"])
    r3 = ctrl.Rule(weight_dev["medium"],
                   damage_score["medium"])
    r4 = ctrl.Rule(crack_sev["minor"],
                   damage_score["medium"])
    r5 = ctrl.Rule(weight_dev["high"] |
                   crack_sev["major"],  damage_score["high"])

    system = ctrl.ControlSystem([r1, r2, r3, r4, r5])
    return system, ctrl.ControlSystemSimulation(system)


_CTRL_SYSTEM, _SIM = _build_system()


def fuzzy_logic_score(reading: dict) -> float:
    """
    Compute crisp damage_score ∈ [0, 100] for the given sensor reading.
    `reading` is a raw sensor reading dict (§4.1 schema).
    """
    w = float(reading.get("weight_deviation_pct", 0))
    c = float(reading.get("crack_severity_pct", 0))

    # Clamp to universe bounds
    w = max(0.0, min(100.0, w))
    c = max(0.0, min(100.0, c))

    # Re-create simulation each call to avoid state pollution
    sim = ctrl.ControlSystemSimulation(_CTRL_SYSTEM)
    sim.input["weight_deviation"] = w
    sim.input["crack_severity"] = c
    sim.compute()
    return float(sim.output["damage_score"])
