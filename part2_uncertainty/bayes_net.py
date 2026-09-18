# part2_uncertainty/bayes_net.py
# Bayesian Network using pgmpy per §4.2.4.
#
# DAG edges: SensorNoise -> WeightReading, CrackVisible -> Damaged, Tilt -> Damaged
# CPDs are fixed per spec.
#
# NOTE: pgmpy renamed BayesianNetwork to DiscreteBayesianNetwork in v0.1.20.
# We try DiscreteBayesianNetwork first and fall back to BayesianNetwork.
# DEVIATION: using "try/except ImportError" fallback — flagged here as required.

from __future__ import annotations
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

import warnings

warnings.filterwarnings("ignore")

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetworkClass
except ImportError:
    from pgmpy.models import BayesianNetwork as BayesianNetworkClass  # type: ignore


# ---------------------------------------------------------------------------
# Build the model once at module load (immutable after construction)
# ---------------------------------------------------------------------------


def _build_model() -> "BayesianNetworkClass":
    model = BayesianNetworkClass([
        ("SensorNoise", "WeightReading"),
        ("CrackVisible", "Damaged"),
        ("Tilt",         "Damaged"),
    ])

    # P(SensorNoise): False=0.80, True=0.20
    cpd_sensor_noise = TabularCPD(
        variable="SensorNoise", variable_card=2,
        values=[[0.80], [0.20]],
        state_names={"SensorNoise": [False, True]},
    )

    # P(WeightReading | SensorNoise)
    # WeightReading: Normal=0, Abnormal=1
    # SensorNoise:   False=0,  True=1
    # P(Abnormal|NoNoise)=0.05, P(Abnormal|Noise)=0.60
    cpd_weight = TabularCPD(
        variable="WeightReading", variable_card=2,
        values=[
            [0.95, 0.40],   # P(Normal | SensorNoise=F, SensorNoise=T)
            [0.05, 0.60],   # P(Abnormal | ...)
        ],
        evidence=["SensorNoise"], evidence_card=[2],
        state_names={
            "WeightReading": ["Normal", "Abnormal"],
            "SensorNoise":   [False, True],
        },
    )

    # P(CrackVisible): False=0.85, True=0.15
    cpd_crack = TabularCPD(
        variable="CrackVisible", variable_card=2,
        values=[[0.85], [0.15]],
        state_names={"CrackVisible": [False, True]},
    )

    # P(Tilt): False=0.90, True=0.10
    cpd_tilt = TabularCPD(
        variable="Tilt", variable_card=2,
        values=[[0.90], [0.10]],
        state_names={"Tilt": [False, True]},
    )

    # P(Damaged | CrackVisible, Tilt)
    # Order of evidence columns: CrackVisible=F,Tilt=F | CrackVisible=F,Tilt=T |
    #                             CrackVisible=T,Tilt=F | CrackVisible=T,Tilt=T
    cpd_damaged = TabularCPD(
        variable="Damaged", variable_card=2,
        values=[
            [0.98, 0.50, 0.20, 0.05],  # P(NotDamaged | ...)
            [0.02, 0.50, 0.80, 0.95],  # P(Damaged    | ...)
        ],
        evidence=["CrackVisible", "Tilt"], evidence_card=[2, 2],
        state_names={
            "Damaged":      [False, True],
            "CrackVisible": [False, True],
            "Tilt":         [False, True],
        },
    )

    model.add_cpds(cpd_sensor_noise, cpd_weight,
                   cpd_crack, cpd_tilt, cpd_damaged)
    assert model.check_model(), "Bayesian Network CPDs are invalid!"
    return model


_MODEL = _build_model()
_INFERENCE = VariableElimination(_MODEL)


def bn_posterior(evidence: dict) -> float:
    """
    P(Damaged=True | evidence) via exact inference (VariableElimination).
    `evidence` is a raw sensor reading dict (§4.1 schema).
    Returns float in [0, 1].
    """
    crack_visible = evidence.get("crack_severity_pct", 0) > 15
    tilt = bool(evidence.get("tilt_triggered", False))

    bn_evidence = {
        "CrackVisible": crack_visible,
        "Tilt":         tilt,
    }

    result = _INFERENCE.query(
        ["Damaged"], evidence=bn_evidence, show_progress=False)
    # result is a DiscreteFactor; get P(Damaged=True)
    return float(result.values[1])  # index 1 = True
