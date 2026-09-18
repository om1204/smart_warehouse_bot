# part2_uncertainty/default_logic.py
# Nonmonotonic vs. Monotonic reasoning demonstration per §4.2.1.
#
# Default logic: Damaged(x) is FALSE unless ab(x) is derivable.
# ab(x) is derivable if "tilt_triggered" or "crack_visible" is in the current fact set.
#
# default_logic_verdict() recomputes from scratch every call — facts can be added
# or retracted freely, and the verdict changes accordingly (this is the core
# default_logicity property: conclusions ARE retractable).
#
# monotonic_verdict() uses an append-only Horn-clause KB — once a conclusion is
# added it is NEVER removed, even if the supporting fact is retracted. This
# demonstrates the contrast with classical (monotonic) logic.

from __future__ import annotations

from typing import List, Set


def default_logic_verdict(facts: Set[str]) -> str:
    """
    Default-logic verdict. Returns "Damaged" if any abnormality fact
    ("tilt_triggered" or "crack_visible") is in `facts`, else "NotDamaged".

    Re-computed from scratch on every call — no state between calls.
    Adding or removing facts WILL change the verdict (default_logicity).
    """
    if "tilt_triggered" in facts or "crack_visible" in facts:
        return "Damaged"
    return "NotDamaged"


def monotonic_verdict(kb: List[str]) -> str:
    """
    Monotonic Horn-clause KB verdict. The KB is an append-only list.
    Once "Damaged" is derived and added to the KB it is NEVER removed,
    even if the fact that caused it is no longer in the working set.

    Simulates classical forward-chaining where conclusions are permanent.
    """
    # Rule: Damaged :- crack_detected.
    # If "crack_detected" ever appears in kb, we derive "Damaged" and it stays.
    derived = set(kb)
    if "crack_detected" in derived:
        derived.add("Damaged")
    return "Damaged" if "Damaged" in derived else "NotDamaged"


def sensor_reading_to_facts(reading: dict) -> Set[str]:
    """Convert a raw sensor reading dict to a set of boolean fact strings."""
    facts: Set[str] = set()
    w = reading.get("weight_deviation_pct", 0)
    c = reading.get("crack_severity_pct", 0)
    t = reading.get("tilt_triggered", False)

    if w > 40:
        facts.add("weight_deviation_high")
    if c > 15:
        facts.add("crack_visible")
    if t:
        facts.add("tilt_triggered")
    return facts
