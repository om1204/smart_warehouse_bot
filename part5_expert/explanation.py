# part5_expert/explanation.py
# Explanation facility per §7.3.
#
# format_trace() prints the firing trace in exact firing order
# (causal chains like R4→R5 are visible as consecutive entries).

from __future__ import annotations

from typing import List, Tuple


def format_trace(trace: List[Tuple[str, str, str]]) -> str:
    """
    Format a rule-firing trace as a human-readable string.
    Each entry is (rule_id, explanation, then_fact).
    Output order matches the exact order rules fired — preserving causal chains.
    """
    lines = []
    for rule_id, explanation, then_fact in trace:
        lines.append(
            f"Rule {rule_id} fired: {explanation} => asserted '{then_fact}'")
    return "\n".join(lines)


def print_trace(trace: List[Tuple[str, str, str]]) -> None:
    """Print the formatted trace to stdout."""
    print(format_trace(trace))
