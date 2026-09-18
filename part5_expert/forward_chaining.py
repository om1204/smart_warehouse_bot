# part5_expert/forward_chaining.py
# Hand-rolled forward-chaining rule engine per §7.1.
# No experta, no CLIPS — pure Python.
#
# Rules are loaded from rules.json on EVERY call to diagnose() — no hardcoded
# rules in Python. This allows add_knowledge.py to add rules without
# changing any .py file.
#
# Algorithm:
#   1. Load facts into working memory (set[str]).
#   2. Loop until no new fact added in a full pass, or max_passes=50 reached:
#        For each rule not yet fired:
#          if all(if_fact in working_memory) and not any(not_fact in working_memory):
#            assert then_fact, add rule_id to fired_rule_ids, append to trace.
#   3. Return diagnoses (facts with prefix "diagnosis:") and the trace.

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set, Tuple

# Path to rules.json relative to this file
_RULES_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "rules.json")


def _load_rules(rules_path: str = _RULES_PATH) -> List[Dict]:
    """Load rule list from rules.json fresh every call."""
    with open(rules_path, "r", encoding="utf-8") as f:
        return json.load(f)


def diagnose(
    symptom_facts: Set[str],
    rules_path: str = _RULES_PATH,
    max_passes: int = 50,
) -> Dict[str, Any]:
    """
    Run the forward-chaining engine on `symptom_facts`.

    Returns {
      "diagnoses":   list[str],                    # facts with prefix "diagnosis:"
      "trace":       list[tuple[str, str, str]],   # (rule_id, explanation, then_fact)
      "fired_rule_ids": list[str],
      "working_memory": set[str],
    }
    """
    rules = _load_rules(rules_path)
    working_memory: Set[str] = set(symptom_facts)
    fired_rule_ids: List[str] = []
    trace: List[Tuple[str, str, str]] = []

    for _ in range(max_passes):
        new_fact_added = False
        for rule in rules:
            rid = rule["id"]
            if rid in fired_rule_ids:
                continue
            # Check if-conditions
            if not all(f in working_memory for f in rule["if"]):
                continue
            # Check not-conditions
            if any(f in working_memory for f in rule.get("not", [])):
                continue
            # Fire the rule
            then_fact = rule["then"]
            working_memory.add(then_fact)
            fired_rule_ids.append(rid)
            trace.append((rid, rule["explanation"], then_fact))
            if then_fact not in symptom_facts:
                new_fact_added = True
        if not new_fact_added:
            break

    diagnoses = sorted(f for f in working_memory if f.startswith("diagnosis:"))
    return {
        "diagnoses":       diagnoses,
        "trace":           trace,
        "fired_rule_ids":  fired_rule_ids,
        "working_memory":  working_memory,
    }
