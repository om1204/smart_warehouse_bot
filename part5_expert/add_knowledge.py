#!/usr/bin/env python
# part5_expert/add_knowledge.py
# Knowledge Acquisition CLI per §7.4.
#
# Prompts for rule fields and appends to rules.json.
# Does NOT modify any .py file.
# Also exposes append_rule() for programmatic use (TC-E4).

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

_DEFAULT_RULES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rules.json"
)


def append_rule(rule_dict: Dict[str, Any], rules_path: str = _DEFAULT_RULES_PATH) -> None:
    """
    Load-modify-write: append `rule_dict` to rules.json.
    Preserves all existing rules. Does not touch any .py file.
    """
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)

    # Check for duplicate id
    existing_ids = {r["id"] for r in rules}
    if rule_dict["id"] in existing_ids:
        print(
            f"Warning: rule id '{rule_dict['id']}' already exists — overwriting.")
        rules = [r for r in rules if r["id"] != rule_dict["id"]]

    rules.append(rule_dict)

    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=1, ensure_ascii=False)

    print(f"Rule '{rule_dict['id']}' appended to {rules_path}.")


def _prompt_rule() -> Optional[Dict[str, Any]]:
    """Interactive CLI prompts. Returns None if user cancels."""
    print("\n=== Knowledge Acquisition — Add a New Rule ===")
    try:
        rule_id = input("Rule ID (e.g. R19): ").strip()
        if not rule_id:
            return None

        if_raw = input(
            "IF facts (comma-separated, e.g. symptom:wheel_rpm_mismatch): ").strip()
        if_facts = [f.strip() for f in if_raw.split(",") if f.strip()]

        not_raw = input(
            "NOT facts (comma-separated, leave blank if none): ").strip()
        not_facts = [f.strip() for f in not_raw.split(",") if f.strip()]

        then_fact = input("THEN fact (e.g. diagnosis:some_fault): ").strip()
        if not then_fact:
            return None

        explanation = input("Explanation text: ").strip()

        return {
            "id":          rule_id,
            "if":          if_facts,
            "not":         not_facts,
            "then":        then_fact,
            "explanation": explanation,
        }
    except (EOFError, KeyboardInterrupt):
        return None


def main():
    rules_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_RULES_PATH
    rule = _prompt_rule()
    if rule is None:
        print("No rule entered — exiting.")
        return
    append_rule(rule, rules_path)


if __name__ == "__main__":
    main()
