#!/usr/bin/env python
# part5_expert/expert_advisor.py
# CLI for the Expert System Maintenance Advisor per §7.6.
#
# Usage:
#   python expert_advisor.py [symptom:fact1 symptom:fact2 ...]
#   (interactive prompt if no args given)

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from part5_expert.forward_chaining import diagnose
from part5_expert.explanation import format_trace


def _interactive_facts():
    print("Enter symptom facts one per line (blank line to finish):")
    facts = set()
    while True:
        try:
            line = input("  > ").strip()
        except EOFError:
            break
        if not line:
            break
        facts.add(line)
    return facts


def main():
    if len(sys.argv) > 1:
        facts = set(sys.argv[1:])
    else:
        facts = _interactive_facts()

    print(f"\nInput facts: {sorted(facts)}")
    result = diagnose(facts)

    print("\n=== Rule Firing Trace ===")
    if result["trace"]:
        print(format_trace(result["trace"]))
    else:
        print("(no rules fired)")

    print("\n=== Diagnoses ===")
    if result["diagnoses"]:
        for d in result["diagnoses"]:
            print(f"  {d}")
    else:
        print("  (no diagnoses reached)")


if __name__ == "__main__":
    main()
