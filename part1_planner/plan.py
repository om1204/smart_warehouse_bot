#!/usr/bin/env python
# part1_planner/plan.py
# CLI entrypoint per §3.5:  python plan.py start.json goal.json
#
# Prints: the goal-stack plan, step-by-step execution trace,
#         the nonlinear plan layers, and any [REACTIVE] log lines.

import json
import os
import sys

# Ensure standard output can handle Unicode characters on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure repo root is on sys.path regardless of where the script is invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from part1_planner.state import PalletState, is_goal_satisfied
from part1_planner.reactive_layer import execute_with_reactive_layer_layer
from part1_planner.constraint_planner import plan_nonlinear
from part1_planner.basic_stack_planner import generate_goal_stack_plan


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python plan.py <start.json> <goal.json> [disturbances.json]")
        sys.exit(1)

    start_path = sys.argv[1]
    goal_path = sys.argv[2]
    dist_path = sys.argv[3] if len(sys.argv) > 3 else None

    with open(start_path) as f:
        start_data = json.load(f)
    with open(goal_path) as f:
        goal_data = json.load(f)

    start_state = PalletState(**start_data)
    disturbances = []
    if dist_path:
        with open(dist_path) as f:
            disturbances = json.load(f)

    print("=" * 60)
    print("START STATE:", start_state)
    print("GOAL       :", goal_data)
    print("=" * 60)

    # --- Goal Stack Plan ---
    print("\n[GOAL-STACK PLANNER]")
    plan, trace = generate_goal_stack_plan(start_state, goal_data)
    print("Plan:", plan)
    print("\nTrace:")
    for line in trace:
        print(" ", line)

    # --- Nonlinear Plan ---
    print("\n[NONLINEAR (POP) PLANNER]")
    pop_plan = plan_nonlinear(start_state, goal_data)
    linear_seq = pop_plan.linearize()
    layers = pop_plan.parallel_steps()
    print("Linearized plan:", linear_seq)
    print("Parallel layers:")
    for i, layer in enumerate(layers):
        print(f"  Layer {i+1}: {layer}")

    # --- Reactive Execution ---
    print("\n[REACTIVE EXECUTION]")
    result = execute_with_reactive_layer_layer(plan, start_state, disturbances)
    print("Execution trace:")
    for line in result["trace"]:
        print(line)
    if result["reactive_layer_log"]:
        print("\nReactive log:")
        for line in result["reactive_layer_log"]:
            print(line)
    final = result["final_state"]
    satisfied = is_goal_satisfied(final, goal_data)
    print(f"\nFinal state: {final}")
    print(f"Goal satisfied: {satisfied}")


if __name__ == "__main__":
    main()
