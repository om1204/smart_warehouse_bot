# integration/control_loop.py
# Main control loop per §8.2.

import json
import os

from part1_planner.basic_stack_planner import generate_goal_stack_plan
from part1_planner.reactive_layer import execute_with_reactive_layer_layer
from part1_planner.state import PalletState
from part2_uncertainty import evaluate_all_methods
from part3_docking.dock_game import simulate_dock_game
from part5_expert.forward_chaining import diagnose


def run(shift_log: list[dict]) -> None:
    """
    Main control loop that dispatches events to respective modules.
    """

    print(f"Starting control loop with {len(shift_log)} events...\n")

    # Pre-load for module A
    data_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "data")
    start_path = os.path.join(data_dir, "start.json")
    goal_path = os.path.join(data_dir, "goal.json")

    with open(start_path, "r") as f:
        start_data = json.load(f)
        # Parse start state (assuming start.json has an 'on' dict)
        start_state = PalletState(on=start_data.get(
            "on", {}), holding=start_data.get("holding"))

    with open(goal_path, "r") as f:
        goal_data = json.load(f)

    # Initialize variables for disturbance batching
    # current_disturbances = []

    for i, event in enumerate(shift_log):
        event_type = event.get("type")
        print(f"--- Event {i+1}: {event_type} ---")

        if event_type == "sensor_reading":
            # Module B -> Module E
            data = event.get("data", {})
            eval_result = evaluate_all_methods(data)

            # Determine if damage is suspected based on multiple methods
            # (e.g., if Bayes > 0.5 or Nonmonotonic == 'Damaged')
            damage_suspected = (eval_result.get("bayes", 0.0) > 0.5) or (
                eval_result.get("default_logic") == "Damaged")
            print(
                f"Module B evaluation completed. Damage suspected: {damage_suspected}")

            if damage_suspected:
                # Dispatch to Module E
                print("Triggering Module E (Expert System) due to suspected damage...")
                # Extract boolean flags from data as symptoms
                symptoms = set()
                if damage_suspected:
                    symptoms.add("symptom:damage_suspected")
                for k, v in data.items():
                    if isinstance(v, bool) and v:
                        symptoms.add(f"symptom:{k}")

                diagnosis_result = diagnose(symptoms)
                print(f"Diagnoses found: {diagnosis_result['diagnoses']}")

        elif event_type == "disturbance":
            # Batch disturbances until we have a non-disturbance or end of log?
            # The spec says "disturbance->A reactive_layer".
            # We'll just execute module A reactive_layer layer on the fly with this disturbance.
            print(f"Module A handling disturbance: {event}")

            # Generate a baseline plan
            plan, _ = generate_goal_stack_plan(start_state, goal_data)

            # Execute with reactive_layer layer
            result = execute_with_reactive_layer_layer(plan, start_state, [event])

            if result["reactive_layer_log"]:
                print(f"Reactive actions taken: {len(result['reactive_layer_log'])}")
            else:
                print(
                    "No reactive_layer repair was needed or disturbance didn't break preconditions.")

        elif event_type == "dock_request":
            # Module C
            opponent = event.get("opponent", "random")
            print(f"Module C simulating dock game vs {opponent}...")
            res = simulate_dock_game(depth=3, opponent=opponent)
            print(
                f"Dock game finished in {res['plies']} plies. Winner: {res['winner']}")

        else:
            print(f"Unknown event type: {event_type}")

        print()

    print("Control loop finished.")
