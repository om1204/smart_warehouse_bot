# part1_planner/reactive_layer.py
# Reactive execution layer per §3.3.5.
#
# TC-A5 trace:
#   Plan = [PICKUP(A), STACK(A,B)], dist = {"after_step": 0, "block": "A"}
#   Step 1: Execute PICKUP(A) → holding='A'. original_steps_done=1.
#            after_step=0 means "after the action with index 0", i.e. after PICKUP(A).
#            Apply disturbance: A released back to PALLET. holding=None.
#   Step 2: Next action STACK(A,B) requires Holding(A) — violated.
#            active_disturbance set → compute repair → PICKUP(A).
#            reactive_layer_log populated.
#   Step 3: Execute repair PICKUP(A) → holding='A'.
#   Step 4: Execute STACK(A,B) → goal met.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from part1_planner.state import (PalletState, _verify_conditions,
                                     _evaluate_predicate, _preconditions, execute_action)


def _apply_disturbance(state: PalletState, disturbance: dict) -> PalletState:
    block = disturbance["block"]
    new_state = state.copy()

    if new_state.holding == block:
        new_state.holding = None
        new_state.on[block] = "PALLET"

    def above(s: PalletState, target: str) -> List[str]:
        return [b for b, sup in s.on.items() if sup == target]

    fallen = []
    queue = [block]
    while queue:
        cur = queue.pop(0)
        for b in above(new_state, cur):
            fallen.append(b)
            queue.append(b)

    if block in new_state.on:
        new_state.on[block] = "PALLET"
    for b in fallen:
        if b in new_state.on:
            new_state.on[b] = "PALLET"

    return new_state


def _compute_repair(state: PalletState, violated_precond: str) -> List[tuple]:
    repair: List[tuple] = []

    if violated_precond.startswith("Holding("):
        x = violated_precond[8:-1]
        if state.arm_empty():
            if state.on.get(x) == "PALLET" and state.clear(x):
                repair.append(("PICKUP", x))
            elif x in state.on and state.on[x] != "PALLET" and state.clear(x):
                repair.append(("UNSTACK", x, state.on[x]))

    elif violated_precond == "ArmEmpty":
        if state.holding:
            repair.append(("PUTDOWN", state.holding))

    elif violated_precond.startswith("Clear("):
        y = violated_precond[6:-1]
        top_blocks = [b for b, sup in state.on.items() if sup == y]
        if top_blocks:
            top = top_blocks[0]
            repair.append(("UNSTACK", top, y))
            repair.append(("PUTDOWN", top))

    elif violated_precond.startswith("On("):
        inner = violated_precond[3:-1]
        x, y = inner.split(",", 1)
        if state.arm_empty() and state.clear(x):
            if state.on.get(x) == "PALLET":
                repair.append(("PICKUP", x))
            elif x in state.on:
                repair.append(("UNSTACK", x, state.on[x]))
            if y == "PALLET":
                repair.append(("PUTDOWN", x))
            else:
                repair.append(("STACK", x, y))

    return repair


def execute_with_reactive_layer_layer(
    plan: List[tuple],
    state: PalletState,
    disturbance_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute `plan` from `state`.

    Disturbance {"after_step": N, "block": B}:
      Applied AFTER the action at 0-based index N in the ORIGINAL plan finishes.

    Returns {"final_state": PalletState, "trace": list[str], "reactive_layer_log": list[str]}
    """
    # dist_map[N] = list of disturbances to apply after original plan action N
    dist_map: Dict[int, List[Dict]] = {}
    for d in disturbance_events:
        dist_map.setdefault(d["after_step"], []).append(d)

    current_state = state.copy()
    trace: List[str] = []
    reactive_layer_log: List[str] = []

    # We maintain a queue of actions; original_steps_done tracks how many
    # actions from the ORIGINAL plan have been executed.
    queue: List[tuple] = list(plan)
    original_steps_done = 0   # incremented only for original plan actions
    active_disturbance: Optional[Dict] = None
    # Set of actions that are repair insertions (not original plan actions)
    repair_set: set = set()

    while queue:
        action = queue[0]
        is_repair_action = id(action) in repair_set if repair_set else False

        # --- Check preconditions ---
        if not _verify_conditions(current_state, action):
            if active_disturbance is not None:
                precs = _preconditions(action)
                violated = [
                    p for p in precs if not _evaluate_predicate(current_state, p)]
                repaired = False
                for vp in violated:
                    repair = _compute_repair(current_state, vp)
                    if repair:
                        block_name = active_disturbance["block"]
                        log_msg = (
                            f"[REACTIVE] step {original_steps_done}: precondition '{vp}' violated by "
                            f"disturbance on block '{block_name}'. "
                            f"Inserted repair: {repair}"
                        )
                        reactive_layer_log.append(log_msg)
                        trace.append(f"  {log_msg}")
                        print(log_msg)
                        # Use unique action objects so we can track them as repair
                        queue = repair + queue
                        active_disturbance = None
                        repaired = True
                        break
                if not repaired:
                    trace.append(
                        f"  ERROR: no repair for violated prec(s) {violated}")
                    break
                continue
            else:
                trace.append(f"  ERROR: preconditions not met for {action}")
                break

        # --- Execute action ---
        queue.pop(0)
        current_state = execute_action(current_state, action)
        trace.append(f"  Executed {action} → {current_state}")

        # Advance original plan counter and check for post-action disturbances
        if not is_repair_action and original_steps_done < len(plan) and action == plan[original_steps_done]:
            original_steps_done += 1
            # Apply any disturbances that fire after this original plan step
            step_index = original_steps_done - 1  # 0-based index of the action just done
            if step_index in dist_map:
                for d in dist_map.pop(step_index):
                    current_state = _apply_disturbance(current_state, d)
                    active_disturbance = d
                    trace.append(
                        f"  [DISTURBANCE] after original step {step_index}: "
                        f"block '{d['block']}' knocked to PALLET → {current_state}"
                    )

    return {
        "final_state": current_state,
        "trace": trace,
        "reactive_layer_log": reactive_layer_log,
    }
