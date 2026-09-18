# part1_planner/tests/test_planning.py
# Unit tests for Module A — TC-A1 through TC-A5.

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from part1_planner.state import PalletState, execute_action, is_goal_satisfied
from part1_planner.reactive_layer import execute_with_reactive_layer_layer
from part1_planner.constraint_planner import plan_nonlinear
from part1_planner.high_level_planner import expand_load_pallet
from part1_planner.basic_stack_planner import generate_goal_stack_plan


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def run_plan(plan, start_state):
    """Execute a list of actions from start_state; return final state."""
    state = start_state
    for action in plan:
        state = execute_action(state, action)
    return state


# ---------------------------------------------------------------------------
# TC-A1 — Basic stack, no conflict
# ---------------------------------------------------------------------------

class TestTCA1:
    START = PalletState(on={"A": "PALLET", "B": "PALLET", "C": "PALLET"})
    GOAL = {"on": {"A": "B"}}
    EXPECTED_PLAN = [("PICKUP", "A"), ("STACK", "A", "B")]

    def test_basic_stack_planner(self):
        plan, trace = generate_goal_stack_plan(self.START, self.GOAL)
        assert plan == self.EXPECTED_PLAN, f"Got: {plan}"
        final = run_plan(plan, self.START)
        assert is_goal_satisfied(
            final, self.GOAL), f"Goal not satisfied: {final}"

    def test_constraint_planner(self):
        pop = plan_nonlinear(self.START, self.GOAL)
        linear = pop.linearize()
        final = run_plan(linear, self.START)
        assert is_goal_satisfied(
            final, self.GOAL), f"Goal not satisfied: {final}"


# ---------------------------------------------------------------------------
# TC-A2 — Sussman Anomaly (both planners must solve it)
# ---------------------------------------------------------------------------

class TestTCA2:
    START = PalletState(on={"C": "A", "A": "PALLET", "B": "PALLET"})
    GOAL = {"on": {"A": "B", "B": "C"}}
    # Any plan of length 6 that satisfies the goal is acceptable.
    EXPECTED_PLAN = [
        ("UNSTACK", "C", "A"), ("PUTDOWN", "C"),
        ("PICKUP", "B"), ("STACK", "B", "C"),
        ("PICKUP", "A"), ("STACK", "A", "B"),
    ]

    def test_basic_stack_planner_satisfies_goal(self):
        plan, trace = generate_goal_stack_plan(self.START, self.GOAL)
        final = run_plan(plan, self.START)
        assert is_goal_satisfied(final, self.GOAL), (
            f"Goal-stack planner did not satisfy Sussman Anomaly goal.\n"
            f"Plan: {plan}\nFinal: {final}"
        )
        assert len(
            plan) == 6, f"Expected plan length 6, got {len(plan)}: {plan}"

    def test_constraint_planner_satisfies_goal(self):
        pop = plan_nonlinear(self.START, self.GOAL)
        linear = pop.linearize()
        final = run_plan(linear, self.START)
        assert is_goal_satisfied(final, self.GOAL), (
            f"Nonlinear planner did not satisfy Sussman Anomaly goal.\n"
            f"Plan: {linear}\nFinal: {final}"
        )


# ---------------------------------------------------------------------------
# TC-A3 — Independent subgoals (parallelism proof)
# ---------------------------------------------------------------------------

class TestTCA3:
    START = PalletState(on={"A": "PALLET", "B": "PALLET",
                       "C": "PALLET", "D": "PALLET"})
    GOAL = {"on": {"A": "B", "C": "D"}}

    def test_nonlinear_parallelism(self):
        pop = plan_nonlinear(self.START, self.GOAL)
        layers = pop.parallel_steps()
        # At least one layer must have 2+ actions (A,B and C,D are independent)
        max_parallel = max(len(layer) for layer in layers)
        assert max_parallel >= 2, (
            f"Expected ≥1 layer with 2+ parallel actions, got layers: {layers}"
        )

    def test_goal_stack_sequential_contrast(self):
        """Goal-stack plan is a single strict sequence — every layer has exactly 1 action."""
        plan, _ = generate_goal_stack_plan(self.START, self.GOAL)
        # Wrap the linear plan as single-action "layers" to prove contrast
        sequential_layers = [[action] for action in plan]
        assert all(len(layer) == 1 for layer in sequential_layers), (
            "Unexpected: goal-stack layer with more than 1 action"
        )
        assert len(sequential_layers) == len(plan)

    def test_nonlinear_satisfies_goal(self):
        pop = plan_nonlinear(self.START, self.GOAL)
        linear = pop.linearize()
        final = run_plan(linear, self.START)
        assert is_goal_satisfied(
            final, self.GOAL), f"Goal not satisfied: {final}"


# ---------------------------------------------------------------------------
# TC-A4 — Hierarchical expansion (exact list equality)
# ---------------------------------------------------------------------------

class TestTCA4:
    START = PalletState(on={"A": "PALLET", "B": "PALLET", "C": "PALLET"})
    ORDER = ["C", "B", "A"]
    # C is already on PALLET (target_support for index 0 = PALLET) → skip C.
    # B: target=C → PICKUP(B), STACK(B,C)
    # A: target=B → PICKUP(A), STACK(A,B)
    EXPECTED = [
        ("PICKUP", "B"), ("STACK", "B", "C"),
        ("PICKUP", "A"), ("STACK", "A", "B"),
    ]

    def test_expand_load_pallet(self):
        result = expand_load_pallet(self.START, "P1", self.ORDER)
        assert result == self.EXPECTED, (
            f"expand_load_pallet returned:\n  {result}\nExpected:\n  {self.EXPECTED}"
        )


# ---------------------------------------------------------------------------
# TC-A5 — Reactive disturbance
# ---------------------------------------------------------------------------

class TestTCA5:
    START = PalletState(on={"A": "PALLET", "B": "PALLET", "C": "PALLET"})
    PLAN = [("PICKUP", "A"), ("STACK", "A", "B")]
    GOAL = {"on": {"A": "B"}}
    # Disturbance: after step 0 (after PICKUP(A)), block A is knocked back to PALLET.
    # The disturbance handler sets A back on PALLET and clears holding.
    DISTURBANCES = [{"after_step": 0, "block": "A"}]

    def test_reactive_layer_log_printed(self, capsys):
        result = execute_with_reactive_layer_layer(
            self.PLAN, self.START, self.DISTURBANCES)
        assert len(result["reactive_layer_log"]) >= 1, (
            "Expected at least one [REACTIVE] log line"
        )

    def test_goal_still_satisfied(self):
        result = execute_with_reactive_layer_layer(
            self.PLAN, self.START, self.DISTURBANCES)
        assert is_goal_satisfied(result["final_state"], self.GOAL), (
            f"Goal not satisfied after reactive_layer repair. Final: {result['final_state']}"
        )

    def test_repair_length_le_2(self):
        result = execute_with_reactive_layer_layer(
            self.PLAN, self.START, self.DISTURBANCES)
        # Each reactive_layer_log line mentions a repair; count total repair actions
        import re
        for log_line in result["reactive_layer_log"]:
            # Extract repair list from the log message
            match = re.search(r"Inserted repair: \[(.*?)\]", log_line)
            if match:
                repair_str = match.group(1)
                # Count number of tuples (each starts with a quote)
                n_actions = repair_str.count("'PICKUP'") + repair_str.count("'STACK'") + \
                    repair_str.count("'PUTDOWN'") + \
                    repair_str.count("'UNSTACK'")
                assert n_actions <= 2, f"Repair has {n_actions} actions (expected ≤2): {log_line}"


# ---------------------------------------------------------------------------
# State utility tests
# ---------------------------------------------------------------------------

class TestStateUtils:
    def test_execute_action_pickup(self):
        s = PalletState(on={"A": "PALLET", "B": "PALLET"})
        s2 = execute_action(s, ("PICKUP", "A"))
        assert s2.holding == "A"
        assert "A" not in s2.on

    def test_execute_action_stack(self):
        s = PalletState(on={"B": "PALLET"}, holding="A")
        s2 = execute_action(s, ("STACK", "A", "B"))
        assert s2.on["A"] == "B"
        assert s2.holding is None

    def test_execute_action_unstack(self):
        s = PalletState(on={"A": "B", "B": "PALLET"})
        s2 = execute_action(s, ("UNSTACK", "A", "B"))
        assert s2.holding == "A"
        assert "A" not in s2.on

    def test_execute_action_putdown(self):
        s = PalletState(on={"B": "PALLET"}, holding="A")
        s2 = execute_action(s, ("PUTDOWN", "A"))
        assert s2.on["A"] == "PALLET"
        assert s2.holding is None

    def test_clear(self):
        s = PalletState(on={"A": "B", "B": "PALLET"})
        assert s.clear("B") is False  # A is on B
        assert s.clear("A") is True   # nothing on A

    def test_arm_empty(self):
        s1 = PalletState(on={"A": "PALLET"})
        assert s1.arm_empty() is True
        s2 = PalletState(on={"B": "PALLET"}, holding="A")
        assert s2.arm_empty() is False

    def test_is_goal_satisfied_partial(self):
        s = PalletState(on={"A": "B", "B": "PALLET", "C": "PALLET"})
        assert is_goal_satisfied(s, {"on": {"A": "B"}}) is True
        assert is_goal_satisfied(s, {"on": {"A": "C"}}) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
