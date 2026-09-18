# part3_docking/tests/test_game.py
# Unit tests for Module C (Game) - TC-C1 to TC-C4

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from part3_docking.minimax import minimax
from part3_docking.ids_search import ids_search
from part3_docking.board_state import START_STATE
from part3_docking.dock_game import simulate_dock_game
from part3_docking.alpha_beta_pruning import alpha_beta_pruning


# ---------------------------------------------------------------------------
# TC-C1 — Minimax vs AlphaBeta node counts
# ---------------------------------------------------------------------------
class TestTCC1:
    def test_alpha_beta_pruning_prunes(self):
        """Alpha-beta should visit fewer (or equal) nodes than minimax at depth 4."""
        depth = 4
        _, _, mm_nodes = minimax(START_STATE, depth, True)
        _, _, ab_nodes = alpha_beta_pruning(
            START_STATE, depth, float("-inf"), float("inf"), True)
        assert ab_nodes <= mm_nodes, (
            f"Alpha-beta ({ab_nodes}) visited more nodes than minimax ({mm_nodes})!"
        )

# ---------------------------------------------------------------------------
# TC-C2 — Minimax vs AlphaBeta values
# ---------------------------------------------------------------------------


class TestTCC2:
    def test_values_match(self):
        """Alpha-beta and minimax should return the exact same value for the same state and depth."""
        depth = 3
        mm_val, mm_move, _ = minimax(START_STATE, depth, True)
        ab_val, ab_move, _ = alpha_beta_pruning(
            START_STATE, depth, float("-inf"), float("inf"), True)
        assert mm_val == ab_val, f"Values differ: MM={mm_val}, AB={ab_val}"

# ---------------------------------------------------------------------------
# TC-C3 — Iterative Deepening
# ---------------------------------------------------------------------------


class TestTCC3:
    def test_ids_runs(self):
        """IDS should return a valid move and value at max depth."""
        depth = 3
        val, move, stats = ids_search(START_STATE, depth)
        assert move is not None, "IDS did not return a valid move"
        assert len(stats) == depth, "IDS did not return stats for each depth"
        assert stats[-1]["depth"] == depth, "Final depth stat mismatch"

# ---------------------------------------------------------------------------
# TC-C4 — Game Simulation
# ---------------------------------------------------------------------------


class TestTCC4:
    def test_simulate_game(self):
        """Game simulation should complete without errors and have a trace."""
        res = simulate_dock_game(depth=2, opponent="random")
        assert "winner" in res
        assert "plies" in res
        assert "trace" in res
        assert res["plies"] > 0
        assert len(res["trace"]) == res["plies"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
