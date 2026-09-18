# part3_docking/ids_search.py
# Iterative Deepening Alpha-Beta (IDA*-style) per §5.2.4.
#
# At each depth, moves are ordered:
#   1. Previous depth's best move first.
#   2. Remaining moves sorted by descending own_distance_reduction
#      = manhattan(current_pos, dock) - manhattan(new_pos, dock).

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from part3_docking.alpha_beta_pruning import alpha_beta_pruning
from part3_docking.board_state import (BoardState, apply_move, legal_moves,
                                      manhattan)


def _order_moves(
    state: BoardState,
    prev_best: Optional[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Order moves: prev_best first, rest by descending distance-reduction heuristic."""
    moves = legal_moves(state)
    if not moves:
        return []

    cur_pos = state.r1_pos if state.turn == 1 else state.r2_pos

    def heuristic(move):
        new_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
        return manhattan(cur_pos) - manhattan(new_pos)

    # Sort all moves by descending heuristic
    rest = sorted(moves, key=heuristic, reverse=True)

    # Move prev_best to front if it's in the move list
    if prev_best is not None and prev_best in rest:
        rest.remove(prev_best)
        rest = [prev_best] + rest

    return rest


def ids_search(
    state: BoardState,
    max_depth: int,
) -> Tuple[float, Optional[Tuple[int, int]], List[Dict[str, Any]]]:
    """
    Iterative deepening alpha-beta search.
    Returns (value, best_move, per_depth_stats).

    per_depth_stats: list of {"depth": d, "nodes_visited": n, "time_seconds": t, "best_move": m}
    """
    per_depth_stats: List[Dict[str, Any]] = []
    prev_best: Optional[Tuple[int, int]] = None
    final_value = 0.0
    final_move = None

    for depth in range(1, max_depth + 1):
        t0 = time.perf_counter()

        # Get ordered moves at root level
        ordered_moves = _order_moves(state, prev_best)
        if not ordered_moves:
            break

        total_nodes = 1  # count root
        best_val = float("-inf")
        best_move = ordered_moves[0]

        for move in ordered_moves:
            child = apply_move(state, move)
            val, _, child_nodes = alpha_beta_pruning(
                child, depth - 1, float("-inf"), float("inf"), False
            )
            total_nodes += child_nodes
            if val > best_val:
                best_val = val
                best_move = move

        elapsed = time.perf_counter() - t0
        prev_best = best_move
        final_value = best_val
        final_move = best_move

        per_depth_stats.append({
            "depth":        depth,
            "nodes_visited": total_nodes,
            "time_seconds": elapsed,
            "best_move":    best_move,
        })

    return final_value, final_move, per_depth_stats
