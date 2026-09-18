# part3_docking/minimax.py
# Plain recursive minimax with no pruning per §5.2.2.

from __future__ import annotations

from typing import Optional, Tuple

from part3_docking.board_state import (BoardState, apply_move, legal_moves,
                                      utility)


def minimax(
    state: BoardState,
    depth: int,
    maximizing: bool,
) -> Tuple[float, Optional[Tuple[int, int]], int]:
    """
    Plain minimax to `depth` plies.
    Returns (value, best_move, nodes_visited).
    nodes_visited increments once per node entered, BEFORE any check — comparable to alpha_beta_pruning.
    """
    nodes = [1]  # count this node

    u = utility(state, depth)
    if u is not None:
        return u, None, 1

    moves = legal_moves(state)
    if not moves:
        return utility(state, 0) or 0.0, None, 1

    best_move = moves[0]
    if maximizing:
        best_val = float("-inf")
        for move in moves:
            child = apply_move(state, move)
            val, _, child_nodes = minimax(child, depth - 1, False)
            nodes[0] += child_nodes
            if val > best_val:
                best_val = val
                best_move = move
        return best_val, best_move, nodes[0]
    else:
        best_val = float("inf")
        for move in moves:
            child = apply_move(state, move)
            val, _, child_nodes = minimax(child, depth - 1, True)
            nodes[0] += child_nodes
            if val < best_val:
                best_val = val
                best_move = move
        return best_val, best_move, nodes[0]
