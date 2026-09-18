# part3_docking/alpha_beta_pruning.py
# Alpha-beta pruning per §5.2.3.
#
# nodes_visited is incremented once per node entered (BEFORE the pruning check)
# so counts are directly comparable to plain minimax.

from __future__ import annotations

from typing import Optional, Tuple

from part3_docking.board_state import (BoardState, apply_move, legal_moves,
                                      utility)


def alpha_beta_pruning(
    state: BoardState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
) -> Tuple[float, Optional[Tuple[int, int]], int]:
    """
    Alpha-beta pruning minimax.
    Returns (value, best_move, nodes_visited).
    """
    nodes = 1  # count this node before any pruning

    u = utility(state, depth)
    if u is not None:
        return u, None, nodes

    moves = legal_moves(state)
    if not moves:
        return utility(state, 0) or 0.0, None, nodes

    best_move = moves[0]
    if maximizing:
        best_val = float("-inf")
        for move in moves:
            child = apply_move(state, move)
            val, _, child_nodes = alpha_beta_pruning(
                child, depth - 1, alpha, beta, False)
            nodes += child_nodes
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break  # β cutoff
        return best_val, best_move, nodes
    else:
        best_val = float("inf")
        for move in moves:
            child = apply_move(state, move)
            val, _, child_nodes = alpha_beta_pruning(
                child, depth - 1, alpha, beta, True)
            nodes += child_nodes
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, best_val)
            if beta <= alpha:
                break  # α cutoff
        return best_val, best_move, nodes
