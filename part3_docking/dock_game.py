#!/usr/bin/env python
# part3_docking/dock_game.py
# CLI game runner per §5.2.6 + Public API simulate_dock_game() per §5.5.
#
# Usage: python dock_game.py --depth 5 [--opponent random|self]

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from part3_docking.ids_search import ids_search
from part3_docking.board_state import (DOCK, MOVE_NAMES, MOVES, START_STATE,
                                      BoardState, apply_move, is_terminal,
                                      legal_moves)
from part3_docking.alpha_beta_pruning import alpha_beta_pruning


def _move_name(move):
    for i, m in enumerate(MOVES):
        if m == move:
            return MOVE_NAMES[i]
    return str(move)


def _board_str(state: BoardState) -> str:
    rows = []
    for r in range(4):
        row = []
        for c in range(4):
            if (r, c) == DOCK:
                cell = "D"
            elif (r, c) == state.r1_pos:
                cell = "1"
            elif (r, c) == state.r2_pos:
                cell = "2"
            else:
                cell = "."
            row.append(cell)
        rows.append(" ".join(row))
    return "\n".join(rows)


def simulate_dock_game(depth: int = 5, opponent: str = "random") -> dict:
    """
    Public API per §5.5.
    Returns {"winner": "R1"|"R2"|"none", "plies": int, "trace": list[str]}
    """
    state = START_STATE
    trace = []
    plies = 0
    max_plies = 200  # safety limit

    while not is_terminal(state) and plies < max_plies:
        turn = state.turn  # 1 = R1 (MAX), 2 = R2 (MIN)
        moves = legal_moves(state)
        if not moves:
            break

        if turn == 1:
            # R1 uses iterative deepening alpha-beta
            _, best_move, _ = ids_search(state, depth)
            if best_move is None:
                best_move = moves[0]
        else:
            if opponent == "random":
                best_move = random.choice(moves)
            else:  # self: R2 also uses alpha-beta as MIN
                _, best_move, _ = alpha_beta_pruning(
                    state, depth, float("-inf"), float("inf"), False)
                if best_move is None:
                    best_move = random.choice(moves)

        board = _board_str(state)
        robot = "R1" if turn == 1 else "R2"
        move_name = _move_name(best_move)
        ply_log = f"Ply {plies+1}: {robot} moves {move_name}\n{board}"
        trace.append(ply_log)

        state = apply_move(state, best_move)
        plies += 1

    if state.r1_pos == DOCK:
        winner = "R1"
    elif state.r2_pos == DOCK:
        winner = "R2"
    else:
        winner = "none"

    return {"winner": winner, "plies": plies, "trace": trace}


def main():
    parser = argparse.ArgumentParser(description="Dock Contention Game")
    parser.add_argument("--depth",    type=int, default=5,
                        help="Search depth")
    parser.add_argument("--opponent", type=str,
                        default="random", choices=["random", "self"])
    args = parser.parse_args()

    print(f"Dock Contention — depth={args.depth}, opponent={args.opponent}")
    print("=" * 40)

    result = simulate_dock_game(args.depth, args.opponent)

    for entry in result["trace"]:
        print(entry)
        print()

    print(
        f"Game over after {result['plies']} plies. Winner: {result['winner']}")


if __name__ == "__main__":
    main()
