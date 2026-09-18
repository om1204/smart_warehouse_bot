# part3_docking/board_state.py
# Dock Contention game state and helpers per §5.1.
#
# Board: 4×4 grid. Dock: (0,0). R1=MAX starts (3,3), R2=MIN starts (3,0).
# Moves: UP, DOWN, LEFT, RIGHT, STAY — filtered for bounds and collision.
# Terminal: robot reaches (0,0). Utility: R1→+10, R2→-10, cutoff→manhattan heuristic.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

DOCK = (0, 0)
# UP, DOWN, LEFT, RIGHT, STAY
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
MOVE_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


@dataclass(frozen=True)
class BoardState:
    r1_pos: Tuple[int, int]   # MAX
    r2_pos: Tuple[int, int]   # MIN
    turn: int                  # 1 = R1's turn (MAX), 2 = R2's turn (MIN)

    def is_r1_turn(self) -> bool:
        return self.turn == 1


def manhattan(pos: Tuple[int, int], target: Tuple[int, int] = DOCK) -> int:
    return abs(pos[0] - target[0]) + abs(pos[1] - target[1])


def legal_moves(state: BoardState) -> List[Tuple[int, int]]:
    """Return all legal moves (as (dr, dc) deltas) for the current player."""
    if state.turn == 1:
        cur, other = state.r1_pos, state.r2_pos
    else:
        cur, other = state.r2_pos, state.r1_pos

    result = []
    for dr, dc in MOVES:
        nr, nc = cur[0] + dr, cur[1] + dc
        if 0 <= nr <= 3 and 0 <= nc <= 3 and (nr, nc) != other:
            result.append((dr, dc))
    return result


def apply_move(state: BoardState, move: Tuple[int, int]) -> BoardState:
    """Apply move (dr, dc) for the current player; switch turns."""
    dr, dc = move
    if state.turn == 1:
        new_r1 = (state.r1_pos[0] + dr, state.r1_pos[1] + dc)
        return BoardState(r1_pos=new_r1, r2_pos=state.r2_pos, turn=2)
    else:
        new_r2 = (state.r2_pos[0] + dr, state.r2_pos[1] + dc)
        return BoardState(r1_pos=state.r1_pos, r2_pos=new_r2, turn=1)


def is_terminal(state: BoardState) -> bool:
    return state.r1_pos == DOCK or state.r2_pos == DOCK


def utility(state: BoardState, depth_remaining: int) -> Optional[float]:
    """
    Return utility if terminal, else heuristic if depth_remaining==0, else None.
    Utility: R1 at dock → +10, R2 at dock → -10.
    Heuristic: manhattan(R2, dock) - manhattan(R1, dock)  [favour R1 closeness]
    """
    if state.r1_pos == DOCK:
        return 10.0
    if state.r2_pos == DOCK:
        return -10.0
    if depth_remaining == 0:
        return float(manhattan(state.r2_pos) - manhattan(state.r1_pos))
    return None


START_STATE = BoardState(r1_pos=(3, 3), r2_pos=(3, 0), turn=1)
