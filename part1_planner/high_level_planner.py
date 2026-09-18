# part1_planner/high_level_planner.py
# Hierarchical Task Network (single-layer expansion) per §3.3.4.
#
# LoadPallet(pallet_id, order) expands into primitive STRIPS actions.
# `order` is a list of block ids from bottom to top of the desired stack.
#
# Expansion logic (TC-A4 exact spec):
#   For each block b in order (index i):
#     target_support = order[i-1] if i > 0 else "PALLET"
#     If b is already on target_support AND the arm is empty → skip (already placed).
#     Otherwise:
#       1. If b is not Clear, UNSTACK everything above it until it is Clear.
#       2. If b is on some other block → UNSTACK(b, support)
#          If b is on PALLET → PICKUP(b)
#       3. STACK(b, target_support) or PUTDOWN(b) if target_support == "PALLET"

from __future__ import annotations

from typing import List, Optional

from part1_planner.state import PalletState, execute_action


def _find_block_on_top_of(state: PalletState, target: str) -> Optional[str]:
    """Return the block sitting directly on `target`, or None."""
    for block, support in state.on.items():
        if support == target:
            return block
    return None


def _clear_block(state: PalletState, block: str) -> tuple[List[tuple], PalletState]:
    """
    Return a list of UNSTACK/PUTDOWN actions to make `block` clear,
    plus the resulting state.
    """
    actions = []
    current = state.copy()
    while not current.clear(block):
        # Find what is on top of block
        top = _find_block_on_top_of(current, block)
        if top is None:
            break
        action = ("UNSTACK", top, block)
        current = execute_action(current, action)
        actions.append(action)
        # Put it down on PALLET
        pd = ("PUTDOWN", top)
        current = execute_action(current, pd)
        actions.append(pd)
    return actions, current


def expand_load_pallet(
    state: PalletState, pallet_id: str, order: List[str]
) -> List[tuple]:
    """
    Expand the non-primitive LoadPallet(pallet_id, order) operator into a
    primitive STRIPS action sequence.

    `order` = [bottom_block, ..., top_block].
    For each block in order (index i):
      - target_support = order[i-1] if i > 0 else "PALLET"
      - If block is already on target_support and arm is empty → skip.
      - Else: clear it if needed, pick it up, stack on target_support.

    TC-A4 exact case:
      start = A,B,C all on PALLET; order = ["C","B","A"]
      C already on PALLET (target_support="PALLET") → skip C.
      B target_support="C"; B is on PALLET → PICKUP(B), STACK(B,C).
      A target_support="B"; A is on PALLET → PICKUP(A), STACK(A,B).
      Result: [PICKUP(B), STACK(B,C), PICKUP(A), STACK(A,B)] ✓
    """
    all_actions: List[tuple] = []
    current_state = state.copy()

    for i, block in enumerate(order):
        target_support = order[i - 1] if i > 0 else "PALLET"

        # Check if block is already correctly placed
        if (current_state.on.get(block) == target_support
                and current_state.arm_empty()):
            continue  # already in place — skip

        # Step 1: Make block Clear (remove anything on top of it)
        clear_actions, current_state = _clear_block(current_state, block)
        all_actions.extend(clear_actions)

        # Step 2: Pick up the block
        if current_state.on.get(block) == "PALLET":
            pick_action = ("PICKUP", block)
        else:
            current_support = current_state.on.get(block)
            pick_action = ("UNSTACK", block, current_support)
        current_state = execute_action(current_state, pick_action)
        all_actions.append(pick_action)

        # Step 3: Place block on target_support
        if target_support == "PALLET":
            place_action = ("PUTDOWN", block)
        else:
            place_action = ("STACK", block, target_support)
        current_state = execute_action(current_state, place_action)
        all_actions.append(place_action)

    return all_actions
