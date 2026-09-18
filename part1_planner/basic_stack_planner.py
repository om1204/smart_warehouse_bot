# part1_planner/basic_stack_planner.py
# Goal Stack Planning per §3.3.2.
#
# This "extended" implementation handles the Sussman Anomaly (TC-A2) by using
# the standard trick documented in Nilsson (1980) and Russell & Norvig:
# process goal subsets in a safe order — first clear all blocks that need to be
# repositioned (without placing them permanently yet), then build the tower from
# the bottom up.
#
# The algorithm:
# 1. Determine the blocks that need to be placed (from goal).
# 2. Find any block currently sitting on top of a goal-block that is NOT itself
#    a goal-block at that position — "obstacles". Move them to PALLET first.
# 3. Build the goal tower bottom-up.
#
# This produces the canonical 6-step plan for TC-A2:
#   [UNSTACK(C,A), PUTDOWN(C), PICKUP(B), STACK(B,C), PICKUP(A), STACK(A,B)]

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from part1_planner.state import PalletState, execute_action


def _all_blocks(state: PalletState, goal: dict) -> List[str]:
    blocks = set(state.on.keys())
    if state.holding:
        blocks.add(state.holding)
    blocks.update(v for v in state.on.values() if v != "PALLET")
    blocks.update(goal.get("on", {}).keys())
    blocks.update(v for v in goal.get("on", {}).values() if v != "PALLET")
    return list(blocks)


def _find_top(state: PalletState, block: str) -> Optional[str]:
    """Find the block sitting directly on top of `block`, or None."""
    for b, s in state.on.items():
        if s == block:
            return b
    return None


def _unstack_to_pallet(state: PalletState, block: str) -> Tuple[List[tuple], PalletState]:
    """
    Remove `block` from wherever it is (and everything on top of it)
    and place everything on PALLET. Returns (actions, new_state).
    """
    actions = []
    # First, recursively clear everything on top of block
    while True:
        top = _find_top(state, block)
        if top is None:
            break
        sub, state = _unstack_to_pallet(state, top)
        actions.extend(sub)

    # Now block is clear; arm should be empty at this point
    if not state.arm_empty():
        pd = ("PUTDOWN", state.holding)
        state = execute_action(state, pd)
        actions.append(pd)

    # Unstack or pickup block
    if state.on.get(block) == "PALLET":
        return actions, state  # already on pallet
    sup = state.on.get(block)
    if sup and sup != "PALLET":
        a = ("UNSTACK", block, sup)
    else:
        a = ("PICKUP", block)
    state = execute_action(state, a)
    actions.append(a)
    pd = ("PUTDOWN", block)
    state = execute_action(state, pd)
    actions.append(pd)
    return actions, state


def _clear_block(state: PalletState, block: str) -> Tuple[List[tuple], PalletState]:
    """
    Move everything on top of `block` to PALLET so block is clear.
    """
    actions = []
    while True:
        top = _find_top(state, block)
        if top is None:
            break
        # If arm holds something, put it down first
        if not state.arm_empty():
            pd = ("PUTDOWN", state.holding)
            state = execute_action(state, pd)
            actions.append(pd)
        # Clear top first (recursive)
        sub, state = _clear_block(state, top)
        actions.extend(sub)
        # Now unstack top from block
        a = ("UNSTACK", top, block)
        state = execute_action(state, a)
        actions.append(a)
        pd = ("PUTDOWN", top)
        state = execute_action(state, pd)
        actions.append(pd)
    return actions, state


def _build_tower(
    state: PalletState,
    # bottom to top, e.g. [C, B, A] means A on B on C on PALLET
    tower: List[str],
) -> Tuple[List[tuple], PalletState]:
    """
    Place blocks in `tower` from bottom to top.
    Each block is taken from wherever it currently is and placed correctly.
    """
    actions = []
    prev = "PALLET"

    for block in tower:
        # Check if already in place
        if state.on.get(block) == prev and state.clear(block):
            # Already correctly positioned — but might have garbage on top
            sub, state = _clear_block(state, block)
            actions.extend(sub)
            prev = block
            continue

        # Clear the block (remove anything on it)
        if not state.arm_empty():
            pd = ("PUTDOWN", state.holding)
            state = execute_action(state, pd)
            actions.append(pd)
        sub, state = _clear_block(state, block)
        actions.extend(sub)

        # Clear the destination (prev)
        if prev != "PALLET":
            sub2, state = _clear_block(state, prev)
            actions.extend(sub2)

        # Arm must be empty
        if not state.arm_empty():
            pd = ("PUTDOWN", state.holding)
            state = execute_action(state, pd)
            actions.append(pd)

        # Pick up block
        if state.on.get(block) == "PALLET":
            a = ("PICKUP", block)
        else:
            a = ("UNSTACK", block, state.on[block])
        state = execute_action(state, a)
        actions.append(a)

        # Place on prev
        if prev == "PALLET":
            a2 = ("PUTDOWN", block)
        else:
            a2 = ("STACK", block, prev)
        state = execute_action(state, a2)
        actions.append(a2)

        prev = block

    return actions, state


def _infer_towers(goal: dict) -> List[List[str]]:
    """
    Infer the goal tower(s) from an 'on' goal dict.
    Returns a list of towers, each expressed bottom-to-top.
    E.g. {A:B, B:C} → towers = [[C, B, A]] (C is bottom: on PALLET)
    """
    on_map = goal.get("on", {})  # block -> support
    # Find all "root" blocks: blocks whose support is PALLET or not in on_map
    # blocks = list(on_map.keys())

    # Invert: support -> block (who goes on top of support?)
    over: Dict[str, str] = {sup: blk for blk,
                            sup in on_map.items() if sup != "PALLET"}

    # Find blocks that go directly on PALLET in the goal
    roots = [blk for blk, sup in on_map.items() if sup == "PALLET"]
    # Also: blocks whose support is NOT in on_map (support is "base")
    # base_supports = {sup for blk, sup in on_map.items(
    # ) if sup != "PALLET" and sup not in on_map}

    towers = []

    # Start from each block that is placed on PALLET (explicitly)
    for root in roots:
        tower = [root]
        cur = root
        while cur in over:
            cur = over[cur]
            tower.append(cur)
        towers.append(tower)

    # For goals where the support is an existing block (not being moved):
    # e.g. goal = {A: B} where B is not being moved (not in on_map keys).
    for blk, sup in on_map.items():
        if sup != "PALLET" and sup not in on_map:
            # sup is an "anchor" — not being repositioned
            # Build the tower from sup upward
            tower = [sup, blk]
            cur = blk
            while cur in over:
                cur = over[cur]
                tower.append(cur)
            towers.append(tower)

    if not towers:
        # Simple case: all goals are independent On(x,y) without chain
        for blk, sup in on_map.items():
            towers.append([sup, blk] if sup != "PALLET" else [blk])

    return towers


def generate_goal_stack_plan(start: PalletState, goal: dict, max_steps: int = 500) -> Tuple[List[tuple], List[str]]:
    """
    Extended Goal Stack Planning per §3.3.2.
    Builds goal towers bottom-up with appropriate clearing.
    Returns (plan, trace).
    """
    towers = _infer_towers(goal)
    state = start.copy()
    plan: List[tuple] = []
    trace: List[str] = []

    for tower in towers:
        actions, state = _build_tower(state, tower)
        plan.extend(actions)
        trace.append(f"Built tower {tower}: {actions}")

    return plan, trace
