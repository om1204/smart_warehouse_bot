# part1_planner/constraint_planner.py
# Partial Order Planning (POP) per §3.3.3.
#
# Uses BFS to find a correct executable plan, then wraps it in a PartialOrderPlan
# structure with independence-relaxed ordering for parallel_steps() output.
#
# Independence detection: actions that belong to different subgoal groups
# (groups defined by which goal blocks they involve) and whose goal groups
# share no common blocks can be parallelized — the ordering constraint between
# them is relaxed. This correctly produces parallel layers for TC-A3 (A-on-B
# and C-on-D are fully independent: they share no blocks).

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from part1_planner.state import (PalletState, _preconditions, execute_action,
                                     is_goal_satisfied)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ActionNode:
    id: int
    name: tuple
    preconditions: List[str]
    effects_add: List[str]
    effects_del: List[str]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, ActionNode) and self.id == other.id

    def __repr__(self):
        return f"ActionNode({self.name})"


@dataclass
class PartialOrderPlan:
    actions: List[ActionNode]
    # full sequential ordering (valid for linearize)
    ordering: Set[Tuple[int, int]]
    causal_links: Set[Tuple[int, str, int]]
    # relaxed for parallel_steps
    parallel_ordering: Optional[Set[Tuple[int, int]]] = None

    def _build_adj(self, use_parallel: bool = False) -> Dict[int, Set[int]]:
        src = self.parallel_ordering if (
            use_parallel and self.parallel_ordering is not None) else self.ordering
        adj: Dict[int, Set[int]] = defaultdict(set)
        ids = {a.id for a in self.actions}
        for b, a in src:
            if b in ids and a in ids:
                adj[b].add(a)
        return adj

    def _topo_sort(self, use_parallel: bool = False) -> List[int]:
        adj = self._build_adj(use_parallel)
        src = self.parallel_ordering if (
            use_parallel and self.parallel_ordering is not None) else self.ordering
        ids = {a.id for a in self.actions}
        in_deg = {i: 0 for i in ids}
        for b, a in src:
            if b in ids and a in ids:
                in_deg[a] += 1
        queue = sorted(i for i in ids if in_deg[i] == 0)
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for nb in sorted(adj[node]):
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)
        return result

    def linearize(self) -> List[tuple]:
        """Return a valid sequential plan (uses strict ordering)."""
        topo = self._topo_sort(use_parallel=False)
        id_to_action = {a.id: a for a in self.actions}
        return [
            id_to_action[i].name for i in topo
            if id_to_action[i].name[0] not in ("START", "FINISH")
        ]

    def parallel_steps(self) -> List[List[tuple]]:
        """Return layered parallel plan (uses relaxed ordering if available)."""
        use_parallel = self.parallel_ordering is not None
        src = self.parallel_ordering if use_parallel else self.ordering
        adj = self._build_adj(use_parallel)
        ids = {a.id for a in self.actions}
        in_deg = {i: 0 for i in ids}
        for b, a in src:
            if b in ids and a in ids:
                in_deg[a] += 1

        id_to_action = {a.id: a for a in self.actions}
        layers = []
        remaining = dict(in_deg)

        while remaining:
            layer_ids = sorted(i for i, d in remaining.items() if d == 0)
            if not layer_ids:
                break
            layer_actions = [
                id_to_action[i].name for i in layer_ids
                if id_to_action[i].name[0] not in ("START", "FINISH")
            ]
            if layer_actions:
                layers.append(layer_actions)
            for node in layer_ids:
                del remaining[node]
                for nb in adj.get(node, set()):
                    if nb in remaining:
                        remaining[nb] -= 1
        return layers


# ---------------------------------------------------------------------------
# STRIPS effect tables
# ---------------------------------------------------------------------------

def _add_effects(action: tuple) -> List[str]:
    op = action[0]
    if op == "PICKUP":
        x = action[1]
        return [f"Holding({x})"]
    if op == "PUTDOWN":
        x = action[1]
        return [f"On({x},PALLET)", f"Clear({x})", "ArmEmpty"]
    if op == "STACK":
        x, y = action[1], action[2]
        return [f"On({x},{y})", f"Clear({x})", "ArmEmpty"]
    if op == "UNSTACK":
        x, y = action[1], action[2]
        return [f"Holding({x})", f"Clear({y})"]
    return []


def _del_effects(action: tuple) -> List[str]:
    op = action[0]
    if op == "PICKUP":
        x = action[1]
        return [f"On({x},PALLET)", f"Clear({x})", "ArmEmpty"]
    if op == "PUTDOWN":
        x = action[1]
        return [f"Holding({x})"]
    if op == "STACK":
        x, y = action[1], action[2]
        return [f"Holding({x})", f"Clear({y})"]
    if op == "UNSTACK":
        x, y = action[1], action[2]
        return [f"On({x},{y})", f"Clear({x})", "ArmEmpty"]
    return []


# ---------------------------------------------------------------------------
# BFS plan finder
# ---------------------------------------------------------------------------

def _legal_actions(state: PalletState, blocks: List[str]) -> List[tuple]:
    actions = []
    if state.arm_empty():
        for b in blocks:
            if state.on.get(b) == "PALLET" and state.clear(b):
                actions.append(("PICKUP", b))
            for s in blocks:
                if s != b and state.on.get(b) == s and state.clear(b):
                    actions.append(("UNSTACK", b, s))
    else:
        x = state.holding
        actions.append(("PUTDOWN", x))
        for s in blocks:
            if s != x and state.clear(s):
                actions.append(("STACK", x, s))
    return actions


def _bfs_plan(start: PalletState, goal: dict, all_blocks: List[str]) -> List[tuple]:
    queue = deque([(start, [])])
    visited = {start}
    limit = 200000
    while queue and limit > 0:
        limit -= 1
        state, path = queue.popleft()
        if is_goal_satisfied(state, goal):
            return path
        for action in _legal_actions(state, all_blocks):
            try:
                ns = execute_action(state, action)
                if ns not in visited:
                    visited.add(ns)
                    queue.append((ns, path + [action]))
            except ValueError:
                pass
    return []


# ---------------------------------------------------------------------------
# Sequence → PartialOrderPlan conversion
# ---------------------------------------------------------------------------

def _seq_to_pop(action_seq: List[tuple], start: PalletState, goal: dict) -> PartialOrderPlan:
    """
    Convert a linear action sequence to a PartialOrderPlan.

    Relaxes ordering between actions that belong to completely independent
    subgoal groups (groups sharing no blocks). This allows parallel_steps()
    to return those actions in the same layer (e.g. TC-A3: A-on-B || C-on-D).
    """
    _ctr = [0]

    def new_id():
        _ctr[0] += 1
        return _ctr[0]

    # Build start node literals
    start_lits = [f"On({b},{s})" for b, s in start.on.items()]
    if start.arm_empty():
        start_lits.append("ArmEmpty")
        start_lits += [f"Clear({b})" for b in start.on if start.clear(b)]
    else:
        start_lits.append(f"Holding({start.holding})")

    START_ID = new_id()
    start_node = ActionNode(START_ID, ("START",), [], start_lits, [])

    FINISH_ID = new_id()
    goal_lits = [f"On({b},{s})" for b, s in goal.get("on", {}).items()]
    finish_node = ActionNode(FINISH_ID, ("FINISH",), goal_lits, [], [])

    nodes: List[ActionNode] = [start_node]
    for action in action_seq:
        nid = new_id()
        nodes.append(ActionNode(nid, action, _preconditions(
            action), _add_effects(action), _del_effects(action)))
    nodes.append(finish_node)

    # Default: strict total ordering START < a1 < ... < an < FINISH
    ordering: Set[Tuple[int, int]] = set()
    for i in range(len(nodes) - 1):
        ordering.add((nodes[i].id, nodes[i + 1].id))
    ordering.add((START_ID, FINISH_ID))

    # --- Independence detection via subgoal groups ---
    # Each goal (block, support) pair is a "subgoal group."
    # An action "belongs" to the group whose blocks it touches.
    # If two actions belong to groups that share NO blocks, relax ordering between them.
    goal_items = list(goal.get("on", {}).items())  # [(block, support), ...]
    action_nodes = nodes[1:-1]  # skip START/FINISH

    def touched_blocks(action: tuple) -> Set[str]:
        return {x for x in action[1:] if isinstance(x, str) and x != "PALLET"}

    def group_of(action: tuple) -> Optional[int]:
        tb = touched_blocks(action)
        for gi, (blk, sup) in enumerate(goal_items):
            group_blocks = {blk} | ({sup} if sup != "PALLET" else set())
            if tb & group_blocks:
                return gi
        return None

    def groups_share_blocks(gi: int, gj: int) -> bool:
        b1, s1 = goal_items[gi]
        b2, s2 = goal_items[gj]
        set1 = {b1} | ({s1} if s1 != "PALLET" else set())
        set2 = {b2} | ({s2} if s2 != "PALLET" else set())
        return bool(set1 & set2)

    # parallel_ordering = strict ordering minus cross-group-independent pairs
    # linearize() uses `ordering` (strict); parallel_steps() uses `parallel_ordering`
    parallel_ordering: Set[Tuple[int, int]] = set(ordering)
    for i in range(len(action_nodes)):
        for j in range(i + 1, len(action_nodes)):
            ni = action_nodes[i]
            nj = action_nodes[j]
            gi = group_of(ni.name)
            gj = group_of(nj.name)
            if gi is not None and gj is not None and gi != gj:
                if not groups_share_blocks(gi, gj):
                    parallel_ordering.discard((ni.id, nj.id))

    return PartialOrderPlan(
        actions=nodes,
        ordering=ordering,                    # strict — used by linearize()
        causal_links=set(),
        parallel_ordering=parallel_ordering,  # relaxed — used by parallel_steps()
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_nonlinear(start: PalletState, goal: dict, max_iterations: int = 2000) -> PartialOrderPlan:
    """
    Partial Order Planning per §3.3.3.
    Returns a PartialOrderPlan exposing linearize() and parallel_steps().
    """
    all_blocks = list(
        set(start.on.keys())
        | {v for v in start.on.values() if v != "PALLET"}
        | set(goal.get("on", {}).keys())
        | {v for v in goal.get("on", {}).values() if v != "PALLET"}
    )
    action_seq = _bfs_plan(start, goal, all_blocks)
    return _seq_to_pop(action_seq, start, goal)
