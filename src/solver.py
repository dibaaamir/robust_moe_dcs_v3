from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import numpy as np

from .environment import SafeControllerGrid, GridState


@dataclass
class SolveResult:
    solved: bool
    expansions: int
    path: list[GridState]


def _reconstruct(parent: dict[GridState, GridState | None], goal: GridState) -> list[GridState]:
    path = []
    cur: GridState | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def directed_search(env: SafeControllerGrid, policy, budget: int = 90) -> SolveResult:
    """Policy-guided on-the-fly best-first exploration."""
    counter = itertools.count()
    frontier: list[tuple[float, int, GridState]] = [(0.0, next(counter), env.start)]
    parent: dict[GridState, GridState | None] = {env.start: None}
    path_cost: dict[GridState, float] = {env.start: 0.0}
    expansions = 0

    while frontier and expansions < budget:
        _, _, state = heapq.heappop(frontier)
        expansions += 1
        if state == env.goal:
            return SolveResult(True, expansions, _reconstruct(parent, state))
        probs = policy.action_probabilities(env, state)
        for action in env.valid_actions(state):
            nxt, _, _ = env.step(state, action)
            if nxt in parent:
                continue
            action_cost = -float(np.log(max(1e-9, probs[action])))
            candidate = path_cost[state] + action_cost
            remaining_x = env.goal.x - nxt.x
            remaining_y = env.goal.y - nxt.y
            heuristic = 0.025 * max(remaining_x, remaining_y)
            parent[nxt] = state
            path_cost[nxt] = candidate
            heapq.heappush(frontier, (candidate + heuristic, next(counter), nxt))

    return SolveResult(False, expansions, [])
