from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

ACTIONS = ((1, 0), (0, 1), (1, 1))
ACTION_NAMES = ("plane-progress", "altitude-progress", "joint-progress")


@dataclass(frozen=True)
class GridState:
    x: int
    y: int


class SafeControllerGrid:
    """Synthetic budget-limited controller-synthesis domain.

    The domain deliberately contains parameter-dependent search structures:
    wide instances favour long horizontal phases, tall instances favour vertical
    phases, and balanced instances favour diagonal/alternating phases. A safe
    route always exists, but decoy branches make the exploration order matter.
    """

    def __init__(self, n: int, k: int, seed: int = 0, obstacle_rate: float = 0.22):
        if n < 2 or k < 1:
            raise ValueError("n must be >= 2 and k must be >= 1")
        self.n = n
        self.k = k
        self.width = n
        self.height = max(2, k + 1)
        self.start = GridState(0, 0)
        self.goal = GridState(self.width - 1, self.height - 1)
        self.seed = seed
        self.obstacle_rate = obstacle_rate
        self.regime = self._regime()
        self.safe_corridor = self._build_corridor()
        self.unsafe = self._build_unsafe_cells()

    def _regime(self) -> str:
        ratio = self.width / self.height
        if ratio >= 1.35:
            return "wide"
        if ratio <= 0.74:
            return "tall"
        return "balanced"

    def _build_corridor(self) -> set[tuple[int, int]]:
        """Create a deterministic but seed-varied safe backbone."""
        rng = np.random.default_rng(17_171 + 131 * self.n + 977 * self.k + self.seed)
        x = y = 0
        corridor = {(x, y)}
        phase = int(rng.integers(0, 3))
        while (x, y) != (self.goal.x, self.goal.y):
            can_x = x < self.goal.x
            can_y = y < self.goal.y
            if self.regime == "wide":
                # Long horizontal phases interrupted by short vertical corrections.
                choose_x = can_x and (not can_y or ((x + phase) % 4 != 3))
                dx, dy = (1, 0) if choose_x else (0, 1)
            elif self.regime == "tall":
                choose_y = can_y and (not can_x or ((y + phase) % 4 != 3))
                dx, dy = (0, 1) if choose_y else (1, 0)
            else:
                # Balanced instances prefer diagonal progress, with alternating turns.
                if can_x and can_y and ((x + y + phase) % 3 != 1):
                    dx, dy = (1, 1)
                elif can_x and (not can_y or (x + phase) % 2 == 0):
                    dx, dy = (1, 0)
                else:
                    dx, dy = (0, 1)
            x += dx
            y += dy
            corridor.add((x, y))
        return corridor

    def _build_unsafe_cells(self) -> set[tuple[int, int]]:
        rng = np.random.default_rng(self.seed + 10_007 * self.n + 100_003 * self.k)
        unsafe: set[tuple[int, int]] = set()
        corridor = self.safe_corridor
        for x in range(self.width):
            for y in range(self.height):
                cell = (x, y)
                if cell in corridor or cell in {(0, 0), (self.goal.x, self.goal.y)}:
                    continue
                # Keep many decoy branches open near the start, but make them terminate.
                distance = x + y
                decoy_open = distance < max(3, (self.width + self.height) // 4)
                base = self.obstacle_rate - (0.10 if decoy_open else 0.0)
                if self.regime == "wide" and y % 3 == 1:
                    base += 0.23
                elif self.regime == "tall" and x % 3 == 1:
                    base += 0.23
                elif self.regime == "balanced" and (x - y) % 4 in (1, 2):
                    base += 0.18
                # Later cells off the safe backbone are more likely to become dead ends.
                if distance > (self.width + self.height) * 0.55:
                    base += 0.12
                if rng.random() < min(0.72, max(0.03, base)):
                    unsafe.add(cell)
        return unsafe

    def valid_actions(self, state: GridState) -> list[int]:
        valid: list[int] = []
        for idx, (dx, dy) in enumerate(ACTIONS):
            nx, ny = state.x + dx, state.y + dy
            if nx >= self.width or ny >= self.height:
                continue
            if (nx, ny) in self.unsafe:
                continue
            valid.append(idx)
        return valid

    def step(self, state: GridState, action: int) -> tuple[GridState, float, bool]:
        if action not in self.valid_actions(state):
            return state, -2.5, False
        dx, dy = ACTIONS[action]
        nxt = GridState(state.x + dx, state.y + dy)
        done = nxt == self.goal
        progress = dx / max(1, self.width - 1) + dy / max(1, self.height - 1)
        # Reward safe progress, but avoid making the diagonal universally optimal.
        action_bonus = 0.0
        if self.regime == "wide" and action == 0:
            action_bonus = 0.08
        elif self.regime == "tall" and action == 1:
            action_bonus = 0.08
        elif self.regime == "balanced" and action == 2:
            action_bonus = 0.08
        reward = 12.0 if done else -0.08 + 0.35 * progress + action_bonus
        return nxt, reward, done

    def state_key(self, state: GridState, bins: int = 10) -> tuple[int, int, int, int]:
        """Scale-independent local representation.

        The policy sees progress and the local action mask, but not the explicit
        domain parameters. This allows transfer while retaining learned specialist
        behaviour from different training distributions.
        """
        bx = min(bins - 1, int(bins * state.x / max(1, self.width - 1)))
        by = min(bins - 1, int(bins * state.y / max(1, self.height - 1)))
        mask = 0
        for action in self.valid_actions(state):
            mask |= 1 << action
        phase = (bx + by) % 3
        return bx, by, mask, phase

    def render_ascii(self, path: Iterable[GridState] | None = None) -> str:
        path_cells = {(s.x, s.y) for s in path or []}
        rows = []
        for y in reversed(range(self.height)):
            row = []
            for x in range(self.width):
                cell = (x, y)
                if cell == (self.start.x, self.start.y):
                    row.append("S")
                elif cell == (self.goal.x, self.goal.y):
                    row.append("G")
                elif cell in path_cells:
                    row.append("*")
                elif cell in self.unsafe:
                    row.append("#")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)
