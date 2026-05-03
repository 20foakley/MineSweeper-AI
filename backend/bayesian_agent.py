from __future__ import annotations

from collections import defaultdict
import random


class BayesianAgent:
    def __init__(self, total_mines: int | None = None, max_exact_vars: int = 18) -> None:
        self.total_mines = total_mines
        self.max_exact_vars = max_exact_vars

    def select_action(
        self,
        state: list[list[int]],
        valid_actions: list[tuple[int, int]],
    ) -> tuple[int, int]:
        if not valid_actions:
            raise ValueError("No valid actions available for BayesianAgent.")

        unknown_set = set(valid_actions)
        constraints = self._collect_constraints(state, unknown_set)
        probability_by_cell = self._estimate_probabilities(
            state=state,
            valid_actions=valid_actions,
            constraints=constraints,
        )

        min_prob = min(probability_by_cell.values())
        best_actions = [cell for cell, prob in probability_by_cell.items() if prob == min_prob]
        return random.choice(best_actions)

    def _collect_constraints(
        self,
        state: list[list[int]],
        unknown_set: set[tuple[int, int]],
    ) -> list[tuple[list[tuple[int, int]], int]]:
        size = len(state)
        constraints: list[tuple[list[tuple[int, int]], int]] = []

        for row in range(size):
            for col in range(size):
                number = state[row][col]
                if number < 0:
                    continue
                unknown_neighbors = [
                    (nr, nc)
                    for nr, nc in self._neighbors(row, col, size)
                    if (nr, nc) in unknown_set
                ]
                if unknown_neighbors:
                    constraints.append((unknown_neighbors, number))

        return constraints

    def _estimate_probabilities(
        self,
        state: list[list[int]],
        valid_actions: list[tuple[int, int]],
        constraints: list[tuple[list[tuple[int, int]], int]],
    ) -> dict[tuple[int, int], float]:
        global_prior = self._global_fallback_prob(state, len(valid_actions))
        probabilities: dict[tuple[int, int], float] = {action: global_prior for action in valid_actions}

        if not constraints:
            return probabilities

        var_to_constraints: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (variables, _) in enumerate(constraints):
            for variable in variables:
                var_to_constraints[variable].append(idx)

        frontier_vars = list(var_to_constraints.keys())
        seen_vars: set[tuple[int, int]] = set()
        for root in frontier_vars:
            if root in seen_vars:
                continue

            component_vars, component_constraints = self._extract_component(
                root=root,
                var_to_constraints=var_to_constraints,
                constraints=constraints,
            )
            seen_vars.update(component_vars)
            component_probs = self._component_probabilities(component_vars, component_constraints)
            for cell, probability in component_probs.items():
                probabilities[cell] = probability

        return probabilities

    def _extract_component(
        self,
        root: tuple[int, int],
        var_to_constraints: dict[tuple[int, int], list[int]],
        constraints: list[tuple[list[tuple[int, int]], int]],
    ) -> tuple[list[tuple[int, int]], list[tuple[list[tuple[int, int]], int]]]:
        queue = [root]
        seen_vars: set[tuple[int, int]] = set()
        seen_constraints: set[int] = set()

        while queue:
            cell = queue.pop()
            if cell in seen_vars:
                continue
            seen_vars.add(cell)

            for constraint_idx in var_to_constraints[cell]:
                if constraint_idx in seen_constraints:
                    continue
                seen_constraints.add(constraint_idx)
                variables, _ = constraints[constraint_idx]
                for variable in variables:
                    if variable not in seen_vars:
                        queue.append(variable)

        component_constraints = [constraints[idx] for idx in seen_constraints]
        return list(seen_vars), component_constraints

    def _component_probabilities(
        self,
        variables: list[tuple[int, int]],
        constraints: list[tuple[list[tuple[int, int]], int]],
    ) -> dict[tuple[int, int], float]:
        if len(variables) > self.max_exact_vars:
            return self._approx_component_probabilities(variables, constraints)

        var_to_idx = {var: idx for idx, var in enumerate(variables)}
        indexed_constraints: list[tuple[list[int], int]] = []
        for vars_list, target in constraints:
            indexed_constraints.append(([var_to_idx[var] for var in vars_list], target))

        assignment = [-1] * len(variables)
        mine_counts = [0] * len(variables)
        solutions = 0

        def is_consistent() -> bool:
            for var_indices, target in indexed_constraints:
                assigned_sum = 0
                unassigned = 0
                for idx in var_indices:
                    value = assignment[idx]
                    if value == -1:
                        unassigned += 1
                    else:
                        assigned_sum += value

                if assigned_sum > target:
                    return False
                if assigned_sum + unassigned < target:
                    return False
            return True

        def backtrack(pos: int) -> None:
            nonlocal solutions
            if pos == len(variables):
                solutions += 1
                for idx, value in enumerate(assignment):
                    mine_counts[idx] += value
                return

            for value in (0, 1):
                assignment[pos] = value
                if is_consistent():
                    backtrack(pos + 1)
            assignment[pos] = -1

        backtrack(0)

        if solutions == 0:
            return self._approx_component_probabilities(variables, constraints)

        return {
            variables[idx]: mine_counts[idx] / solutions
            for idx in range(len(variables))
        }

    def _approx_component_probabilities(
        self,
        variables: list[tuple[int, int]],
        constraints: list[tuple[list[tuple[int, int]], int]],
    ) -> dict[tuple[int, int], float]:
        local_constraints: dict[tuple[int, int], list[float]] = defaultdict(list)
        for vars_list, target in constraints:
            count = len(vars_list)
            if count == 0:
                continue
            prob = max(0.0, min(1.0, target / count))
            for variable in vars_list:
                local_constraints[variable].append(prob)

        probabilities: dict[tuple[int, int], float] = {}
        for variable in variables:
            probs = local_constraints.get(variable, [])
            if probs:
                probabilities[variable] = self._combine_probs(probs)
            else:
                probabilities[variable] = 0.5
        return probabilities

    @staticmethod
    def _combine_probs(probabilities: list[float]) -> float:
        # Independence approximation: P(any mine source) = 1 - product(1 - p_i)
        prod = 1.0
        for prob in probabilities:
            prod *= (1.0 - prob)
        return 1.0 - prod

    def _global_fallback_prob(self, state: list[list[int]], unknown_count: int) -> float:
        if unknown_count == 0:
            return 1.0
        if self.total_mines is None:
            total = len(state) * len(state)
            return unknown_count / total
        return max(0.0, min(1.0, self.total_mines / unknown_count))

    @staticmethod
    def _neighbors(row: int, col: int, size: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < size and 0 <= nc < size:
                    cells.append((nr, nc))
        return cells
