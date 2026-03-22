"""Monte Carlo Tree Search for multi-step retrosynthesis route finding.

Uses the RL policy for node expansion and reward function for evaluation.
Finds complete synthesis routes from target molecule to buyable starting materials.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCTSNode:
    """A node in the MCTS search tree representing a molecule.

    Attributes:
        smiles: SMILES string of this molecule.
        depth: Depth in the synthesis tree (0 = target molecule).
        parent: Parent node (None for root).
        children: List of child node groups (each group is one reactant set).
        visit_count: Number of times this node has been visited.
        total_value: Sum of all backpropagated values through this node.
        is_terminal: Whether this molecule is in the stock list (buyable).
        is_expanded: Whether this node has been expanded by the policy.
    """

    smiles: str
    depth: int = 0
    parent: Optional["MCTSNode"] = None
    children: list[list["MCTSNode"]] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    is_terminal: bool = False
    is_expanded: bool = False

    @property
    def value(self) -> float:
        """Average value of this node (total_value / visit_count)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def uct_score(self, exploration_constant: float = 1.41) -> float:
        """Compute Upper Confidence Bound for Trees (UCT) score.

        UCT = value/visits + c * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_constant: Controls exploration vs exploitation tradeoff.

        Returns:
            UCT score. Returns infinity for unvisited nodes.
        """
        if self.visit_count == 0:
            return float("inf")
        if self.parent is None:
            return self.value
        exploitation = self.total_value / self.visit_count
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return exploitation + exploration


@dataclass
class MCTSResult:
    """Result of an MCTS search.

    Attributes:
        best_route: Tree dict of the best synthesis route found.
        score: Score of the best route.
        all_routes: List of all complete routes found, sorted by score.
        stats: Search statistics.
    """

    best_route: Optional[dict] = None
    score: float = 0.0
    all_routes: list[dict] = field(default_factory=list)
    stats: dict = field(
        default_factory=lambda: {
            "simulations": 0,
            "time_seconds": 0.0,
            "routes_found": 0,
            "max_depth_reached": 0,
        }
    )


class MCTS:
    """Monte Carlo Tree Search engine for retrosynthesis.

    Finds multi-step synthesis routes from a target molecule to
    commercially available starting materials.

    Usage:
        from models.policy import RetroPolicy
        from env.Rewards import RewardCalculator
        from data.stock.loader import StockList

        policy = RetroPolicy()
        rewards = RewardCalculator()
        stock = StockList()
        stock.load()

        mcts = MCTS(policy, rewards, stock)
        result = mcts.search("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # ibuprofen
        print(result.best_route)
    """

    def __init__(
        self,
        policy,
        reward_calculator,
        stock_list,
        max_depth: int = 6,
        max_simulations: int = 500,
        top_k: int = 10,
        exploration_constant: float = 1.41,
    ) -> None:
        """Initialize MCTS search engine.

        Args:
            policy: RetroPolicy instance for generating reactant predictions.
            reward_calculator: RewardCalculator instance for scoring nodes.
            stock_list: StockList instance for checking buyability.
            max_depth: Maximum depth of the synthesis tree.
            max_simulations: Maximum number of MCTS iterations.
            top_k: Number of reactant candidates to expand per node.
            exploration_constant: UCT exploration constant (c).
        """
        self.policy = policy
        self.reward_calculator = reward_calculator
        self.stock_list = stock_list
        self.max_depth = max_depth
        self.max_simulations = max_simulations
        self.top_k = top_k
        self.exploration_constant = exploration_constant

    def search(self, target_smiles: str, time_budget: float = 60.0) -> MCTSResult:
        """Run MCTS to find synthesis routes for a target molecule.

        Terminates when any of these conditions is met:
        - max_simulations reached
        - time_budget exceeded
        - High-confidence complete route found (all leaves buyable)

        Args:
            target_smiles: SMILES string of the target molecule.
            time_budget: Maximum search time in seconds.

        Returns:
            MCTSResult with best route, all routes, and search statistics.
        """
        start_time = time.time()
        max_depth_reached = 0

        # Create root node
        root = MCTSNode(
            smiles=target_smiles,
            depth=0,
            parent=None,
            is_terminal=self.stock_list.is_buyable(target_smiles),
        )

        # Edge case: target is already buyable
        if root.is_terminal:
            route = {
                "smiles": target_smiles,
                "score": 1.0,
                "in_stock": True,
                "children": [],
            }
            return MCTSResult(
                best_route=route,
                score=1.0,
                all_routes=[route],
                stats={
                    "simulations": 0,
                    "time_seconds": time.time() - start_time,
                    "routes_found": 1,
                    "max_depth_reached": 0,
                },
            )

        simulations = 0

        for _ in range(self.max_simulations):
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                break

            # Select
            leaf = self._select(root)

            # Track max depth
            if leaf.depth > max_depth_reached:
                max_depth_reached = leaf.depth

            # Expand (if not terminal and not at max depth)
            if not leaf.is_terminal and not leaf.is_expanded and leaf.depth < self.max_depth:
                new_children = self._expand(leaf)
                # If expansion produced children, pick the first unvisited one for simulation
                if new_children:
                    leaf = new_children[0]
                    if leaf.depth > max_depth_reached:
                        max_depth_reached = leaf.depth

            # Simulate
            value = self._simulate(leaf)

            # Backpropagate
            self._backpropagate(leaf, value)

            simulations += 1

            # Early termination check: look for a high-confidence complete route
            if simulations % 50 == 0:
                routes = self.extract_routes(root, top_n=1)
                if (
                    routes
                    and routes[0].get("score", 0) > 0.95
                    and self._is_route_complete(routes[0])
                ):
                    break

        # Extract final routes
        all_routes = self.extract_routes(root, top_n=3)
        best_route = all_routes[0] if all_routes else None
        best_score = best_route["score"] if best_route else 0.0

        return MCTSResult(
            best_route=best_route,
            score=best_score,
            all_routes=all_routes,
            stats={
                "simulations": simulations,
                "time_seconds": time.time() - start_time,
                "routes_found": len(all_routes),
                "max_depth_reached": max_depth_reached,
            },
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCT to find a leaf node.

        At each level, pick the child group with best average UCT, then
        pick the molecule in that group with the highest UCT score.

        Args:
            node: Root node to start selection from.

        Returns:
            Selected leaf node for expansion.
        """
        current = node

        while current.is_expanded and current.children and not current.is_terminal:
            # Pick the child group with the best average UCT score
            best_group = None
            best_group_score = float("-inf")

            for group in current.children:
                if not group:
                    continue
                # Average UCT across all molecules in this group
                group_score = sum(
                    child.uct_score(self.exploration_constant) for child in group
                ) / len(group)
                if group_score > best_group_score:
                    best_group_score = group_score
                    best_group = group

            if best_group is None:
                break

            # Within the best group, pick the molecule with the highest UCT
            best_child = max(
                best_group,
                key=lambda c: c.uct_score(self.exploration_constant),
            )

            # If the best child is terminal (buyable), try to find a non-terminal
            # sibling in the same group, or move to a non-terminal child in the group
            # that still needs expansion
            if best_child.is_terminal:
                non_terminal = [c for c in best_group if not c.is_terminal and not c.is_expanded]
                if non_terminal:
                    best_child = max(
                        non_terminal,
                        key=lambda c: c.uct_score(self.exploration_constant),
                    )
                else:
                    # All children in best group are terminal or expanded;
                    # descend into expanded non-terminal children
                    expandable = [
                        c for c in best_group if not c.is_terminal and c.is_expanded and c.children
                    ]
                    if expandable:
                        best_child = max(
                            expandable,
                            key=lambda c: c.uct_score(self.exploration_constant),
                        )
                    else:
                        # Everything in this group is fully resolved
                        break

            current = best_child

        return current

    def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        """Expansion phase: use policy to generate children for a node.

        Calls policy.predict_greedy() to get top-K reactant sets,
        then creates child nodes for each reactant.

        Args:
            node: Node to expand.

        Returns:
            List of newly created child nodes.
        """
        node.is_expanded = True
        all_new_children: list[MCTSNode] = []

        try:
            predictions = self.policy.predict_greedy(node.smiles, num_beams=self.top_k)
        except Exception:
            return all_new_children

        if not predictions:
            return all_new_children

        for prediction in predictions:
            if not prediction or not isinstance(prediction, str):
                continue

            # Each prediction may contain '.' separating multiple reactants
            reactant_smiles_list = [s.strip() for s in prediction.split(".") if s.strip()]
            if not reactant_smiles_list:
                continue

            # Create a group of child nodes (one per reactant in this disconnection)
            group: list[MCTSNode] = []
            for reactant_smi in reactant_smiles_list:
                child = MCTSNode(
                    smiles=reactant_smi,
                    depth=node.depth + 1,
                    parent=node,
                    is_terminal=self.stock_list.is_buyable(reactant_smi),
                )
                group.append(child)

            if group:
                node.children.append(group)
                all_new_children.extend(group)

        return all_new_children

    def _simulate(self, node: MCTSNode) -> float:
        """Simulation phase: rollout from node using greedy policy.

        Continues until all molecules are buyable or max_depth reached.

        Args:
            node: Starting node for simulation.

        Returns:
            Reward value from the simulation rollout.
        """
        # If terminal, this molecule is already buyable
        if node.is_terminal:
            return 1.0

        # Greedy rollout: track molecules that still need to be resolved
        pending = [(node.smiles, node.depth)]
        total_reward = 0.0
        steps = 0
        max_steps = self.max_depth - node.depth  # remaining depth budget

        if max_steps <= 0:
            # At max depth and not terminal: score what we have
            try:
                reward = self.reward_calculator.combined_reward(
                    node.smiles, [node.smiles], self.stock_list
                )
                return reward
            except Exception:
                return 0.0

        resolved_count = 0
        total_molecules = 0

        while pending and steps < max_steps:
            next_pending = []

            for smi, depth in pending:
                total_molecules += 1

                if self.stock_list.is_buyable(smi):
                    resolved_count += 1
                    continue

                if depth >= self.max_depth:
                    # Can't go deeper; score this molecule as-is
                    try:
                        reward = self.reward_calculator.combined_reward(smi, [smi], self.stock_list)
                        total_reward += reward
                    except Exception:
                        pass
                    continue

                # Get greedy prediction for this molecule
                try:
                    predictions = self.policy.predict_greedy(smi, num_beams=1)
                except Exception:
                    continue

                if not predictions:
                    continue

                best_prediction = predictions[0]
                if not best_prediction or not isinstance(best_prediction, str):
                    continue

                reactants = [s.strip() for s in best_prediction.split(".") if s.strip()]
                if not reactants:
                    continue

                # Score this step
                try:
                    step_reward = self.reward_calculator.combined_reward(
                        smi, reactants, self.stock_list
                    )
                    total_reward += step_reward
                except Exception:
                    pass

                # Add reactants to next round of pending
                for reactant in reactants:
                    if self.stock_list.is_buyable(reactant):
                        resolved_count += 1
                        total_molecules += 1
                    else:
                        next_pending.append((reactant, depth + 1))

            pending = next_pending
            steps += 1

        # Combine: fraction resolved + accumulated step rewards
        if total_molecules > 0:
            resolution_bonus = resolved_count / total_molecules
        else:
            resolution_bonus = 0.0

        if steps > 0:
            avg_step_reward = total_reward / steps
        else:
            avg_step_reward = 0.0

        # Weighted combination: resolution matters more
        combined = 0.6 * resolution_bonus + 0.4 * avg_step_reward
        return min(max(combined, 0.0), 1.0)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagation phase: update values from leaf to root.

        Args:
            node: Leaf node to start backpropagation from.
            value: Reward value to propagate.
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    def extract_routes(self, root: MCTSNode, top_n: int = 3) -> list[dict]:
        """Extract the top-N synthesis routes from the search tree.

        A route is "complete" when all leaf molecules are in the stock list.

        Route tree format:
        {
            "smiles": str,
            "score": float,
            "in_stock": bool,
            "children": [...]  # recursive
        }

        Args:
            root: Root node of the search tree.
            top_n: Number of top routes to return.

        Returns:
            List of route dicts, sorted by score descending.
        """
        complete_routes: list[dict] = []
        partial_routes: list[dict] = []

        if not root.children:
            # No expansions happened; return the root as a trivial route
            route = {
                "smiles": root.smiles,
                "score": root.value,
                "in_stock": root.is_terminal,
                "children": [],
            }
            if root.is_terminal:
                return [route]
            return [route] if top_n > 0 else []

        # Each child group represents a different disconnection strategy.
        # Build a route tree for each group and score it.
        for group in root.children:
            route_tree = self._build_route_tree(root, group)
            if route_tree is not None:
                if self._is_route_complete(route_tree):
                    complete_routes.append(route_tree)
                else:
                    partial_routes.append(route_tree)

        # Sort by score descending
        complete_routes.sort(key=lambda r: r["score"], reverse=True)
        partial_routes.sort(key=lambda r: r["score"], reverse=True)

        # Prefer complete routes; fall back to partial if needed
        result = complete_routes[:top_n]
        if len(result) < top_n:
            remaining = top_n - len(result)
            result.extend(partial_routes[:remaining])

        return result

    def _build_route_tree(self, node: MCTSNode, group: list[MCTSNode]) -> Optional[dict]:
        """Recursively build a route dict from a node and one of its child groups.

        Args:
            node: The parent node.
            group: One child group (list of reactant nodes for one disconnection).

        Returns:
            Route dict, or None if the group is empty.
        """
        if not group:
            return None

        children_dicts = []
        group_score_sum = 0.0
        group_count = 0

        for child in group:
            child_dict: dict
            if child.is_terminal:
                child_dict = {
                    "smiles": child.smiles,
                    "score": max(child.value, 1.0) if child.visit_count > 0 else 1.0,
                    "in_stock": True,
                    "children": [],
                }
            elif child.children:
                # Pick the best child group for this node (highest average value)
                best_sub = None
                best_sub_score = float("-inf")
                for sub_group in child.children:
                    sub_score = (
                        sum(c.value for c in sub_group) / len(sub_group) if sub_group else 0.0
                    )
                    if sub_score > best_sub_score:
                        best_sub_score = sub_score
                        best_sub = sub_group

                if best_sub is not None:
                    child_dict_result = self._build_route_tree(child, best_sub)
                    child_dict = (
                        child_dict_result
                        if child_dict_result
                        else {
                            "smiles": child.smiles,
                            "score": child.value,
                            "in_stock": False,
                            "children": [],
                        }
                    )
                else:
                    child_dict = {
                        "smiles": child.smiles,
                        "score": child.value,
                        "in_stock": False,
                        "children": [],
                    }
            else:
                # Leaf that is not terminal and not expanded
                child_dict = {
                    "smiles": child.smiles,
                    "score": child.value,
                    "in_stock": False,
                    "children": [],
                }

            children_dicts.append(child_dict)
            group_score_sum += child_dict["score"]
            group_count += 1

        # Route score: average of children scores weighted with parent value
        children_avg = group_score_sum / group_count if group_count > 0 else 0.0
        if node.visit_count > 0:
            route_score = 0.5 * node.value + 0.5 * children_avg
        else:
            route_score = children_avg

        return {
            "smiles": node.smiles,
            "score": route_score,
            "in_stock": node.is_terminal,
            "children": children_dicts,
        }

    @staticmethod
    def _is_route_complete(route: dict) -> bool:
        """Check if all leaves of a route tree are in stock.

        Args:
            route: A route dict with recursive "children".

        Returns:
            True if every leaf node has in_stock=True.
        """
        if not route.get("children"):
            # Leaf node
            return route.get("in_stock", False)
        return all(MCTS._is_route_complete(child) for child in route["children"])
