"""Gym-style retrosynthesis environment for episodic RL interaction.

Wraps the policy, reward calculator, and stock list into a step-based
interface for multi-step retrosynthetic planning.
"""

from typing import Any, Dict, List, Optional, Tuple


class ChemEnv:
    """Step-based retrosynthesis environment.

    Manages a set of molecules that need to be synthesized, applying
    retrosynthetic transformations one at a time until all molecules
    are commercially available (in stock) or max depth is reached.

    Usage:
        from models.policy import RetroPolicy
        from env.Rewards import RewardCalculator
        from data.stock.loader import StockList

        policy = RetroPolicy()
        rewards = RewardCalculator()
        stock = StockList()
        stock.load()

        env = ChemEnv(policy, rewards, stock)
        state = env.reset("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # ibuprofen

        while not state["done"]:
            state, reward, done, info = env.step()

        route = env.get_route()
    """

    def __init__(
        self,
        policy,
        reward_calculator,
        stock_list,
        max_depth: int = 6,
    ) -> None:
        """Initialize the retrosynthesis environment.

        Args:
            policy: RetroPolicy instance for generating retrosynthetic predictions.
            reward_calculator: RewardCalculator instance for scoring steps.
            stock_list: StockList instance for checking buyability.
            max_depth: Maximum number of retrosynthetic steps allowed.
        """
        self.policy = policy
        self.reward_calculator = reward_calculator
        self.stock_list = stock_list
        self.max_depth = max_depth

        # Internal state -- initialized properly in reset()
        self._molecules: List[str] = []
        self._depths: List[int] = []
        self._in_stock: List[bool] = []
        self._route_tree: Dict[str, Any] = {}
        self._done: bool = True
        self._current_step: int = 0

    def reset(self, target_smiles: str) -> Dict[str, Any]:
        """Reset the environment with a new target molecule.

        Args:
            target_smiles: SMILES string of the molecule to synthesize.

        Returns:
            Initial state dict:
            {
                "molecules": List[str],   # [target_smiles]
                "depths": List[int],      # [0]
                "in_stock": List[bool],   # [True/False]
                "route_tree": dict,       # root node
                "done": bool              # True only if target is already buyable
            }
        """
        target_buyable = self.stock_list.is_buyable(target_smiles)

        self._molecules = [target_smiles]
        self._depths = [0]
        self._in_stock = [target_buyable]
        self._done = target_buyable  # done immediately if already in stock
        self._current_step = 0

        self._route_tree = {
            "smiles": target_smiles,
            "in_stock": target_buyable,
            "depth": 0,
            "children": [],
        }

        return self._get_state()

    def step(self, molecule_idx: int = 0) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Take one retrosynthetic step on a molecule in the current set.

        Selects the molecule at molecule_idx from the unresolved molecules,
        runs the policy to predict reactants, scores the prediction, and
        updates the state.

        Args:
            molecule_idx: Index into the list of *unresolved* molecules
                         (default: 0, the first unresolved molecule).

        Returns:
            Tuple of (state, reward, done, info):
            - state: Updated state dict.
            - reward: Float reward for this step.
            - done: True if episode is finished.
            - info: Dict with per-step details (reward breakdown, etc.).
        """
        # If already done, return current state with zero reward
        if self._done:
            return (
                self._get_state(),
                0.0,
                True,
                {"error": "Episode already done"},
            )

        # Gather indices of unresolved molecules
        unresolved_indices = [
            i for i, in_stock in enumerate(self._in_stock) if not in_stock
        ]

        # Handle molecule_idx out of range
        if not unresolved_indices or molecule_idx < 0 or molecule_idx >= len(unresolved_indices):
            self._done = True
            return (
                self._get_state(),
                0.0,
                True,
                {"error": "molecule_idx out of range for unresolved molecules"},
            )

        # Pick the target molecule
        actual_idx = unresolved_indices[molecule_idx]
        molecule = self._molecules[actual_idx]
        molecule_depth = self._depths[actual_idx]

        # Check if we've hit max depth
        if molecule_depth >= self.max_depth:
            self._done = True
            return (
                self._get_state(),
                0.0,
                True,
                {"error": "Max depth reached", "decomposed": molecule, "reactants": []},
            )

        # Call policy to predict reactants
        try:
            predictions = self.policy.predict(molecule, num_candidates=1, temperature=1.0)
        except Exception:
            predictions = []

        # Handle empty or invalid predictions
        if not predictions or not predictions[0] or not predictions[0].strip():
            self._done = True
            return (
                self._get_state(),
                0.0,
                True,
                {
                    "error": "No valid prediction from policy",
                    "decomposed": molecule,
                    "reactants": [],
                },
            )

        # Split the first prediction on '.' to get individual reactants
        raw_prediction = predictions[0].strip()
        reactants = [r.strip() for r in raw_prediction.split(".") if r.strip()]

        if not reactants:
            self._done = True
            return (
                self._get_state(),
                0.0,
                True,
                {
                    "error": "Empty reactants after splitting",
                    "decomposed": molecule,
                    "reactants": [],
                },
            )

        # Compute reward
        reward = self.reward_calculator.combined_reward(
            molecule, reactants, self.stock_list
        )

        # Build reward breakdown for info dict
        reward_breakdown = self._compute_reward_breakdown(molecule, reactants)

        # Update state: remove the decomposed molecule, add reactants
        self._molecules.pop(actual_idx)
        self._depths.pop(actual_idx)
        self._in_stock.pop(actual_idx)

        # Insert reactants at the position where the decomposed molecule was
        reactant_depth = molecule_depth + 1
        for i, reactant in enumerate(reactants):
            buyable = self.stock_list.is_buyable(reactant)
            self._molecules.insert(actual_idx + i, reactant)
            self._depths.insert(actual_idx + i, reactant_depth)
            self._in_stock.insert(actual_idx + i, buyable)

        # Update route tree: find the node for the decomposed molecule and add children
        self._update_route_tree(self._route_tree, molecule, molecule_depth, reactants)

        self._current_step += 1

        # Check done conditions
        all_resolved = all(self._in_stock)
        max_depth_hit = any(
            d >= self.max_depth
            for d, s in zip(self._depths, self._in_stock)
            if not s
        )

        if all_resolved or max_depth_hit:
            self._done = True

        info = {
            "reward_breakdown": reward_breakdown,
            "decomposed": molecule,
            "reactants": reactants,
        }

        return (self._get_state(), reward, self._done, info)

    def get_route(self) -> Dict:
        """Return the synthesis route tree built so far.

        Returns:
            Route tree dict:
            {
                "smiles": str,
                "in_stock": bool,
                "depth": int,
                "children": [...]  # recursive
            }
        """
        return self._route_tree

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> Dict[str, Any]:
        """Build the current state dictionary."""
        return {
            "molecules": list(self._molecules),
            "depths": list(self._depths),
            "in_stock": list(self._in_stock),
            "route_tree": self._route_tree,
            "done": self._done,
        }

    def _update_route_tree(
        self,
        node: Dict[str, Any],
        target_smiles: str,
        target_depth: int,
        reactants: List[str],
    ) -> bool:
        """Recursively find the node matching *target_smiles* at *target_depth*
        that has no children yet, and attach reactant children to it.

        Returns True if the node was found and updated, False otherwise.
        """
        if (
            node["smiles"] == target_smiles
            and node["depth"] == target_depth
            and not node["children"]
        ):
            child_depth = target_depth + 1
            for reactant in reactants:
                buyable = self.stock_list.is_buyable(reactant)
                node["children"].append(
                    {
                        "smiles": reactant,
                        "in_stock": buyable,
                        "depth": child_depth,
                        "children": [],
                    }
                )
            return True

        # Search children depth-first
        for child in node["children"]:
            if self._update_route_tree(child, target_smiles, target_depth, reactants):
                return True

        return False

    def _compute_reward_breakdown(
        self, product: str, reactants: List[str]
    ) -> Dict[str, float]:
        """Compute individual reward components for the info dict."""
        rc = self.reward_calculator

        validity_scores = [rc.validity_reward(r) for r in reactants]
        plausibility_scores = [rc.plausibility_reward(r) for r in reactants]
        stock_scores = [rc.stock_reward(r, self.stock_list) for r in reactants]

        return {
            "validity": sum(validity_scores) / len(validity_scores),
            "plausibility": sum(plausibility_scores) / len(plausibility_scores),
            "sascore": rc.sascore_reward(product, reactants),
            "stock": sum(stock_scores) / len(stock_scores),
            "atom_conservation": rc.atom_conservation_reward(product, reactants),
            "combined": rc.combined_reward(product, reactants, self.stock_list),
        }
