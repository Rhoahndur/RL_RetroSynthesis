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
        raise NotImplementedError

    def reset(self, target_smiles: str) -> Dict[str, Any]:
        """Reset the environment with a new target molecule.

        Args:
            target_smiles: SMILES string of the molecule to synthesize.

        Returns:
            Initial state dict:
            {
                "molecules": List[str],   # [target_smiles]
                "depths": List[int],      # [0]
                "in_stock": List[bool],   # [False] (assuming target isn't buyable)
                "route_tree": dict,       # root node
                "done": bool              # False
            }
        """
        raise NotImplementedError

    def step(self, molecule_idx: int = 0) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Take one retrosynthetic step on a molecule in the current set.

        Selects the molecule at molecule_idx, runs the policy to predict
        reactants, scores the prediction, and updates the state.

        Args:
            molecule_idx: Index of the molecule to decompose (default: 0,
                         the first unresolved molecule).

        Returns:
            Tuple of (state, reward, done, info):
            - state: Updated state dict.
            - reward: Float reward for this step.
            - done: True if episode is finished.
            - info: Dict with per-step details (reward breakdown, etc.).
        """
        raise NotImplementedError

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
        raise NotImplementedError
