"""Multi-component reward function for scoring retrosynthetic predictions.

Combines validity, plausibility, synthetic accessibility, and stock match
into a single scalar reward in [0, 1].
"""

import math
from collections import Counter

from rdkit import Chem

# Soft stock matching: partial credit for Tanimoto similarity above this threshold
STOCK_SIMILARITY_THRESHOLD = 0.6
STOCK_SIMILARITY_SCALE = 1.0 - STOCK_SIMILARITY_THRESHOLD  # 0.4

# Common reaction byproducts, sorted by total atoms descending for greedy matching
COMMON_BYPRODUCTS = [
    ("AcOH", Counter({6: 2, 1: 4, 8: 2})),  # acetic acid CH3COOH - 8 atoms
    ("EtOH", Counter({6: 2, 1: 6, 8: 1})),  # ethanol - 9 atoms
    ("MeOH", Counter({6: 1, 1: 4, 8: 1})),  # methanol - 6 atoms
    ("SO2", Counter({16: 1, 8: 2})),  # sulfur dioxide - 3 atoms
    ("CO2", Counter({6: 1, 8: 2})),  # carbon dioxide - 3 atoms
    ("H2O", Counter({1: 2, 8: 1})),  # water - 3 atoms
    ("NH3", Counter({7: 1, 1: 3})),  # ammonia - 4 atoms
    ("HCl", Counter({1: 1, 17: 1})),  # hydrochloric acid
    ("HBr", Counter({1: 1, 35: 1})),  # hydrobromic acid
    ("HI", Counter({1: 1, 53: 1})),  # hydroiodic acid
    ("HF", Counter({1: 1, 9: 1})),  # hydrofluoric acid
    ("N2", Counter({7: 2})),  # nitrogen gas
]

# Default reward component weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "validity": 0.3,
    "plausibility": 0.2,
    "sascore": 0.2,
    "stock": 0.3,
}


class RewardCalculator:
    """Computes multi-objective rewards for retrosynthetic predictions.

    Usage:
        from data.stock.loader import StockList
        stock = StockList()
        stock.load()
        calc = RewardCalculator(weights=DEFAULT_WEIGHTS)
        reward = calc.combined_reward(product_smi, reactant_smi_list, stock)
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialize with optional custom reward weights.

        Args:
            weights: Dict mapping component names to float weights.
                     Keys: "validity", "plausibility", "sascore", "stock".
                     Defaults to DEFAULT_WEIGHTS if None.
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def validity_reward(self, smiles: str) -> float:
        """Check if a SMILES string parses into a valid RDKit molecule.

        Args:
            smiles: SMILES string to validate.

        Returns:
            1.0 if valid, 0.0 if invalid or empty.
        """
        if not smiles or not isinstance(smiles, str):
            return 0.0
        try:
            mol = Chem.MolFromSmiles(smiles)
            return 1.0 if mol is not None else 0.0
        except Exception:
            return 0.0

    def plausibility_reward(self, smiles: str) -> float:
        """Check if a SMILES string passes RDKit sanitization (valency checks).

        Args:
            smiles: SMILES string to check.

        Returns:
            1.0 if plausible, 0.0 if fails sanitization.
        """
        if not smiles or not isinstance(smiles, str):
            return 0.0
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return 0.0
            Chem.SanitizeMol(mol)
            return 1.0
        except Exception:
            return 0.0

    def sascore_reward(self, product_smiles: str, reactant_smiles_list: list[str]) -> float:
        """Score based on synthetic accessibility improvement.

        Reactants should be simpler (lower SAscore) than the product.
        Uses RDKit descriptors to estimate synthetic accessibility.

        Args:
            product_smiles: The target product SMILES.
            reactant_smiles_list: List of predicted reactant SMILES.

        Returns:
            Float in [0, 1]. Higher means reactants are meaningfully simpler.
            Returns 0.0 if any SMILES are invalid.
        """
        if not product_smiles or not reactant_smiles_list:
            return 0.0

        product_sa = self.compute_sascore(product_smiles)
        if product_sa is None:
            return 0.0

        reactant_scores = []
        for r_smi in reactant_smiles_list:
            r_sa = self.compute_sascore(r_smi)
            if r_sa is None:
                return 0.0
            reactant_scores.append(r_sa)

        # Average SA score of reactants. Lower SA = easier to synthesize.
        avg_reactant_sa = sum(reactant_scores) / len(reactant_scores)

        # Improvement ratio: how much simpler are reactants vs product.
        # product_sa - avg_reactant_sa > 0 means reactants are simpler (lower SA).
        # Normalize using a sigmoid-like mapping to [0, 1].
        # A difference of 0 maps to 0.5; positive differences (reactants simpler)
        # map above 0.5; negative differences map below 0.5.
        improvement = product_sa - avg_reactant_sa

        # Use a sigmoid centered at 0 with a scaling factor.
        # A difference of ~2 SA-score points should give a high reward (~0.88).
        reward = 1.0 / (1.0 + math.exp(-improvement))

        return max(0.0, min(1.0, reward))

    def stock_reward(self, smiles: str, stock_list) -> float:
        """Score a molecule based on stock availability or similarity to buyables.

        Returns 1.0 for exact matches. For non-matches, computes Morgan
        fingerprint Tanimoto similarity to the nearest buyable molecule and
        gives partial credit for similarity above 0.6 (linear scale to 1.0).

        Args:
            smiles: SMILES string to check.
            stock_list: StockList instance with is_buyable() and
                        nearest_similarity() methods.

        Returns:
            Float in [0, 1]. 1.0 if buyable, partial credit if similar.
        """
        if not smiles or not isinstance(smiles, str):
            return 0.0
        try:
            if stock_list.is_buyable(smiles):
                return 1.0
            # Soft reward: partial credit for molecules close to buyables
            if hasattr(stock_list, "nearest_similarity"):
                similarity = stock_list.nearest_similarity(smiles)
                if similarity > STOCK_SIMILARITY_THRESHOLD:
                    return (similarity - STOCK_SIMILARITY_THRESHOLD) / STOCK_SIMILARITY_SCALE
            return 0.0
        except Exception:
            return 0.0

    def atom_conservation_reward(
        self, product_smiles: str, reactant_smiles_list: list[str]
    ) -> float:
        """Bidirectional atom conservation check with byproduct awareness.

        Checks both directions:
        1. Coverage: reactant atoms must cover product atoms.
        2. Excess penalty: unexplained excess reactant atoms are penalized,
           but common byproducts (H2O, CO2, AcOH, etc.) are forgiven.

        Args:
            product_smiles: The target product SMILES.
            reactant_smiles_list: List of predicted reactant SMILES.

        Returns:
            Float in [0, 1]. coverage * excess_penalty, clamped to [0, 1].
            Returns 0.0 for invalid inputs.
        """
        if not product_smiles or not reactant_smiles_list:
            return 0.0

        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return 0.0
            product_mol = Chem.AddHs(product_mol)

            # Count product atoms by element (atomic number), including H
            product_counts: Counter = Counter()
            for atom in product_mol.GetAtoms():
                product_counts[atom.GetAtomicNum()] += 1

            # Count combined reactant atoms, including H
            reactant_counts: Counter = Counter()
            for r_smi in reactant_smiles_list:
                r_mol = Chem.MolFromSmiles(r_smi)
                if r_mol is None:
                    return 0.0
                r_mol = Chem.AddHs(r_mol)
                for atom in r_mol.GetAtoms():
                    reactant_counts[atom.GetAtomicNum()] += 1

            total_product_atoms = sum(product_counts.values())
            if total_product_atoms == 0:
                return 0.0

            # Coverage — fraction of product atoms present in reactants
            covered = sum(min(product_counts[z], reactant_counts.get(z, 0)) for z in product_counts)
            coverage = covered / total_product_atoms

            # Compute excess per element
            excess: Counter = Counter()
            for z in reactant_counts:
                diff = reactant_counts[z] - product_counts.get(z, 0)
                if diff > 0:
                    excess[z] = diff

            # Greedily subtract common byproducts from excess
            for _name, formula in COMMON_BYPRODUCTS:
                while True:
                    if all(excess.get(z, 0) >= cnt for z, cnt in formula.items()):
                        for z, cnt in formula.items():
                            excess[z] -= cnt
                            if excess[z] <= 0:
                                del excess[z]
                    else:
                        break

            # Remaining unexplained excess
            remaining_excess = sum(excess.values())
            total_reactant_atoms = sum(reactant_counts.values())

            # Excess penalty
            if total_reactant_atoms > 0:
                excess_penalty = 1.0 - (remaining_excess / total_reactant_atoms)
                excess_penalty = max(0.0, min(1.0, excess_penalty))
            else:
                excess_penalty = 1.0

            return max(0.0, min(1.0, coverage * excess_penalty))

        except Exception:
            return 0.0

    def combined_reward(
        self,
        product_smiles: str,
        reactant_smiles_list: list[str],
        stock_list,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute the weighted combination of all reward components.

        Args:
            product_smiles: The target product SMILES.
            reactant_smiles_list: List of predicted reactant SMILES strings.
            stock_list: StockList instance for buyability checking.
            weights: Optional override weights. Uses self.weights if None.

        Returns:
            Float in [0, 1]. Weighted sum of all components.
        """
        if not product_smiles or not reactant_smiles_list:
            return 0.0

        w = weights or self.weights

        # Validity: average across all reactants
        validity_scores = [self.validity_reward(r) for r in reactant_smiles_list]
        avg_validity = sum(validity_scores) / len(validity_scores)

        # Plausibility: average across all reactants
        plausibility_scores = [self.plausibility_reward(r) for r in reactant_smiles_list]
        avg_plausibility = sum(plausibility_scores) / len(plausibility_scores)

        # SA score improvement
        sa_reward = self.sascore_reward(product_smiles, reactant_smiles_list)

        # Stock: average across all reactants
        stock_scores = [self.stock_reward(r, stock_list) for r in reactant_smiles_list]
        avg_stock = sum(stock_scores) / len(stock_scores)

        # Atom conservation (already considers all reactants together)
        atom_reward = self.atom_conservation_reward(product_smiles, reactant_smiles_list)

        # Weighted sum. Atom conservation acts as a multiplier rather than
        # an additive component -- a reaction that doesn't conserve atoms
        # should be penalized heavily regardless of other scores.
        weighted_sum = (
            w.get("validity", 0.0) * avg_validity
            + w.get("plausibility", 0.0) * avg_plausibility
            + w.get("sascore", 0.0) * sa_reward
            + w.get("stock", 0.0) * avg_stock
        )

        # Normalize by total weight so result stays in [0, 1] even if
        # weights don't sum to 1.
        total_weight = sum(w.get(k, 0.0) for k in ("validity", "plausibility", "sascore", "stock"))
        if total_weight > 0:
            weighted_sum = weighted_sum / total_weight

        # Apply atom conservation as a soft gate: poor conservation
        # drags down the overall reward.
        final_reward = weighted_sum * (0.5 + 0.5 * atom_reward)

        return max(0.0, min(1.0, final_reward))

    @staticmethod
    def compute_sascore(smiles: str) -> float | None:
        """Compute the Synthetic Accessibility score for a single molecule.

        Uses the Ertl & Schuffenhauer algorithm (J. Cheminformatics 1:8, 2009)
        based on molecular fragment contributions and complexity penalties.

        Args:
            smiles: SMILES string.

        Returns:
            SAscore float (1-10, lower = easier to synthesize).
            Returns None if SMILES is invalid.
        """
        if not smiles or not isinstance(smiles, str):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            from lib.sascorer import calculateScore

            return calculateScore(mol)
        except Exception:
            return None
