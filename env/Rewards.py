"""Multi-component reward function for scoring retrosynthetic predictions.

Combines validity, plausibility, synthetic accessibility, and stock match
into a single scalar reward in [0, 1].
"""

import math
from collections import Counter
from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# Default reward component weights
DEFAULT_WEIGHTS: Dict[str, float] = {
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

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
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

    def sascore_reward(self, product_smiles: str, reactant_smiles_list: List[str]) -> float:
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
        """Check if a molecule is in the buyables stock list.

        For a single SMILES, returns 1.0 if buyable, 0.0 if not.
        Also handles lists of reactants by accepting a single SMILES string
        (the combined_reward method handles averaging across reactants).

        Args:
            smiles: SMILES string to check.
            stock_list: StockList instance with is_buyable() method.

        Returns:
            1.0 if buyable, 0.0 if not.
        """
        if not smiles or not isinstance(smiles, str):
            return 0.0
        try:
            return 1.0 if stock_list.is_buyable(smiles) else 0.0
        except Exception:
            return 0.0

    def atom_conservation_reward(self, product_smiles: str, reactant_smiles_list: List[str]) -> float:
        """Check that atoms are approximately conserved in the reaction.

        Reactants should contain at least the atoms present in the product
        (they may contain extra atoms from reagents/catalysts).

        Args:
            product_smiles: The target product SMILES.
            reactant_smiles_list: List of predicted reactant SMILES.

        Returns:
            Float in [0, 1]. 1.0 if atoms are conserved, lower if atoms
            are missing or dramatically wrong.
        """
        if not product_smiles or not reactant_smiles_list:
            return 0.0

        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return 0.0

            # Count atoms in product (by atomic number)
            product_atoms: Counter = Counter()
            for atom in product_mol.GetAtoms():
                product_atoms[atom.GetAtomicNum()] += 1

            # Count atoms in combined reactants
            reactant_atoms: Counter = Counter()
            for r_smi in reactant_smiles_list:
                r_mol = Chem.MolFromSmiles(r_smi)
                if r_mol is None:
                    return 0.0
                for atom in r_mol.GetAtoms():
                    reactant_atoms[atom.GetAtomicNum()] += 1

            # For each atom type in the product, check how well it is covered
            # by the reactants. Reactants may have extra atoms (leaving groups,
            # reagent fragments) which is fine. Missing atoms are penalized.
            total_product_atoms = sum(product_atoms.values())
            if total_product_atoms == 0:
                return 0.0

            covered_atoms = 0
            for atomic_num, count in product_atoms.items():
                reactant_count = reactant_atoms.get(atomic_num, 0)
                # Credit = min(what we need, what we have)
                covered_atoms += min(count, reactant_count)

            # Fraction of product atoms accounted for in reactants
            conservation_ratio = covered_atoms / total_product_atoms

            return max(0.0, min(1.0, conservation_ratio))

        except Exception:
            return 0.0

    def combined_reward(
        self,
        product_smiles: str,
        reactant_smiles_list: List[str],
        stock_list,
        weights: Optional[Dict[str, float]] = None,
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
    def compute_sascore(smiles: str) -> Optional[float]:
        """Compute the Synthetic Accessibility score for a single molecule.

        Uses a proxy based on RDKit molecular descriptors:
        - Lipophilicity (MolLogP)
        - Ring count (CalcNumRings)
        - Rotatable bonds (CalcNumRotatableBonds)
        - Heavy atom count (CalcNumHeavyAtoms)

        These are combined into a score where simpler molecules receive
        lower scores, roughly matching the 1-10 range of the original
        Ertl & Schuffenhauer SA score.

        Args:
            smiles: SMILES string.

        Returns:
            SAscore float (typically 1-10, lower = easier to synthesize).
            Returns None if SMILES is invalid.
        """
        if not smiles or not isinstance(smiles, str):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Compute descriptors
            logp = abs(Descriptors.MolLogP(mol))
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            num_heavy_atoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)

            # Size complexity: larger molecules are harder.
            # Map heavy atom count into a contribution.  Typical drug-like
            # molecules have 15-30 heavy atoms.
            size_score = math.log(max(num_heavy_atoms, 1) + 1.0)  # ~1.1 to ~3.5

            # Ring complexity: more rings (especially fused) = harder
            ring_score = num_rings * 0.5  # 0 to ~3

            # Flexibility: many rotatable bonds can make synthesis easier
            # (linear chains) but also indicates a larger molecule.
            flex_score = num_rot_bonds * 0.1  # 0 to ~1.5

            # Polarity/lipophilicity: extreme logP values indicate
            # harder-to-handle compounds.
            lipo_score = logp * 0.2  # 0 to ~2

            # Combine into a raw score
            raw_score = 1.0 + size_score + ring_score + flex_score + lipo_score

            # Clamp to typical SA score range [1, 10]
            sa_score = max(1.0, min(10.0, raw_score))

            return sa_score

        except Exception:
            return None
