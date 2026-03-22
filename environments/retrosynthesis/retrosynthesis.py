"""Verifiers environment for retrosynthetic route prediction.

Teaches an LLM to predict reactant molecules (as SMILES) that can be combined
to synthesize a given target molecule. Self-contained -- all reward logic is
inline so the package can be pushed to Prime Intellect's hub independently.
"""

import asyncio
import json
import math
import re
from collections import Counter
from typing import Optional

import verifiers as vf
from datasets import Dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a retrosynthesis expert. Given a target molecule as a SMILES string, predict the reactant molecules that can be combined to synthesize the target.

Rules:
- Output ONLY the reactant SMILES strings separated by '.'
- Do NOT include any explanation, reasoning, or extra text
- Each reactant must be a valid SMILES string
- Prefer simpler, commercially available starting materials
- Ensure atom conservation: reactant atoms should cover the product atoms

Example:
Input: CC(=O)Oc1ccccc1C(=O)O
Output: OC(=O)c1ccccc1O.CC(=O)OC(C)=O"""

# ---------------------------------------------------------------------------
# ~200 buyable SMILES (canonical) for stock checking
# ---------------------------------------------------------------------------

BUYABLE_SMILES = {
    "O",
    "CO",
    "CCO",
    "CCCO",
    "CC(C)O",
    "CC(=O)O",
    "CC=O",
    "C=O",
    "CC(=O)OC(C)=O",
    "ClC(Cl)Cl",
    "ClCCl",
    "CS(C)=O",
    "c1ccncc1",
    "c1ccoc1",
    "c1ccsc1",
    "c1ccccc1",
    "Cc1ccccc1",
    "Oc1ccccc1",
    "Nc1ccccc1",
    "Clc1ccccc1",
    "Brc1ccccc1",
    "O=[N+]([O-])c1ccccc1",
    "N",
    "CN",
    "CCN",
    "CC(C)N",
    "C(=O)N",
    "NCC(=O)O",
    "Nc1ccc(O)cc1",
    "OC(=O)c1ccccc1O",
    "CC(C)Cc1ccccc1",
    "CC(=O)Cl",
    "CCC(=O)Cl",
    "O=C(Cl)c1ccccc1",
    "CI",
    "CBr",
    "CCl",
    "CCBr",
    "CCCl",
    "CCI",
    "C=C",
    "C=Cc1ccccc1",
    "C#C",
    "C#Cc1ccccc1",
    "O=CO",
    "OC(=O)c1ccccc1",
    "O=C(O)CC(=O)O",
    "[Na+].[OH-]",
    "[K+].[OH-]",
    "[Na+].[Cl-]",
    "O=S(=O)(O)O",
    "O=C(O)/C=C\\C(=O)O",
    "O=C(O)/C=C/C(=O)O",
    "CC(=O)OCC",
    "CCOC(=O)CC(=O)OCC",
    "CCOC(C)=O",
    "CC(C)=O",
    "O=C(c1ccccc1)c1ccccc1",
    "O=C1CCCCC1",
    "C1CCOC1",
    "C1COCCO1",
    "C1CCNCC1",
    "C1CNCCN1",
    "CCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCC=CCCCCCCCC",
    "B(O)O",
    "OB(O)c1ccccc1",
    "C1=CC=C(B(O)O)C=C1",
    "[Al+3].[Cl-].[Cl-].[Cl-]",
    "c1ccc2[nH]ccc2c1",
    "c1ccc2ccccc2c1",
    "CC(=O)c1ccc(CC(C)C)cc1",
    "O=C1OC(=O)c2ccccc21",
}

# ---------------------------------------------------------------------------
# Demo molecules with known retrosynthetic answers
# ---------------------------------------------------------------------------

DEMO_MOLECULES = [
    {
        "product": "CC(=O)Oc1ccccc1C(=O)O",
        "name": "Aspirin",
        "reactants": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
    },
    {
        "product": "CC(=O)Nc1ccc(O)cc1",
        "name": "Acetaminophen",
        "reactants": "Nc1ccc(O)cc1.CC(=O)OC(C)=O",
    },
    {
        "product": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "name": "Ibuprofen",
        "reactants": "CC(C)Cc1ccc(C(C)C(=O)OCC)cc1.O",
    },
    {
        "product": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "name": "Caffeine",
        "reactants": "Cn1c(=O)c2[nH]cnc2n(C)c1=O.CI",
    },
]

# ---------------------------------------------------------------------------
# Additional common drug molecules for training
# ---------------------------------------------------------------------------

TRAINING_MOLECULES = [
    {"product": "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O", "name": "Testosterone"},
    {"product": "OC(=O)CC(O)(CC(=O)O)C(=O)O", "name": "Citric acid"},
    {"product": "CC(=O)OC1=CC2=C(S1)CCN2C", "name": "Clopidogrel fragment"},
    {"product": "c1ccc(-c2ccccc2)cc1", "name": "Biphenyl"},
    {"product": "CC(=O)OC(CC(=O)[O-])C(=O)[O-]", "name": "Calcium citrate fragment"},
    {"product": "OC(=O)c1cc(O)c(O)c(O)c1", "name": "Gallic acid"},
    {"product": "NC(=O)c1ccc(O)cc1", "name": "4-Hydroxybenzamide"},
    {"product": "CC(O)c1ccccc1", "name": "1-Phenylethanol"},
    {"product": "O=Cc1ccc(O)c(OC)c1", "name": "Vanillin"},
    {"product": "OC(c1ccccc1)c1ccccc1", "name": "Benzhydrol"},
    {"product": "CC(=O)c1ccccc1", "name": "Acetophenone"},
    {"product": "OC(=O)/C=C/c1ccccc1", "name": "Cinnamic acid"},
    {"product": "O=C(O)c1ccccc1", "name": "Benzoic acid"},
    {"product": "COc1ccc(C=O)cc1", "name": "Anisaldehyde"},
    {"product": "CC(C)CC(=O)O", "name": "Isovaleric acid"},
    {"product": "CCCC(=O)O", "name": "Butyric acid"},
    {"product": "c1ccc(OCc2ccccc2)cc1", "name": "Benzyl phenyl ether"},
    {"product": "CC(=O)c1ccc(O)cc1", "name": "4-Hydroxyacetophenone"},
    {"product": "Oc1ccc(Cl)cc1", "name": "4-Chlorophenol"},
    {"product": "Nc1ccc(Cl)cc1", "name": "4-Chloroaniline"},
]

# ---------------------------------------------------------------------------
# SMILES format regex -- matches strings that look like valid SMILES
# ---------------------------------------------------------------------------

_SMILES_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)\\/=#$:.%\s]+$")

# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _parse_reactants(completion) -> list[str]:
    """Extract reactant SMILES strings from a completion.

    Args:
        completion: list of message dicts, e.g.
            [{"role": "assistant", "content": "CCO.CC=O"}]

    Returns:
        List of stripped SMILES fragments. Empty list on failure.
    """
    try:
        content = completion[-1]["content"].strip()
        parts = content.split(".")
        return [p.strip() for p in parts if p.strip()]
    except Exception:
        return []


def _canonicalize(smi: str) -> Optional[str]:
    """Canonicalize a SMILES string via RDKit.

    Returns None if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _compute_sascore(smiles: str) -> Optional[float]:
    """Compute synthetic accessibility score for a single molecule.

    Uses RDKit descriptors as a proxy (lipophilicity, ring count,
    rotatable bonds, heavy atom count). Returns a value in [1, 10]
    where lower means easier to synthesize.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        logp = abs(Descriptors.MolLogP(mol))
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        num_heavy_atoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)

        size_score = math.log(max(num_heavy_atoms, 1) + 1.0)
        ring_score = num_rings * 0.5
        flex_score = num_rot_bonds * 0.1
        lipo_score = logp * 0.2

        raw_score = 1.0 + size_score + ring_score + flex_score + lipo_score
        return max(1.0, min(10.0, raw_score))
    except Exception:
        return None


def _get_atom_counts(smiles: str) -> Optional[Counter]:
    """Count atoms (by atomic number) in a molecule.

    Returns None if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        counts: Counter = Counter()
        for atom in mol.GetAtoms():
            counts[atom.GetAtomicNum()] += 1
        return counts
    except Exception:
        return None


def _check_validity(content: str) -> float:
    """Synchronous helper: parse reactants from raw content and return
    the fraction that are valid SMILES."""
    parts = content.split(".")
    fragments = [p.strip() for p in parts if p.strip()]
    if not fragments:
        return 0.0
    valid = 0
    for frag in fragments:
        mol = Chem.MolFromSmiles(frag)
        if mol is not None:
            valid += 1
    return valid / len(fragments)


def _check_sascore(content: str, product_smiles: str) -> float:
    """Synchronous helper: compute SA-score reward for a completion."""
    product_sa = _compute_sascore(product_smiles)
    if product_sa is None:
        return 0.0

    parts = content.split(".")
    fragments = [p.strip() for p in parts if p.strip()]
    if not fragments:
        return 0.0

    reactant_scores = []
    for frag in fragments:
        sa = _compute_sascore(frag)
        if sa is not None:
            reactant_scores.append(sa)

    if not reactant_scores:
        return 0.0

    avg_reactant_sa = sum(reactant_scores) / len(reactant_scores)
    improvement = product_sa - avg_reactant_sa
    return 1.0 / (1.0 + math.exp(-improvement))


def _check_stock(content: str, stock_set: set) -> float:
    """Synchronous helper: return fraction of reactants that are buyable."""
    parts = content.split(".")
    fragments = [p.strip() for p in parts if p.strip()]
    if not fragments:
        return 0.0

    buyable = 0
    for frag in fragments:
        canon = _canonicalize(frag)
        if canon is not None and canon in stock_set:
            buyable += 1
    return buyable / len(fragments)


def _check_atom_conservation(content: str, product_smiles: str) -> float:
    """Synchronous helper: return fraction of product atoms covered by
    the combined reactants."""
    product_counts = _get_atom_counts(product_smiles)
    if product_counts is None:
        return 0.0

    total_product_atoms = sum(product_counts.values())
    if total_product_atoms == 0:
        return 0.0

    parts = content.split(".")
    fragments = [p.strip() for p in parts if p.strip()]
    if not fragments:
        return 0.0

    reactant_counts: Counter = Counter()
    for frag in fragments:
        frag_counts = _get_atom_counts(frag)
        if frag_counts is None:
            # Skip invalid fragments rather than returning 0
            continue
        reactant_counts += frag_counts

    if not reactant_counts:
        return 0.0

    covered = 0
    for atomic_num, count in product_counts.items():
        covered += min(count, reactant_counts.get(atomic_num, 0))

    return max(0.0, min(1.0, covered / total_product_atoms))


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_dataset(split: str, difficulty: str) -> Dataset:
    """Build a HuggingFace Dataset for the given split.

    Args:
        split: "train" or "test".
        difficulty: Unused for now (reserved for future filtering).

    Returns:
        Dataset with columns: question, answer, info.
    """
    if split == "test":
        molecules = DEMO_MOLECULES
    else:
        molecules = DEMO_MOLECULES + TRAINING_MOLECULES

    stock_list = sorted(BUYABLE_SMILES)

    rows = []
    for mol in molecules:
        product = mol["product"]
        answer = mol.get("reactants", "")
        info = json.dumps(
            {
                "product_smiles": product,
                "stock_list": stock_list,
            }
        )
        rows.append(
            {
                "question": f"Predict the reactants for: {product}",
                "answer": answer,
                "info": info,
            }
        )

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Rubric (5 async reward functions)
# ---------------------------------------------------------------------------


def build_rubric() -> vf.Rubric:
    """Build the reward rubric with 5 weighted reward functions."""

    # ------------------------------------------------------------------
    # 1. Format reward (weight 0.1)
    # ------------------------------------------------------------------
    async def format_reward(completion, **kwargs) -> float:
        """Check that the output looks like valid SMILES notation
        (no explanatory text, markdown, etc.)."""
        try:
            content = completion[-1]["content"].strip()
            if not content:
                return 0.0
            if _SMILES_RE.match(content):
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # 2. Validity reward (weight 0.25)
    # ------------------------------------------------------------------
    async def validity_reward(completion, **kwargs) -> float:
        """Return the fraction of reactant fragments that parse as valid
        SMILES via RDKit."""
        try:
            content = completion[-1]["content"].strip()
            if not content:
                return 0.0
            result = await asyncio.to_thread(_check_validity, content)
            return result
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # 3. SA-score reward (weight 0.2)
    # ------------------------------------------------------------------
    async def sascore_reward(completion, info, **kwargs) -> float:
        """Reward based on whether reactants are simpler (lower SA score)
        than the target product."""
        try:
            if isinstance(info, str):
                info = json.loads(info)
            product_smiles = info["product_smiles"]
            content = completion[-1]["content"].strip()
            if not content:
                return 0.0
            result = await asyncio.to_thread(_check_sascore, content, product_smiles)
            return result
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # 4. Stock reward (weight 0.3)
    # ------------------------------------------------------------------
    async def stock_reward(completion, info, **kwargs) -> float:
        """Return the fraction of predicted reactants that are
        commercially available (present in the stock list)."""
        try:
            if isinstance(info, str):
                info = json.loads(info)
            stock_list = info["stock_list"]
            stock_set = set(stock_list)
            content = completion[-1]["content"].strip()
            if not content:
                return 0.0
            result = await asyncio.to_thread(_check_stock, content, stock_set)
            return result
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # 5. Atom conservation reward (weight 0.15)
    # ------------------------------------------------------------------
    async def atom_conservation_reward(completion, info, **kwargs) -> float:
        """Return the fraction of product atoms that are accounted for
        in the combined reactants."""
        try:
            if isinstance(info, str):
                info = json.loads(info)
            product_smiles = info["product_smiles"]
            content = completion[-1]["content"].strip()
            if not content:
                return 0.0
            result = await asyncio.to_thread(_check_atom_conservation, content, product_smiles)
            return result
        except Exception:
            return 0.0

    # Build and return the rubric
    return vf.Rubric(
        funcs=[
            format_reward,
            validity_reward,
            sascore_reward,
            stock_reward,
            atom_conservation_reward,
        ],
        weights=[0.1, 0.25, 0.2, 0.3, 0.15],
    )


# ---------------------------------------------------------------------------
# Entry point required by verifiers framework
# ---------------------------------------------------------------------------


def load_environment(
    split: str = "train",
    difficulty: str = "all",
    **kwargs,
) -> vf.Environment:
    """Load the retrosynthesis verifiers environment.

    Args:
        split: "train" or "test".
        difficulty: Reserved for future use.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        A configured vf.SingleTurnEnv ready for training or evaluation.
    """
    dataset = build_dataset(split, difficulty)
    eval_dataset = build_dataset("test", difficulty)
    rubric = build_rubric()

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        system_prompt=SYSTEM_PROMPT,
    )
