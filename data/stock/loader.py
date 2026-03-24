"""Stock list loader for buyable/commercially available molecules.

Provides O(1) lookup to check if a molecule is purchasable.
All SMILES are stored and compared in canonical form via RDKit.

Supported formats:
  - CSV (smiles,name,category with header row)
  - Gzipped SMILES (.smi.gz, one canonical SMILES per line)
"""

import gzip
import os

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

DEFAULT_STOCK_PATH = os.path.join(os.path.dirname(__file__), "buyables.csv")


class StockList:
    """Set of commercially available (buyable) molecules for retrosynthesis termination.

    Usage:
        stock = StockList()
        stock.load()  # or stock.load("/path/to/buyables.csv")
        stock.load("/path/to/buyables_full.smi.gz")  # gzipped SMILES
        stock.is_buyable("CCO")  # True if ethanol is in the list
    """

    def __init__(self) -> None:
        self._canonical_smiles: set[str] = set()
        self._fingerprints: list = []
        self._loaded = False

    @property
    def _smiles(self) -> set[str]:
        """Alias used by load_smi_gz (matches internal set)."""
        return self._canonical_smiles

    def load(self, path: str | None = None) -> "StockList":
        """Load buyable molecules from a file.

        Auto-detects format based on file extension:
          - ``.smi.gz`` -- gzipped SMILES (one per line)
          - anything else -- CSV with ``smiles,name,category`` header

        Args:
            path: Path to stock file.  Defaults to ``data/stock/buyables.csv``.

        Returns:
            self (for chaining)
        """
        if path is None:
            path = DEFAULT_STOCK_PATH

        if path.endswith(".smi.gz"):
            self.load_smi_gz(path)
            return self

        # CSV loading (original behaviour)
        self._canonical_smiles.clear()

        with open(path) as f:
            f.readline()  # skip header row
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # CSV columns: smiles,name,category
                parts = line.split(",", 2)
                if not parts:
                    continue
                raw_smiles = parts[0].strip()
                canon = self.canonicalize(raw_smiles)
                if canon is not None:
                    self._canonical_smiles.add(canon)

        self._loaded = True
        self._precompute_fingerprints()
        return self

    def load_smi_gz(self, path: str = "data/stock/buyables_full.smi.gz") -> None:
        """Load stock list from a gzipped SMILES file (one SMILES per line)."""
        self._canonical_smiles.clear()
        with gzip.open(path, "rt") as f:
            for line in f:
                smi = line.strip()
                if smi:
                    canon = self.canonicalize(smi)
                    if canon:
                        self._canonical_smiles.add(canon)
        self._loaded = True
        self._precompute_fingerprints()
        print(f"Loaded {len(self._canonical_smiles)} buyable molecules from {path}")

    def _precompute_fingerprints(self) -> None:
        """Precompute Morgan fingerprints for similarity lookups."""
        self._fingerprints = []
        for smi in self._canonical_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                self._fingerprints.append(fp)

    def is_buyable(self, smiles: str) -> bool:
        """Check if a molecule is in the stock list.

        Canonicalizes the input SMILES before lookup so that different
        valid SMILES representations of the same molecule all match.

        Args:
            smiles: SMILES string to check.

        Returns:
            True if the molecule is buyable, False otherwise.
            Returns False for invalid SMILES (no crash).
        """
        canon = self.canonicalize(smiles)
        if canon is None:
            return False
        return canon in self._canonical_smiles

    @staticmethod
    def canonicalize(smiles: str) -> str | None:
        """Convert a SMILES string to its canonical RDKit form.

        Args:
            smiles: Input SMILES string.

        Returns:
            Canonical SMILES string, or None if the input is invalid.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    def nearest_similarity(self, smiles: str) -> float:
        """Find the max Tanimoto similarity to any buyable molecule.

        Uses Morgan fingerprints (radius=2, 2048 bits).

        Args:
            smiles: SMILES string to compare.

        Returns:
            Max Tanimoto similarity in [0, 1]. Returns 0.0 for invalid SMILES.
        """
        if not self._fingerprints:
            return 0.0
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            similarities = DataStructs.BulkTanimotoSimilarity(fp, self._fingerprints)
            return max(similarities) if similarities else 0.0
        except Exception:
            return 0.0

    def __len__(self) -> int:
        """Return the number of molecules in the stock list."""
        return len(self._canonical_smiles)

    def __contains__(self, smiles: str) -> bool:
        """Allow `smiles in stock_list` syntax."""
        return self.is_buyable(smiles)
