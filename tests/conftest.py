"""Shared test fixtures.

Provides a MockPolicy that mimics RetroPolicy's interface without
downloading the actual model, plus pre-loaded StockList and RewardCalculator.
"""

import sys
from pathlib import Path
from typing import List

import pytest
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.stock.loader import StockList
from env.Rewards import RewardCalculator


# ---------------------------------------------------------------------------
# Mock policy — no model download, deterministic outputs
# ---------------------------------------------------------------------------

class MockPolicy:
    """Lightweight stand-in for RetroPolicy used in unit tests.

    Returns hard-coded reactant SMILES so tests are fast and deterministic.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model_name = "mock"
        self._call_count = 0

    # Mapping of known products to plausible reactant strings
    _RESPONSES = {
        # Aspirin -> salicylic acid + acetic anhydride
        "CC(=O)Oc1ccccc1C(=O)O": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
        # Acetaminophen -> p-aminophenol + acetic anhydride
        "CC(=O)Nc1ccc(O)cc1": "Nc1ccc(O)cc1.CC(=O)OC(C)=O",
    }
    _DEFAULT_RESPONSE = "CCO.CC(=O)O"

    def predict(self, product_smiles: str, num_candidates: int = 5,
                temperature: float = 1.0) -> List[str]:
        resp = self._RESPONSES.get(product_smiles, self._DEFAULT_RESPONSE)
        self._call_count += 1
        return [resp] * num_candidates

    def predict_greedy(self, product_smiles: str, num_beams: int = 5) -> List[str]:
        resp = self._RESPONSES.get(product_smiles, self._DEFAULT_RESPONSE)
        self._call_count += 1
        return [resp] * num_beams

    def log_prob(self, product_smiles: str, reactant_smiles: str) -> torch.Tensor:
        return torch.tensor(-2.0, requires_grad=True)

    def get_model(self):
        """Return a tiny nn.Module so optimizer can be created."""
        return torch.nn.Linear(1, 1)

    def save_checkpoint(self, path, step, reward, optimizer=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": step, "reward": reward, "model_state_dict": {}}, path)

    def load_checkpoint(self, path):
        data = torch.load(path, map_location="cpu")
        return {"step": data.get("step", 0), "reward": data.get("reward", 0.0),
                "has_optimizer": False}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_policy():
    return MockPolicy()


@pytest.fixture
def stock_list():
    sl = StockList()
    sl.load()
    return sl


@pytest.fixture
def reward_calc():
    return RewardCalculator()
