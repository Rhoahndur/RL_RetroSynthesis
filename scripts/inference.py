"""Inference script — load checkpoint, run MCTS, return structured results.

Used by the Streamlit app to perform retrosynthetic search on a target molecule.

Usage:
    # As a module (from Streamlit):
    from scripts.inference import load_model, run_inference
    policy = load_model("models/checkpoints/best.pt")
    result = run_inference("CC(C)Cc1ccc(cc1)C(C)C(=O)O", policy, reward_calc, stock)

    # As a CLI:
    python scripts/inference.py --target "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
"""

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_model(checkpoint_path: Optional[str] = None, device: Optional[str] = None):
    """Load a RetroPolicy from a checkpoint or use pre-trained weights.

    Args:
        checkpoint_path: Path to .pt checkpoint. If None, uses pre-trained model.
        device: Target device. Auto-detects if None.

    Returns:
        RetroPolicy instance ready for inference.
    """
    raise NotImplementedError


def mol_to_base64_image(smiles: str, size: tuple = (300, 300)) -> Optional[str]:
    """Render a molecule SMILES as a PNG image encoded in base64.

    Args:
        smiles: SMILES string to render.
        size: Image dimensions (width, height).

    Returns:
        Base64-encoded PNG string, or None if SMILES is invalid.
    """
    raise NotImplementedError


def run_inference(
    target_smiles: str,
    policy,
    reward_calc,
    stock_list,
    max_simulations: int = 500,
    time_budget: float = 60.0,
) -> Dict:
    """Run MCTS retrosynthetic search and return structured results.

    Args:
        target_smiles: SMILES string of the target molecule.
        policy: RetroPolicy instance (pre-loaded).
        reward_calc: RewardCalculator instance.
        stock_list: StockList instance (pre-loaded).
        max_simulations: Max MCTS iterations.
        time_budget: Max search time in seconds.

    Returns:
        Dict with structure:
        {
            "target": str,
            "routes": [
                {
                    "smiles": str,
                    "score": float,
                    "in_stock": bool,
                    "children": [...]
                }
            ],
            "best_score": float,
            "stats": {
                "simulations": int,
                "time_seconds": float,
                "routes_found": int
            },
            "molecules": [
                {
                    "smiles": str,
                    "sascore": float,
                    "in_stock": bool,
                    "image_b64": str
                }
            ]
        }
    """
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrosynthesis inference")
    parser.add_argument("--target", type=str, required=True, help="Target SMILES")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--max_simulations", type=int, default=500)
    parser.add_argument("--time_budget", type=float, default=60.0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Will be implemented to load model, run inference, print results
    raise NotImplementedError
