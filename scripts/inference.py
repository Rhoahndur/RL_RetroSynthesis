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
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.stock.loader import StockList
from env.MCTS import MCTS
from env.Rewards import RewardCalculator
from models.policy import RetroPolicy


def load_model(checkpoint_path: str | None = None, device: str | None = None) -> RetroPolicy:
    """Load a RetroPolicy from a checkpoint or use pre-trained weights.

    Args:
        checkpoint_path: Path to .pt checkpoint. If None, uses pre-trained model.
        device: Target device. Auto-detects if None.

    Returns:
        RetroPolicy instance ready for inference.
    """
    policy = RetroPolicy(device=device)

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        meta = policy.load_checkpoint(checkpoint_path)
        print(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(step={meta['step']}, reward={meta['reward']:.4f})"
        )
    else:
        if checkpoint_path is not None:
            print(
                f"Checkpoint not found at {checkpoint_path}, "
                "using pre-trained ReactionT5v2 weights."
            )
        else:
            print("No checkpoint specified, using pre-trained ReactionT5v2 weights.")

    return policy


def mol_to_base64_image(smiles: str, size: tuple = (300, 300)) -> str | None:
    """Render a molecule SMILES as a PNG image encoded in base64.

    Args:
        smiles: SMILES string to render.
        size: Image dimensions (width, height).

    Returns:
        Base64-encoded PNG string, or None if SMILES is invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None


def flatten_route_molecules(route_dict: dict) -> list[str]:
    """Recursively traverse a route tree dict and collect all unique SMILES.

    Args:
        route_dict: Route tree dict with "smiles" and "children" keys.

    Returns:
        Deduplicated list of SMILES strings found in the tree.
    """
    seen = set()
    result = []

    def _traverse(node: dict) -> None:
        smi = node.get("smiles")
        if smi and smi not in seen:
            seen.add(smi)
            result.append(smi)
        for child in node.get("children", []):
            _traverse(child)

    _traverse(route_dict)
    return result


def run_inference(
    target_smiles: str,
    policy,
    reward_calc,
    stock_list,
    max_simulations: int = 500,
    time_budget: float = 60.0,
) -> dict:
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
    from rdkit import Chem

    # Validate target SMILES
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return {
            "target": target_smiles,
            "routes": [],
            "best_score": 0.0,
            "stats": {
                "simulations": 0,
                "time_seconds": 0.0,
                "routes_found": 0,
            },
            "molecules": [],
            "error": f"Invalid SMILES: {target_smiles}",
        }

    # Run MCTS search
    mcts = MCTS(policy, reward_calc, stock_list, max_simulations=max_simulations)
    result = mcts.search(target_smiles, time_budget=time_budget)

    # Build molecule info list from the best route
    molecules = []
    if result.best_route is not None:
        unique_smiles = flatten_route_molecules(result.best_route)
        for smi in unique_smiles:
            sa = RewardCalculator.compute_sascore(smi)
            molecules.append(
                {
                    "smiles": smi,
                    "sascore": sa if sa is not None else 0.0,
                    "in_stock": stock_list.is_buyable(smi),
                    "image_b64": mol_to_base64_image(smi),
                }
            )

    # Build stats dict (only the three documented keys)
    stats = {
        "simulations": result.stats.get("simulations", 0),
        "time_seconds": result.stats.get("time_seconds", 0.0),
        "routes_found": result.stats.get("routes_found", 0),
    }

    output = {
        "target": target_smiles,
        "routes": result.all_routes,
        "best_score": result.score,
        "stats": stats,
        "molecules": molecules,
    }

    # Add a note if no routes were found
    if not result.all_routes:
        output["note"] = "No synthesis routes found for this target."

    return output


def print_route_tree(route: dict, indent: int = 0) -> None:
    """Print a route tree in a human-readable indented format.

    Args:
        route: Route dict with "smiles", "score", "in_stock", "children".
        indent: Current indentation level.
    """
    prefix = "  " * indent
    stock_tag = " [IN STOCK]" if route.get("in_stock") else ""
    score_str = f" (score: {route['score']:.3f})" if "score" in route else ""
    print(f"{prefix}{route['smiles']}{score_str}{stock_tag}")
    for child in route.get("children", []):
        print_route_tree(child, indent + 1)


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

    # Load model
    policy = load_model(checkpoint_path=args.checkpoint, device=args.device)

    # Create reward calculator and stock list
    reward_calc = RewardCalculator()
    stock_list = StockList()
    stock_list.load()

    # Run inference
    result = run_inference(
        target_smiles=args.target,
        policy=policy,
        reward_calc=reward_calc,
        stock_list=stock_list,
        max_simulations=args.max_simulations,
        time_budget=args.time_budget,
    )

    # Print results as formatted JSON
    print("\n=== Inference Results (JSON) ===")
    print(json.dumps(result, indent=2, default=str))

    # Print the best route tree in human-readable format
    if result["routes"]:
        print("\n=== Best Route Tree ===")
        print_route_tree(result["routes"][0])
    else:
        print("\nNo routes found.")
