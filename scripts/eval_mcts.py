"""MCTS full-route success rate evaluation.

Runs Monte Carlo Tree Search on a test set of molecules and measures
how often complete synthesis routes (all leaves buyable) are found.

Usage:
    python scripts/eval_mcts.py --num_molecules 10 --max_simulations 100
    python scripts/eval_mcts.py --num_molecules 4 --max_simulations 10 --time_budget 5
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.stock.loader import StockList
from env.MCTS import MCTS
from env.Rewards import RewardCalculator

DEMO_MOLECULES = [
    {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin"},
    {"smiles": "CC(=O)Nc1ccc(O)cc1", "name": "Acetaminophen"},
    {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "name": "Caffeine"},
    {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "name": "Ibuprofen"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCTS full-route success rate evaluation")
    parser.add_argument("--num_molecules", type=int, default=10)
    parser.add_argument("--max_simulations", type=int, default=100)
    parser.add_argument("--time_budget", type=float, default=30.0)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="rhoahndur/retrosyn-targets")
    parser.add_argument("--stock_path", type=str, default="data/stock/buyables.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def compute_route_depth(route: dict) -> int:
    """Compute the maximum depth of a route tree.

    Args:
        route: Route dict with recursive "children" key.

    Returns:
        Max depth (0 for leaf nodes).
    """
    children = route.get("children", [])
    if not children:
        return 0
    return 1 + max(compute_route_depth(child) for child in children)


def load_test_molecules(dataset_name: str, num_molecules: int) -> list:
    """Load test molecules, always including the 4 demo molecules.

    Args:
        dataset_name: HF dataset identifier.
        num_molecules: Total number of molecules to return.

    Returns:
        List of {"smiles": str, "name": str} dicts.
    """
    molecules = list(DEMO_MOLECULES)

    if num_molecules <= len(molecules):
        return molecules[:num_molecules]

    # Try loading additional molecules from HF Hub
    needed = num_molecules - len(molecules)
    demo_smiles = {m["smiles"] for m in molecules}

    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="test")
        candidates = []
        for row in ds:
            question = row.get("question", "")
            product = question.replace("Predict the reactants for: ", "")
            if product and product not in demo_smiles:
                sa = RewardCalculator.compute_sascore(product)
                if sa is not None:
                    candidates.append(
                        {"smiles": product, "name": f"USPTO-{len(candidates)}", "sa": sa}
                    )

        if candidates:
            # Select evenly across SA buckets for diversity
            easy = [c for c in candidates if c["sa"] <= 3.0]
            medium = [c for c in candidates if 3.0 < c["sa"] <= 5.0]
            hard = [c for c in candidates if c["sa"] > 5.0]

            rng = random.Random(42)
            for bucket in [easy, medium, hard]:
                rng.shuffle(bucket)

            # Round-robin from buckets
            bucket_iters = [iter(easy), iter(medium), iter(hard)]
            added = 0
            while added < needed:
                progressed = False
                for it in bucket_iters:
                    if added >= needed:
                        break
                    try:
                        c = next(it)
                        molecules.append({"smiles": c["smiles"], "name": c["name"]})
                        added += 1
                        progressed = True
                    except StopIteration:
                        continue
                if not progressed:
                    break
    except Exception:
        pass

    return molecules[:num_molecules]


def evaluate_molecule(mcts: MCTS, smiles: str, name: str) -> dict:
    """Evaluate MCTS route finding for a single molecule.

    Args:
        mcts: Configured MCTS instance.
        smiles: Target molecule SMILES.
        name: Molecule name for display.

    Returns:
        Result dict with success, routes, score, depth, and timing info.
    """
    result = mcts.search(smiles)

    num_complete = sum(1 for r in result.all_routes if MCTS._is_route_complete(r))
    depth = compute_route_depth(result.best_route) if result.best_route else 0

    return {
        "smiles": smiles,
        "name": name,
        "success": num_complete > 0,
        "num_routes": len(result.all_routes),
        "num_complete": num_complete,
        "best_score": result.score,
        "route_depth": depth,
        "simulations": result.stats["simulations"],
        "time_seconds": result.stats["time_seconds"],
    }


def print_results(molecule_results: list, args) -> None:
    """Print evaluation results as a formatted table."""
    n = len(molecule_results)
    sims = args.max_simulations
    budget = args.time_budget

    print(f"\nMCTS Route Evaluation (N={n}, sims={sims}, budget={budget}s)")
    sep = "-" * 72
    print(sep)
    print(
        f"{'Molecule':20s} {'Success':>8s} {'Routes':>7s} {'Complete':>9s} "
        f"{'Score':>6s} {'Depth':>6s} {'Time':>7s}"
    )
    print(sep)

    for r in molecule_results:
        success_str = "Yes" if r["success"] else "No"
        print(
            f"{r['name']:20s} {success_str:>8s} {r['num_routes']:>7d} "
            f"{r['num_complete']:>9d} {r['best_score']:>6.2f} "
            f"{r['route_depth']:>6d} {r['time_seconds']:>6.1f}s"
        )

    print(sep)

    # Summary
    successes = sum(1 for r in molecule_results if r["success"])
    success_rate = successes / n if n > 0 else 0.0

    successful_results = [r for r in molecule_results if r["success"]]
    avg_depth = (
        sum(r["route_depth"] for r in successful_results) / len(successful_results)
        if successful_results
        else 0.0
    )
    avg_time = sum(r["time_seconds"] for r in molecule_results) / n if n > 0 else 0.0
    avg_sims = sum(r["simulations"] for r in molecule_results) / n if n > 0 else 0.0

    print("\nSummary:")
    print(f"  Success rate:    {success_rate:.1%}")
    print(f"  Avg route depth: {avg_depth:.1f}")
    print(f"  Avg time:        {avg_time:.1f}s")
    print(f"  Avg simulations: {avg_sims:.0f}")


def main() -> None:
    args = parse_args()

    print("Loading model...")
    from models.policy import RetroPolicy

    device = RetroPolicy.detect_device() if args.device == "auto" else args.device
    policy = RetroPolicy(device=device)

    reward_calc = RewardCalculator()
    stock = StockList()
    stock.load(args.stock_path)

    print(f"Loading test molecules ({args.num_molecules})...")
    molecules = load_test_molecules(args.dataset, args.num_molecules)
    print(f"  Loaded {len(molecules)} molecules")

    print(f"Evaluating MCTS (sims={args.max_simulations}, budget={args.time_budget}s)...\n")
    results = []
    for mol in molecules:
        mcts_instance = MCTS(
            policy,
            reward_calc,
            stock,
            max_depth=args.max_depth,
            max_simulations=args.max_simulations,
            top_k=args.top_k,
        )
        r = evaluate_molecule(mcts_instance, mol["smiles"], mol["name"])
        results.append(r)
        status = "OK" if r["success"] else "FAIL"
        print(f"  [{status}] {mol['name']} ({r['time_seconds']:.1f}s)")

    print_results(results, args)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
