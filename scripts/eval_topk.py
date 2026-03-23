"""Top-K exact match evaluation for retrosynthesis models.

Compares model predictions against ground-truth reactions from USPTO-50K.
Reports accuracy overall and stratified by synthetic accessibility.

Usage:
    python scripts/eval_topk.py --num_examples 100
    python scripts/eval_topk.py --num_examples 20 --top_k 5
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit import Chem

from env.Rewards import RewardCalculator

# Known reactions for fallback when HF Hub is unavailable
FALLBACK_REACTIONS = [
    {
        "product": "CC(=O)Oc1ccccc1C(=O)O",
        "ground_truth": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
    },
    {
        "product": "CC(=O)Nc1ccc(O)cc1",
        "ground_truth": "Nc1ccc(O)cc1.CC(=O)OC(C)=O",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-K exact match evaluation")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="rhoahndur/retrosyn-targets")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--by_reaction_type",
        action="store_true",
        help="Break down results by reaction class (requires class labels in dataset)",
    )
    return parser.parse_args()


def canonicalize_reaction(smiles_str: str) -> frozenset:
    """Canonicalize a dot-separated SMILES string into an order-independent set.

    Args:
        smiles_str: Dot-separated reactant SMILES (e.g. "CCO.CC(=O)O").

    Returns:
        Frozenset of canonical SMILES. Invalid fragments are skipped.
    """
    if not smiles_str or not isinstance(smiles_str, str):
        return frozenset()
    parts = smiles_str.split(".")
    canonical = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            mol = Chem.MolFromSmiles(part)
            if mol is not None:
                canonical.add(Chem.MolToSmiles(mol))
        except Exception:
            continue
    return frozenset(canonical)


def compute_sascore_bucket(product_smiles: str) -> str:
    """Classify a molecule's synthetic accessibility into easy/medium/hard.

    Args:
        product_smiles: SMILES string.

    Returns:
        "easy" (SA 1-3), "medium" (SA 3-5), "hard" (SA 5+), or "unknown".
    """
    score = RewardCalculator.compute_sascore(product_smiles)
    if score is None:
        return "unknown"
    if score <= 3.0:
        return "easy"
    if score <= 5.0:
        return "medium"
    return "hard"


def load_eval_dataset(dataset_name: str, num_examples: int) -> list:
    """Load evaluation examples from HuggingFace Hub or fallback.

    Args:
        dataset_name: HF dataset identifier.
        num_examples: Number of examples to load.

    Returns:
        List of {"product": str, "ground_truth": str} dicts.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="test")
        rows = []
        for row in ds:
            question = row.get("question", "")
            product = question.replace("Predict the reactants for: ", "")
            answer = row.get("answer", "")
            if product and answer:
                entry = {"product": product, "ground_truth": answer}
                # Preserve reaction type/class if present in dataset
                for key in ("reaction_type", "class", "reaction_class"):
                    if key in row and row[key]:
                        entry["reaction_type"] = str(row[key])
                        break
                rows.append(entry)
        if rows:
            rng = random.Random(42)
            rng.shuffle(rows)
            return rows[:num_examples]
    except Exception:
        pass

    return FALLBACK_REACTIONS[:num_examples]


def evaluate(examples: list, policy, top_k: int, by_reaction_type: bool = False) -> dict:
    """Run top-K exact match evaluation.

    Args:
        examples: List of {"product": str, "ground_truth": str,
                  optional "reaction_type": str}.
        policy: Policy with predict_greedy(smiles, num_beams=K) method.
        top_k: Maximum K for top-K matching.
        by_reaction_type: If True, also break down by reaction class.

    Returns:
        Results dict with overall, per-bucket, and optionally per-reaction-type
        accuracy at K=1,3,5,10.
    """
    k_values = [k for k in [1, 3, 5, 10] if k <= top_k]

    # Accumulators: overall and per-bucket
    overall = {k: 0 for k in k_values}
    buckets: dict[str, dict] = {}
    bucket_counts: dict[str, int] = {}

    # Per-reaction-type accumulators
    rxn_types: dict[str, dict] = {}
    rxn_type_counts: dict[str, int] = {}
    has_reaction_types = False

    for example in examples:
        product = example["product"]
        gt = canonicalize_reaction(example["ground_truth"])
        if not gt:
            continue

        bucket = compute_sascore_bucket(product)
        if bucket not in buckets:
            buckets[bucket] = {k: 0 for k in k_values}
            bucket_counts[bucket] = 0
        bucket_counts[bucket] += 1

        # Track reaction type if available
        rxn_type = example.get("reaction_type")
        if rxn_type and by_reaction_type:
            has_reaction_types = True
            if rxn_type not in rxn_types:
                rxn_types[rxn_type] = {k: 0 for k in k_values}
                rxn_type_counts[rxn_type] = 0
            rxn_type_counts[rxn_type] += 1

        try:
            predictions = policy.predict_greedy(product, num_beams=top_k)
        except Exception:
            predictions = []

        pred_sets = [canonicalize_reaction(p) for p in predictions if p]

        for k in k_values:
            matched = any(p == gt for p in pred_sets[:k])
            if matched:
                overall[k] += 1
                buckets[bucket][k] += 1
                if rxn_type and by_reaction_type and rxn_type in rxn_types:
                    rxn_types[rxn_type][k] += 1

    total = sum(bucket_counts.values())
    results = {
        "total_examples": total,
        "top_k": top_k,
        "overall": {f"top_{k}": overall[k] / total if total > 0 else 0.0 for k in k_values},
        "by_bucket": {},
    }

    for bucket in sorted(buckets.keys()):
        n = bucket_counts[bucket]
        results["by_bucket"][bucket] = {
            "count": n,
            **{f"top_{k}": buckets[bucket][k] / n if n > 0 else 0.0 for k in k_values},
        }

    # Add reaction type breakdown if requested and data exists
    if by_reaction_type and has_reaction_types:
        results["by_reaction_type"] = {}
        for rxn_type in sorted(rxn_types.keys()):
            n = rxn_type_counts[rxn_type]
            results["by_reaction_type"][rxn_type] = {
                "count": n,
                **{f"top_{k}": rxn_types[rxn_type][k] / n if n > 0 else 0.0 for k in k_values},
            }
    elif by_reaction_type:
        print("WARNING: --by_reaction_type requested but no reaction class labels found in dataset")

    return results


def print_results(results: dict) -> None:
    """Print evaluation results as a formatted table."""
    total = results["total_examples"]
    k_values = [k for k in [1, 3, 5, 10] if f"top_{k}" in results["overall"]]

    header = f"{'':12s}" + "".join(f"{'Top-' + str(k):>8s}" for k in k_values)
    sep = "-" * len(header)

    print(f"\nTop-K Exact Match Results (N={total})")
    print(sep)
    print(header)
    print(sep)

    # Overall row
    row = f"{'Overall':12s}"
    for k in k_values:
        pct = results["overall"][f"top_{k}"] * 100
        row += f"{pct:7.1f}%"
    print(row)

    # Per-bucket rows
    for bucket in ["easy", "medium", "hard", "unknown"]:
        if bucket not in results["by_bucket"]:
            continue
        data = results["by_bucket"][bucket]
        label = f"{bucket} (n={data['count']})"
        row = f"{label:12s}"
        for k in k_values:
            pct = data[f"top_{k}"] * 100
            row += f"{pct:7.1f}%"
        print(row)

    print(sep)

    # Reaction type breakdown
    if "by_reaction_type" in results:
        rxn_data = results["by_reaction_type"]
        print(f"\nAccuracy by Reaction Type ({len(rxn_data)} classes)")
        max_label = max(len(rt) for rt in rxn_data) + 8  # room for (n=XX)
        rxn_header = f"{'':>{max_label}s}" + "".join(f"{'Top-' + str(k):>8s}" for k in k_values)
        rxn_sep = "-" * len(rxn_header)
        print(rxn_sep)
        print(rxn_header)
        print(rxn_sep)

        # Sort by highest-K accuracy descending
        sort_key = f"top_{k_values[-1]}"
        sorted_types = sorted(rxn_data.items(), key=lambda x: x[1][sort_key], reverse=True)
        for rxn_type, data in sorted_types:
            label = f"{rxn_type} (n={data['count']})"
            row = f"{label:>{max_label}s}"
            for k in k_values:
                pct = data[f"top_{k}"] * 100
                row += f"{pct:7.1f}%"
            # Flag blind spots
            worst_k = k_values[-1]
            if data[f"top_{worst_k}"] < 0.1 and data["count"] >= 3:
                row += "  << blind spot"
            print(row)
        print(rxn_sep)


def main() -> None:
    args = parse_args()

    print(f"Loading eval dataset ({args.num_examples} examples)...")
    examples = load_eval_dataset(args.dataset, args.num_examples)
    print(f"  Loaded {len(examples)} examples")

    print("Loading model...")
    from models.policy import RetroPolicy

    device = RetroPolicy.detect_device() if args.device == "auto" else args.device
    policy = RetroPolicy(device=device)

    print(f"Evaluating top-{args.top_k} exact match...")
    results = evaluate(examples, policy, args.top_k, by_reaction_type=args.by_reaction_type)
    print_results(results)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
