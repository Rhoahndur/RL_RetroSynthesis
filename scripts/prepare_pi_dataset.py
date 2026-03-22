"""Prepare USPTO-50K as a HuggingFace dataset for Prime Intellect training.

Usage:
    python scripts/prepare_pi_dataset.py
    python scripts/prepare_pi_dataset.py --push --hub-name myuser/retrosyn-targets
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare USPTO-50K as a HuggingFace dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--stock-path", type=str, default="data/stock/buyables.csv")
    parser.add_argument("--push", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HuggingFace Hub dataset name (e.g. myuser/retrosyn-targets)",
    )
    return parser.parse_args()


def load_stock_list(stock_path: str) -> list[str]:
    """Read buyables.csv, return list of canonical SMILES.

    Args:
        stock_path: Path to buyables CSV (columns: smiles,name,category).

    Returns:
        List of canonical SMILES strings.
    """
    from rdkit import Chem

    smiles_list = []
    if not os.path.isfile(stock_path):
        print(f"WARNING: Stock file not found at '{stock_path}', returning empty list.")
        return smiles_list

    with open(stock_path) as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if not parts:
                continue
            raw = parts[0].strip()
            try:
                mol = Chem.MolFromSmiles(raw)
                if mol is not None:
                    smiles_list.append(Chem.MolToSmiles(mol))
            except Exception:
                continue

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for smi in smiles_list:
        if smi not in seen:
            seen.add(smi)
            unique.append(smi)

    return unique


def find_column(df, candidates: list[str]) -> str:
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")


def canonicalize(smi: str):
    """Return canonical SMILES or None if invalid."""
    from rdkit import Chem

    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def prepare_dataset(
    output_dir: str = "data/processed",
    stock_path: str = "data/stock/buyables.csv",
    push: bool = False,
    hub_name: str | None = None,
) -> None:
    """Download USPTO-50K, format as HuggingFace Dataset with question/answer/info columns.

    Args:
        output_dir: Directory to save parquet files.
        stock_path: Path to buyables CSV for building the stock list context.
        push: Whether to push the dataset to HuggingFace Hub.
        hub_name: Hub dataset name (required if push is True).
    """
    from datasets import Dataset, DatasetDict
    from tdc import RetroSyn

    # 1. Download USPTO-50K
    print("Downloading USPTO-50K via TDC...")
    data = RetroSyn(name="USPTO-50k")
    split = data.get_split()
    print(f"  Splits received: {list(split.keys())}")

    # 2. Load buyable SMILES for the info column context
    print(f"Loading stock list from {stock_path}...")
    stock_smiles = load_stock_list(stock_path)
    print(f"  Stock list contains {len(stock_smiles)} unique molecules.")
    # Limit to 100 entries to keep info column manageable
    stock_context = stock_smiles[:100]

    # 3. Detect column names in TDC dataframes
    sample_df = split["train"]
    product_col = find_column(sample_df, ["input", "Input", "Product", "product", "PRODUCT"])
    reactant_col = find_column(
        sample_df, ["output", "Output", "Reactants", "reactants", "REACTANTS"]
    )
    print(f"  Product column: '{product_col}', Reactant column: '{reactant_col}'")

    # 4. Build rows for each split
    def build_rows(df, split_name: str) -> dict[str, list]:
        questions = []
        answers = []
        infos = []
        skipped = 0

        for _, row in df.iterrows():
            product_raw = str(row[product_col]).strip()
            reactant_raw = str(row[reactant_col]).strip()

            product = canonicalize(product_raw)
            if product is None:
                skipped += 1
                continue

            # Canonicalize each reactant in the dot-separated list
            reactant_parts = [r.strip() for r in reactant_raw.split(".") if r.strip()]
            canon_reactants = []
            for r in reactant_parts:
                cr = canonicalize(r)
                if cr is not None:
                    canon_reactants.append(cr)

            if not canon_reactants:
                skipped += 1
                continue

            answer = ".".join(canon_reactants)

            question = f"Predict the reactants for: {product}"
            info = json.dumps(
                {
                    "product_smiles": product,
                    "stock_list": stock_context,
                },
                ensure_ascii=False,
            )

            questions.append(question)
            answers.append(answer)
            infos.append(info)

        print(f"  [{split_name}] {len(questions)} rows built, {skipped} skipped (invalid SMILES)")
        return {"question": questions, "answer": answers, "info": infos}

    print("Building dataset rows...")
    train_rows = build_rows(split["train"], "train")
    test_rows = build_rows(split["test"], "test")

    # Also process valid split if present
    has_valid = "valid" in split and len(split["valid"]) > 0
    if has_valid:
        valid_rows = build_rows(split["valid"], "valid")

    # 5. Create HuggingFace Datasets
    train_ds = Dataset.from_dict(train_rows)
    test_ds = Dataset.from_dict(test_rows)

    splits = {"train": train_ds, "test": test_ds}
    if has_valid:
        valid_ds = Dataset.from_dict(valid_rows)
        splits["valid"] = valid_ds

    ds_dict = DatasetDict(splits)

    print("\nDataset summary:")
    for name, ds in ds_dict.items():
        print(f"  {name}: {len(ds)} examples")

    # 6. Save as parquet files
    os.makedirs(output_dir, exist_ok=True)
    for name, ds in ds_dict.items():
        parquet_path = os.path.join(output_dir, f"pi_{name}.parquet")
        ds.to_parquet(parquet_path)
        print(f"  Saved {parquet_path}")

    # 7. Optionally push to HuggingFace Hub
    if push:
        if hub_name is None:
            print("ERROR: --hub-name is required when using --push")
            sys.exit(1)
        print(f"Pushing dataset to HuggingFace Hub as '{hub_name}'...")
        ds_dict.push_to_hub(hub_name)
        print(f"  Pushed successfully to https://huggingface.co/datasets/{hub_name}")

    print("\nDone.")


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        output_dir=args.output_dir,
        stock_path=args.stock_path,
        push=args.push,
        hub_name=args.hub_name,
    )
