"""Data preparation script — download and process training targets.

Downloads USPTO-50K via TDC, extracts product SMILES, canonicalizes,
deduplicates, and saves training/validation splits.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --stock_path data/stock/buyables.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--stock_path", type=str, default="data/stock/buyables.csv")
    return parser.parse_args()


def download_and_process(output_dir: str, stock_path: str) -> None:
    """Download USPTO-50K, extract products, canonicalize, deduplicate, save.

    Steps:
    1. Download USPTO-50K via TDC: RetroSyn(name='USPTO-50k')
    2. Extract unique product SMILES from train/test splits
    3. Canonicalize all SMILES via RDKit
    4. Deduplicate
    5. Filter out molecules already in stock list
    6. Save: training_targets.csv (from train split), validation_targets.csv (from test split)
    7. Print stats

    Args:
        output_dir: Directory to save processed CSVs.
        stock_path: Path to buyables CSV for filtering.
    """
    import pandas as pd
    from rdkit import Chem
    from tdc import RetroSyn

    # ------------------------------------------------------------------
    # 1. Download USPTO-50K via TDC
    # ------------------------------------------------------------------
    print("Downloading USPTO-50K via TDC...")
    data = RetroSyn(name="USPTO-50k")
    split = data.get_split()  # dict with 'train', 'valid', 'test'
    print(f"  Splits received: {list(split.keys())}")

    # ------------------------------------------------------------------
    # Helper: canonicalize a SMILES string
    # ------------------------------------------------------------------
    def canonicalize(smi: str):
        """Return canonical SMILES or None if invalid."""
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helper: detect the product column name in a TDC dataframe
    # ------------------------------------------------------------------
    def find_product_column(df: pd.DataFrame) -> str:
        """TDC RetroSyn may name the product column differently across versions."""
        for candidate in ["input", "Input", "Product", "product", "PRODUCT"]:
            if candidate in df.columns:
                return candidate
        raise KeyError(f"Could not find product column. Available columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # Helper: process one split (extract, canonicalize, deduplicate)
    # ------------------------------------------------------------------
    def process_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Extract product SMILES, canonicalize, and deduplicate."""
        col = find_product_column(df)
        raw_products = df[col].dropna().tolist()
        total_raw = len(raw_products)

        # Canonicalize
        canonical = []
        for smi in raw_products:
            c = canonicalize(str(smi).strip())
            if c is not None:
                canonical.append(c)
        valid_count = len(canonical)

        # Deduplicate (preserve order with dict.fromkeys)
        unique = list(dict.fromkeys(canonical))
        unique_count = len(unique)

        print(f"  [{split_name}] raw={total_raw}  valid={valid_count}  unique={unique_count}")
        return pd.DataFrame({"smiles": unique})

    # ------------------------------------------------------------------
    # 2-4. Process train and test splits
    # ------------------------------------------------------------------
    print("Processing splits...")
    train_df = process_split(split["train"], "train")
    # Use test split for validation targets (standard practice for USPTO-50K)
    valid_df = process_split(split["test"], "test")

    # ------------------------------------------------------------------
    # 5. Filter out buyable molecules (if stock file exists)
    # ------------------------------------------------------------------
    stock_loaded = False
    try:
        if os.path.isfile(stock_path):
            from data.stock.loader import StockList

            print(f"Loading stock list from {stock_path}...")
            stock = StockList()
            stock.load(stock_path)
            print(f"  Stock list contains {len(stock)} molecules.")

            train_before = len(train_df)
            valid_before = len(valid_df)

            train_df = train_df[~train_df["smiles"].apply(stock.is_buyable)].reset_index(drop=True)
            valid_df = valid_df[~valid_df["smiles"].apply(stock.is_buyable)].reset_index(drop=True)

            print(f"  [train] {train_before} -> {len(train_df)} after removing buyables")
            print(f"  [test]  {valid_before} -> {len(valid_df)} after removing buyables")
            stock_loaded = True
        else:
            print(f"WARNING: Stock file not found at '{stock_path}' — skipping buyable filtering.")
    except Exception as e:
        print(f"WARNING: Could not load stock list ({e}) — skipping buyable filtering.")

    # ------------------------------------------------------------------
    # 6. Save processed CSVs
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "training_targets.csv")
    valid_path = os.path.join(output_dir, "validation_targets.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    print(f"Saved {train_path}  ({len(train_df)} molecules)")
    print(f"Saved {valid_path}  ({len(valid_df)} molecules)")

    # ------------------------------------------------------------------
    # 7. Summary stats
    # ------------------------------------------------------------------
    print("")
    print("=== Data Preparation Summary ===")
    print(f"  Training targets:   {len(train_df)}")
    print(f"  Validation targets: {len(valid_df)}")
    print(f"  Buyable filtering:  {'applied' if stock_loaded else 'skipped'}")
    print(f"  Output directory:   {output_dir}")
    print("================================")


if __name__ == "__main__":
    args = parse_args()
    download_and_process(args.output_dir, args.stock_path)
