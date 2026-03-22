"""Streamlit application for retrosynthesis demo.

Interactive UI that accepts a SMILES input, runs MCTS-based retrosynthetic
search, and displays the synthesis route with molecule visualizations.

Usage:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Preset demo molecules
PRESETS = {
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}


def main():
    """Main Streamlit app entry point.

    Layout:
    1. Header: title + description
    2. Input: SMILES text field + 4 preset buttons + search button
    3. Target display: 2D/3D molecule viewer + properties
    4. Results: route tree visualization + reward breakdown + search stats

    TODO: Implement full Streamlit UI.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
