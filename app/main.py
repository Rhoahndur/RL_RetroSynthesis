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

import base64
import re

import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors

from data.stock.loader import StockList
from env.Rewards import RewardCalculator
from scripts.inference import load_model, run_inference

# Preset demo molecules
PRESETS = {
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}


def stock_badge(in_stock: bool) -> str:
    """Return a colored badge string for stock status."""
    if in_stock:
        return "🟢 In Stock"
    return "🟡 Intermediate"


def _find_best_checkpoint() -> str | None:
    """Find the checkpoint with the highest reward in the checkpoints dir.

    Checkpoint filenames are expected to contain a reward value, e.g.
    ``step_100_reward_0.85.pt`` or ``best_0.9.pt``.

    Returns:
        Path string to the best checkpoint, or None if no checkpoints exist.
    """
    ckpt_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
    if not ckpt_dir.exists():
        return None

    pt_files = list(ckpt_dir.glob("*.pt"))
    if not pt_files:
        return None

    best_path = None
    best_reward = -float("inf")

    for p in pt_files:
        # Try to extract a reward number from the filename
        match = re.search(r"reward[_\-]?([\d.]+)", p.stem)
        if match:
            reward = float(match.group(1))
            if reward > best_reward:
                best_reward = reward
                best_path = p
        elif best_path is None:
            # Fallback: pick any checkpoint if none have reward in name
            best_path = p

    return str(best_path) if best_path else None


@st.cache_resource
def load_resources():
    """Load policy model, reward calculator, and stock list.

    Tries to load from the best checkpoint first; falls back to the
    pre-trained model if no checkpoints are found.
    """
    # Find best checkpoint
    ckpt_path = _find_best_checkpoint()

    # Load policy
    policy = load_model(checkpoint_path=ckpt_path)

    # Load reward calculator
    reward_calc = RewardCalculator()

    # Load stock list
    stock_list = StockList()
    stock_list.load()

    return policy, reward_calc, stock_list


def render_molecule_image(smiles: str, size: tuple = (300, 300)):
    """Render a 2D molecule image from SMILES. Returns a PIL Image or None."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


def render_3d_viewer(smiles: str):
    """Render an interactive 3D molecule viewer using py3Dmol + stmol."""
    try:
        import py3Dmol
        from stmol import showmol

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        mblock = Chem.MolToMolBlock(mol)

        view = py3Dmol.view(width=400, height=300)
        view.addModel(mblock, "mol")
        view.setStyle({"stick": {}})
        view.setBackgroundColor("white")
        view.zoomTo()
        showmol(view, height=300, width=400)
    except Exception:
        st.caption("3D viewer unavailable.")


def display_target_molecule(smiles: str, reward_calc: RewardCalculator, stock_list: StockList):
    """Show the target molecule with 2D image, properties, and 3D viewer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string. Please check your input.")
        return False

    st.subheader("Target Molecule")

    col_img, col_props = st.columns([1, 1])

    with col_img:
        img = render_molecule_image(smiles, size=(400, 400))
        if img is not None:
            st.image(img, caption=smiles, use_container_width=True)

    with col_props:
        st.markdown("**Properties**")

        # Molecular formula
        formula = rdMolDescriptors.CalcMolFormula(mol)
        st.markdown(f"**Molecular Formula:** `{formula}`")

        # SAscore
        sa_score = reward_calc.compute_sascore(smiles)
        if sa_score is not None:
            st.markdown(f"**SA Score:** {sa_score:.2f} / 10")
            # Progress bar: lower is easier; invert for display (1=easy, 10=hard)
            st.progress(min(1.0, sa_score / 10.0))
        else:
            st.markdown("**SA Score:** N/A")

        # Buyable status
        buyable = stock_list.is_buyable(smiles)
        if buyable:
            st.success("This molecule is commercially available (buyable).")
        else:
            st.info("This molecule is NOT in the stock list -- synthesis route needed.")

    # 3D viewer
    with st.expander("3D Molecule Viewer", expanded=False):
        render_3d_viewer(smiles)

    return True


def display_route_tree(
    route: dict, reward_calc: RewardCalculator, stock_list: StockList, depth: int = 0
):
    """Recursively display a retrosynthesis route tree.

    Each node shows its molecule image, SMILES, SAscore, and stock status.
    Children are laid out in columns at each decomposition level.
    """
    smiles = route.get("smiles", "")
    children = route.get("children", [])
    in_stock = route.get("in_stock", stock_list.is_buyable(smiles))

    if depth > 0:
        # Show decomposition arrow
        st.markdown(
            f"{'&nbsp;' * (depth * 4)}⬇️ **Step {depth}:** Decompose into reactants",
            unsafe_allow_html=True,
        )

    if not children:
        # Leaf node: single molecule
        _display_molecule_card(smiles, in_stock, reward_calc, stock_list)
    else:
        # Show children side by side
        cols = st.columns(len(children))
        for i, child in enumerate(children):
            with cols[i]:
                child_smiles = child.get("smiles", "")
                child_in_stock = child.get("in_stock", stock_list.is_buyable(child_smiles))
                _display_molecule_card(child_smiles, child_in_stock, reward_calc, stock_list)

                # Recurse into grandchildren
                grandchildren = child.get("children", [])
                if grandchildren:
                    display_route_tree(child, reward_calc, stock_list, depth=depth + 1)


def _display_molecule_card(
    smiles: str, in_stock: bool, reward_calc: RewardCalculator, stock_list: StockList
):
    """Display a single molecule card with image, SMILES, SAscore, stock badge."""
    img = render_molecule_image(smiles, size=(250, 250))
    if img is not None:
        st.image(img, use_container_width=True)

    st.code(smiles, language=None)

    sa = reward_calc.compute_sascore(smiles)
    if sa is not None:
        st.caption(f"SA Score: {sa:.2f}")

    st.markdown(stock_badge(in_stock))


def display_results(result: dict, reward_calc: RewardCalculator, stock_list: StockList):
    """Display inference results: route tree, stats, and reward breakdown."""
    routes = result.get("routes", [])
    stats = result.get("stats", {})
    best_score = result.get("best_score", 0.0)
    molecules = result.get("molecules", [])

    if not routes:
        st.warning("No complete synthesis route found. Try a simpler molecule.")
        return

    # -- Route Tree --
    st.subheader("Synthesis Route")

    # Show the target at the top
    target_smiles = result.get("target", "")
    st.markdown(f"**Target:** `{target_smiles}`")
    st.markdown("---")

    # Display the best route (first route)
    best_route = routes[0] if routes else None
    if best_route:
        display_route_tree(best_route, reward_calc, stock_list, depth=0)

    st.markdown("---")

    # -- All molecules in the route --
    if molecules:
        st.subheader("All Molecules in Route")
        # Lay out in rows of up to 4
        row_size = 4
        for row_start in range(0, len(molecules), row_size):
            row_mols = molecules[row_start : row_start + row_size]
            cols = st.columns(len(row_mols))
            for j, mol_info in enumerate(row_mols):
                with cols[j]:
                    m_smiles = mol_info.get("smiles", "")
                    m_in_stock = mol_info.get("in_stock", stock_list.is_buyable(m_smiles))
                    m_sa = mol_info.get("sascore", None)

                    # Try image from base64, fall back to direct render
                    img_b64 = mol_info.get("image_b64", None)
                    if img_b64:
                        img_bytes = base64.b64decode(img_b64)
                        st.image(img_bytes, use_container_width=True)
                    else:
                        img = render_molecule_image(m_smiles, size=(250, 250))
                        if img is not None:
                            st.image(img, use_container_width=True)

                    st.code(m_smiles, language=None)

                    if m_sa is not None:
                        st.caption(f"SA Score: {m_sa:.2f}")
                    else:
                        sa_val = reward_calc.compute_sascore(m_smiles)
                        if sa_val is not None:
                            st.caption(f"SA Score: {sa_val:.2f}")

                    st.markdown(stock_badge(m_in_stock))

    st.markdown("---")

    # -- Search Stats --
    st.subheader("Search Statistics")

    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Simulations", stats.get("simulations", "N/A"))
    with stat_cols[1]:
        time_s = stats.get("time_seconds", 0)
        st.metric("Search Time", f"{time_s:.1f}s")
    with stat_cols[2]:
        st.metric("Routes Found", stats.get("routes_found", 0))
    with stat_cols[3]:
        st.metric("Best Score", f"{best_score:.3f}")

    # -- Additional routes --
    if len(routes) > 1:
        with st.expander(f"View all {len(routes)} routes"):
            for idx, route in enumerate(routes[1:], start=2):
                st.markdown(f"**Route {idx}** (score: {route.get('score', 'N/A')})")
                display_route_tree(route, reward_calc, stock_list, depth=0)
                st.markdown("---")


def main():
    """Main Streamlit app entry point.

    Layout:
    1. Header: title + description
    2. Input: SMILES text field + 4 preset buttons + search button
    3. Target display: 2D/3D molecule viewer + properties
    4. Results: route tree visualization + reward breakdown + search stats
    """
    # ---- 1. Page Config + Title ----
    st.set_page_config(page_title="Retrosynthesis AI", layout="wide")
    st.title("Retrosynthesis AI — RL-Powered Route Finder")
    st.markdown(
        "Enter a molecule SMILES to find synthesis routes from commercially available starting materials."
    )

    # ---- 2. Model Loading (cached) ----
    with st.spinner("Loading model, reward calculator, and stock list..."):
        try:
            policy, reward_calc, stock_list = load_resources()
        except Exception as e:
            st.error(f"Failed to load resources: {e}")
            st.stop()

    # ---- 3. Input Section ----
    if "smiles_input" not in st.session_state:
        st.session_state.smiles_input = ""

    st.subheader("Input")

    # Preset buttons
    preset_cols = st.columns(4)
    for idx, (name, smi) in enumerate(PRESETS.items()):
        with preset_cols[idx]:
            if st.button(name, use_container_width=True):
                st.session_state.smiles_input = smi
                st.rerun()

    # SMILES text input
    smiles = st.text_input(
        "Enter SMILES string",
        value=st.session_state.smiles_input,
        key="smiles_field",
        placeholder="e.g. CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    )

    # Keep session state in sync with the text field
    if smiles != st.session_state.smiles_input:
        st.session_state.smiles_input = smiles

    # ---- 4. Target Molecule Display ----
    if smiles:
        valid = display_target_molecule(smiles, reward_calc, stock_list)
        if not valid:
            st.stop()
    else:
        st.info("Enter a SMILES string above or click a preset to get started.")
        st.stop()

    # ---- 5. Search Button + Results ----
    st.markdown("---")
    search_clicked = st.button("🔍 Find Synthesis Route", type="primary", use_container_width=True)

    if search_clicked:
        with st.spinner("Searching for synthesis routes..."):
            try:
                result = run_inference(
                    smiles,
                    policy,
                    reward_calc,
                    stock_list,
                    max_simulations=200,
                    time_budget=30,
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.stop()

        if result is None:
            st.warning("No complete synthesis route found. Try a simpler molecule.")
        else:
            display_results(result, reward_calc, stock_list)


if __name__ == "__main__":
    main()
