"""Streamlit application for retrosynthesis demo.

Interactive UI that accepts a SMILES input, runs retrosynthetic search
via Prime Intellect API or local ReactionT5 model, and displays routes.

Usage:
    streamlit run app/main.py
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re

import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors

from data.stock.loader import StockList
from env.Rewards import RewardCalculator
from scripts.inference import load_model, run_inference

try:
    from scripts.inference_pi import create_pi_client, run_inference_pi

    PI_AVAILABLE = True
except ImportError:
    PI_AVAILABLE = False

# Preset demo molecules
PRESETS = {
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}


# ---- Cached resource loaders ----


@st.cache_resource
def get_shared_resources():
    """Load reward calculator and stock list (lightweight, used by all backends)."""
    rc = RewardCalculator()
    sl = StockList()
    sl.load()
    return rc, sl


@st.cache_resource
def get_local_policy():
    """Load the local ReactionT5 policy model."""
    ckpt_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
    best_path = None
    best_reward = -float("inf")
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("*.pt"):
            match = re.search(r"reward[_\-]?([\d.]+)", p.stem)
            if match:
                r = float(match.group(1))
                if r > best_reward:
                    best_reward = r
                    best_path = p
            elif best_path is None:
                best_path = p
    return load_model(checkpoint_path=str(best_path) if best_path else None)


# ---- Display helpers ----


def stock_badge(in_stock: bool) -> str:
    if in_stock:
        return "🟢 In Stock"
    return "🟡 Intermediate"


def render_molecule_image(smiles: str, size=(300, 300)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


def display_target_molecule(smiles, reward_calc, stock_list):
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
        formula = rdMolDescriptors.CalcMolFormula(mol)
        st.markdown(f"**Molecular Formula:** `{formula}`")
        sa_score = reward_calc.compute_sascore(smiles)
        if sa_score is not None:
            st.markdown(f"**SA Score:** {sa_score:.2f} / 10")
            st.progress(min(1.0, sa_score / 10.0))
        buyable = stock_list.is_buyable(smiles)
        if buyable:
            st.success("Commercially available (buyable).")
        else:
            st.info("NOT in stock -- synthesis route needed.")

    with st.expander("3D Molecule Viewer", expanded=False):
        try:
            import py3Dmol
            from stmol import showmol

            mol3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol3d, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol3d)
            view = py3Dmol.view(width=400, height=300)
            view.addModel(Chem.MolToMolBlock(mol3d), "mol")
            view.setStyle({"stick": {}})
            view.setBackgroundColor("white")
            view.zoomTo()
            showmol(view, height=300, width=400)
        except Exception:
            st.caption("3D viewer unavailable.")
    return True


def molecule_label(smiles, stock_list):
    """Return a human-readable label for known molecules, or the SMILES."""
    known = {
        "CC(=O)Oc1ccccc1C(=O)O": "Aspirin",
        "OC(=O)c1ccccc1O": "Salicylic acid",
        "CC(=O)OC(C)=O": "Acetic anhydride",
        "CC(=O)Nc1ccc(O)cc1": "Acetaminophen",
        "Nc1ccc(O)cc1": "p-Aminophenol",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O": "Ibuprofen",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O": "Caffeine",
        "CCO": "Ethanol",
        "CO": "Methanol",
        "CC(=O)O": "Acetic acid",
        "CC(=O)Cl": "Acetyl chloride",
        "O": "Water",
        "CC(C)Cc1ccccc1": "Isobutylbenzene",
        "c1ccccc1": "Benzene",
        "CC(=O)c1ccccc1": "Acetophenone",
    }
    # Try canonical match
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canon = Chem.MolToSmiles(mol)
            if canon in known:
                return known[canon]
    except Exception:
        pass
    # Try direct match
    if smiles in known:
        return known[smiles]
    return None


def role_badge(in_stock, is_target=False):
    """Return a styled badge showing the molecule's role."""
    if is_target:
        return ":red[**TARGET PRODUCT**]"
    if in_stock:
        return ":green[**STARTING MATERIAL (In Stock)**]"
    return ":orange[**INTERMEDIATE**]"


def display_molecule_card(smiles, in_stock, reward_calc, stock_list, is_target=False):
    """Display a molecule card with image, name, SMILES, SA score, and role."""
    img = render_molecule_image(smiles, size=(250, 250))
    if img is not None:
        st.image(img, use_container_width=True)

    # Show name if known
    name = molecule_label(smiles, stock_list)
    if name:
        st.markdown(f"**{name}**")

    st.code(smiles, language=None)

    sa = reward_calc.compute_sascore(smiles)
    if sa is not None:
        st.caption(f"SA Score: {sa:.2f} (lower = easier to make)")

    st.markdown(role_badge(in_stock, is_target))


def _clean_route(route, ancestor_smiles=None):
    """Remove self-referencing children and invalid SMILES from a route tree.

    The model sometimes predicts the product as its own reactant, causing
    infinite recursion. This filters those out and deduplicates children.
    """
    if ancestor_smiles is None:
        ancestor_smiles = set()

    smiles = route.get("smiles", "")
    children = route.get("children", [])

    # Filter children: remove duplicates, self-references, and invalid SMILES
    seen = set()
    clean_children = []
    for child in children:
        cs = child.get("smiles", "")
        # Skip if: same as any ancestor (circular), already seen, or empty
        if not cs or cs in ancestor_smiles or cs in seen:
            continue
        # Skip if it's the same molecule as the parent
        try:
            parent_mol = Chem.MolFromSmiles(smiles)
            child_mol = Chem.MolFromSmiles(cs)
            if (
                parent_mol
                and child_mol
                and Chem.MolToSmiles(parent_mol) == Chem.MolToSmiles(child_mol)
            ):
                continue
        except Exception:
            pass
        seen.add(cs)
        # Recursively clean grandchildren
        new_ancestors = ancestor_smiles | {smiles}
        clean_child = {
            "smiles": cs,
            "score": child.get("score", 0),
            "in_stock": child.get("in_stock", False),
            "children": _clean_route(child, new_ancestors).get("children", []),
        }
        clean_children.append(clean_child)

    return {
        "smiles": smiles,
        "score": route.get("score", 0),
        "in_stock": route.get("in_stock", False),
        "children": clean_children,
    }


def display_retrosynthesis_diagram(route, reward_calc, stock_list, step=1, max_depth=4):
    """Display a retrosynthesis route as a step-by-step diagram.

    Shows: Target → arrow → Reactants → arrow → deeper reactants...
    Cleans the route tree first to remove self-references and duplicates.
    """
    # Clean the route on first call
    if step == 1:
        route = _clean_route(route)

    smiles = route.get("smiles", "")
    children = route.get("children", [])
    in_stock = route.get("in_stock", stock_list.is_buyable(smiles))

    # Show the target molecule at top level
    if step == 1:
        display_molecule_card(smiles, in_stock, reward_calc, stock_list, is_target=True)

    if not children:
        return

    # Reaction arrow + header
    st.markdown("---")
    product_name = molecule_label(smiles, stock_list) or smiles
    st.markdown(
        f"### Step {step}: Retrosynthetic Disconnection\n\n"
        f"**{product_name}** can be synthesized from:"
    )

    # Show reactants side by side (cap at 4 columns for readability)
    display_children = children[:4]
    cols = st.columns(max(len(display_children), 1))
    for i, child in enumerate(display_children):
        with cols[i]:
            cs = child.get("smiles", "")
            ci = child.get("in_stock", stock_list.is_buyable(cs))
            display_molecule_card(cs, ci, reward_calc, stock_list)

    # Equation summary
    reactant_names = []
    for child in display_children:
        cs = child.get("smiles", "")
        n = molecule_label(cs, stock_list)
        reactant_names.append(n if n else cs[:30])
    st.markdown(f"> **{' + '.join(reactant_names)}** --> **{product_name}**")

    # Recurse into intermediates (not in stock, have children)
    if step < max_depth:
        for child in display_children:
            cs = child.get("smiles", "")
            ci = child.get("in_stock", stock_list.is_buyable(cs))
            if child.get("children") and not ci:
                st.markdown("---")
                child_name = molecule_label(cs, stock_list) or cs
                st.markdown(f"#### Decomposing intermediate: **{child_name}**")
                display_retrosynthesis_diagram(
                    child, reward_calc, stock_list, step=step + 1, max_depth=max_depth
                )


def display_results(result, reward_calc, stock_list):
    routes = result.get("routes", [])
    stats = result.get("stats", {})
    best_score = result.get("best_score", 0.0)

    if not routes:
        st.warning("No synthesis route found. Try a simpler molecule.")
        if "error" in result:
            st.error(result["error"])
        return

    # ---- Retrosynthesis Diagram ----
    st.subheader("Retrosynthesis Route")

    best_route = routes[0]
    display_retrosynthesis_diagram(best_route, reward_calc, stock_list)

    # ---- Route Summary ----
    st.markdown("---")
    st.subheader("Route Summary")

    # Collect unique molecules from the cleaned route
    clean = _clean_route(best_route)
    starting_set = set()
    intermediate_set = set()

    def collect_molecules(node, depth=0):
        s = node.get("smiles", "")
        if not s:
            return
        ins = node.get("in_stock", stock_list.is_buyable(s))
        kids = node.get("children", [])
        if not kids:
            if ins:
                starting_set.add(s)
            else:
                intermediate_set.add(s)
        else:
            if depth > 0:
                intermediate_set.add(s)
            for c in kids:
                collect_molecules(c, depth + 1)

    collect_molecules(clean)
    starting_materials = list(starting_set)
    intermediates = list(intermediate_set)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Starting Materials (In Stock):**")
        if starting_materials:
            for sm in starting_materials:
                name = molecule_label(sm, stock_list)
                label = f"{name} (`{sm}`)" if name else f"`{sm}`"
                st.markdown(f"- :green[{label}]")
        else:
            st.markdown("- *None found in stock*")

    with col2:
        st.markdown("**Intermediates:**")
        if intermediates:
            for im in intermediates:
                name = molecule_label(im, stock_list)
                label = f"{name} (`{im}`)" if name else f"`{im}`"
                st.markdown(f"- :orange[{label}]")
        else:
            st.markdown("- *Direct synthesis (no intermediates)*")

    # ---- Search Stats ----
    st.markdown("---")
    st.subheader("Search Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Simulations", stats.get("simulations", "N/A"))
    c2.metric("Search Time", f"{stats.get('time_seconds', 0):.1f}s")
    c3.metric("Routes Found", stats.get("routes_found", 0))
    c4.metric("Best Score", f"{best_score:.3f}")

    if len(routes) > 1:
        with st.expander(f"View all {len(routes)} routes"):
            for idx, route in enumerate(routes[1:], start=2):
                st.markdown(f"**Route {idx}** (score: {route.get('score', 'N/A')})")
                display_retrosynthesis_diagram(route, reward_calc, stock_list)
                st.markdown("---")


# ---- Main App ----


def main():
    st.set_page_config(page_title="Retrosynthesis AI", layout="wide")
    st.title("Retrosynthesis AI — RL-Powered Route Finder")
    st.markdown(
        "Enter a molecule SMILES to find synthesis routes from commercially available starting materials."
    )

    # ---- Sidebar ----
    st.sidebar.title("Configuration")
    backend = st.sidebar.selectbox(
        "Inference Backend",
        ["Prime Intellect API", "Local Model (ReactionT5)"],
        index=0 if PI_AVAILABLE else 1,
        key="backend",
    )

    pi_api_key = ""
    pi_model_id = ""
    if backend == "Prime Intellect API":
        pi_api_key = st.sidebar.text_input(
            "API Key",
            type="password",
            value=os.environ.get("PRIME_API_KEY", ""),
            key="pi_api_key",
        )
        pi_model_id = st.sidebar.text_input(
            "Deployment ID",
            key="pi_model_id",
        )
        if not pi_api_key or not pi_model_id:
            st.sidebar.warning("Enter API key and Deployment ID to use PI backend.")
        else:
            st.sidebar.success("Ready: Prime Intellect API")
    else:
        st.sidebar.success("Ready: Local Model")

    # ---- Load resources ----
    reward_calc, stock_list = get_shared_resources()
    policy = None
    if backend == "Local Model (ReactionT5)":
        try:
            policy = get_local_policy()
        except Exception as e:
            st.sidebar.error(f"Model load failed: {e}")

    # ---- Input ----
    st.subheader("Input")

    # Preset buttons — set the widget value directly via its key
    preset_cols = st.columns(4)
    for idx, (name, smi) in enumerate(PRESETS.items()):
        with preset_cols[idx]:
            if st.button(name, use_container_width=True):
                st.session_state["smiles_input"] = smi
                st.rerun()

    # Single source of truth for SMILES: the widget key "smiles_input"
    smiles = st.text_input(
        "Enter SMILES string",
        key="smiles_input",
        placeholder="e.g. CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    )

    if not smiles:
        st.info("Enter a SMILES string above or click a preset to get started.")
        st.stop()

    # ---- Target display ----
    valid = display_target_molecule(smiles, reward_calc, stock_list)
    if not valid:
        st.stop()

    # ---- Search ----
    st.markdown("---")
    if st.button("Find Synthesis Route", type="primary", use_container_width=True):
        if backend == "Prime Intellect API" and (not pi_api_key or not pi_model_id):
            st.error("Enter your PI API Key and Deployment ID in the sidebar first.")
        elif backend == "Local Model (ReactionT5)" and policy is None:
            st.error("Local model not loaded. Check console for errors.")
        else:
            with st.spinner("Searching for synthesis routes..."):
                try:
                    if backend == "Prime Intellect API":
                        client = create_pi_client(api_key=pi_api_key)
                        result = run_inference_pi(
                            smiles, client, pi_model_id, reward_calc, stock_list
                        )
                    else:
                        result = run_inference(
                            smiles,
                            policy,
                            reward_calc,
                            stock_list,
                            max_simulations=200,
                            time_budget=30,
                        )
                    st.session_state["search_result"] = result
                    st.session_state["search_smiles"] = smiles
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.code(traceback.format_exc())

    # ---- Display persisted results ----
    if "search_result" in st.session_state and st.session_state.get("search_smiles") == smiles:
        display_results(st.session_state["search_result"], reward_calc, stock_list)


if __name__ == "__main__":
    main()
