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
from rdkit.Chem import AllChem, rdMolDescriptors

from data.stock.loader import StockList
from env.Rewards import RewardCalculator
from scripts.inference import load_model, run_inference

try:
    from scripts.inference_pi import create_pi_client, run_inference_pi

    PI_AVAILABLE = True
except ImportError:
    PI_AVAILABLE = False

try:
    from scripts.inference_hf import run_inference_hf

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

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


def render_molecule_svg(smiles: str, max_width: int = 250):
    """Render a molecule as crisp SVG, scaled proportionally to atom count.

    Sizing: ~15px per heavy atom + base of 60px, capped at max_width.
    A 6-membered ring (benzene, 6 atoms) ≈ 150px wide.
    """
    try:
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        n_atoms = mol.GetNumHeavyAtoms()
        base = max(120, min(max_width, int(n_atoms * 15 + 60)))
        w, h = base, int(base * 0.75)

        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 1.5
        # Use white background for visibility on dark themes
        opts.setBackgroundColour((1, 1, 1, 1))
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Add rounded corners via wrapping div
        return svg
    except Exception:
        return None


def show_molecule(smiles: str, max_width: int = 250):
    """Display a molecule SVG in Streamlit with rounded white card styling."""
    svg = render_molecule_svg(smiles, max_width=max_width)
    if svg:
        st.markdown(
            f'<div style="background:white; border-radius:8px; '
            f'padding:8px; display:inline-block;">{svg}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"`{smiles}`")


def display_target_molecule(smiles, reward_calc, stock_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string. Please check your input.")
        return False

    st.subheader("Target Molecule")
    col_img, col_props = st.columns([1, 1])
    with col_img:
        show_molecule(smiles, max_width=350)
        st.caption(smiles)
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


def infer_reaction_type(product_smiles, reactant_smiles_list):
    """Infer a likely reaction type from the product and reactants."""
    try:
        reactant_str = " ".join(reactant_smiles_list).lower()
        prod = product_smiles.lower()

        # Check for common patterns
        if "c(=o)oc(c)=o" in reactant_str or "cc(=o)oc(c)=o" in reactant_str:
            if "c(=o)o" in prod:
                return "Acetylation (Ester)"
            if "c(=o)n" in prod:
                return "Acetylation (Amide)"
            return "Acylation"
        if "c(=o)cl" in reactant_str:
            return "Acyl Chloride Coupling"
        if any(s.strip() == "O" for s in reactant_smiles_list):
            return "Hydrolysis"
        if "[al" in reactant_str:
            return "Friedel-Crafts"
        if "b(o)o" in reactant_str:
            return "Suzuki Coupling"
        if "[na" in reactant_str or "[k" in reactant_str or "[li" in reactant_str:
            return "Base-Mediated"
        if "oc(=o)" in prod and "o" in reactant_str:
            return "Esterification"
        if "nc(=o)" in prod or "c(=o)n" in prod:
            return "Amide Bond Formation"
        if "ci" in reactant_str or "cbr" in reactant_str:
            return "Alkylation"
    except Exception:
        pass
    return "Retrosynthetic Disconnection"


def display_molecule_compact(smiles, stock_list):
    """Compact molecule display: SVG + name, for the diagram."""
    show_molecule(smiles, max_width=200)
    name = molecule_label(smiles, stock_list)
    if name:
        st.caption(f"**{name}**")
    else:
        st.caption(f"`{smiles[:40]}`")
    if stock_list.is_buyable(smiles):
        st.markdown(":green[In Stock]")


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
        # Skip if RDKit can't parse it (invalid/truncated SMILES)
        child_mol = Chem.MolFromSmiles(cs)
        if child_mol is None:
            continue
        # Skip if it's the same molecule as the parent (canonical match)
        try:
            parent_mol = Chem.MolFromSmiles(smiles)
            if parent_mol and Chem.MolToSmiles(parent_mol) == Chem.MolToSmiles(child_mol):
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


def _render_reactants_row(display_children, stock_list):
    """Render reactant molecules side by side with + signs between them."""
    n = len(display_children)
    if n == 0:
        return
    # Build columns: [mol] [+] [mol] [+] [mol] ...
    col_spec = []
    for i in range(n):
        col_spec.append(2)
        if i < n - 1:
            col_spec.append(0.3)
    cols = st.columns(col_spec)
    col_idx = 0
    for i, child in enumerate(display_children):
        with cols[col_idx]:
            display_molecule_compact(child.get("smiles", ""), stock_list)
        col_idx += 1
        if i < n - 1:
            with cols[col_idx]:
                st.markdown(
                    "<div style='display:flex; align-items:center; "
                    "justify-content:center; height:200px; font-size:2em;'>+</div>",
                    unsafe_allow_html=True,
                )
            col_idx += 1


def display_retrosynthesis_diagram(route, reward_calc, stock_list, step=1, max_depth=4):
    """Display a retrosynthesis route as a visual diagram.

    Layout per step (forward synthesis direction):
        [Reactant 1] + [Reactant 2]
             ↓ Reaction Type
        [Product structure]

    Reagent details are in expandable sections.
    """
    # Clean the route on first call
    if step == 1:
        route = _clean_route(route)

    smiles = route.get("smiles", "")
    children = route.get("children", [])
    display_children = children[:4]

    if not display_children:
        return

    # Separate valid and invalid reactants
    valid_children = []
    invalid_smiles = []
    for child in display_children:
        cs = child.get("smiles", "")
        if cs and Chem.MolFromSmiles(cs) is not None:
            valid_children.append(child)
        elif cs:
            invalid_smiles.append(cs)

    if not valid_children:
        st.warning(f"Step {step}: Model produced no valid reactants.")
        return

    # Infer reaction type
    reactant_smiles = [c.get("smiles", "") for c in valid_children]
    reaction_type = infer_reaction_type(smiles, reactant_smiles)
    product_name = molecule_label(smiles, stock_list) or smiles

    # ---- First: recurse into intermediates (deepest steps shown first) ----
    if step < max_depth:
        for child in valid_children:
            cs = child.get("smiles", "")
            ci = stock_list.is_buyable(cs)
            if child.get("children") and not ci:
                child_name = molecule_label(cs, stock_list) or cs
                st.markdown(f"##### Decomposing: **{child_name}**")
                display_retrosynthesis_diagram(
                    child, reward_calc, stock_list, step=step + 1, max_depth=max_depth
                )
                st.markdown("---")

    # ---- DIAGRAM: Reactants → Arrow → Product (forward direction) ----

    # Reactants
    _render_reactants_row(valid_children, stock_list)

    # Reaction arrow pointing DOWN (reactants above → product below)
    st.markdown(
        f"<div style='text-align:center; padding: 10px 0;'>"
        f"<span style='font-size: 2em;'>⬇</span><br>"
        f"<span style='background-color: #262730; padding: 4px 12px; "
        f"border-radius: 12px; font-size: 0.9em;'>"
        f"Step {step}: {reaction_type}</span></div>",
        unsafe_allow_html=True,
    )

    # Product structure (centered)
    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        show_molecule(smiles, max_width=300)
        st.markdown(f"**{product_name}**")
        if step == 1:
            st.caption(":red[TARGET PRODUCT]")
        else:
            st.caption(":orange[INTERMEDIATE]")

    # Note invalid predictions if any
    if invalid_smiles:
        st.caption(
            f"Note: {len(invalid_smiles)} invalid prediction(s) were filtered out from this step."
        )

    # Reagent details in expander
    reactant_names = []
    for c in valid_children:
        n = molecule_label(c.get("smiles", ""), stock_list)
        reactant_names.append(n if n else c.get("smiles", "")[:30])

    with st.expander(f"Step {step} details — {reaction_type}", expanded=False):
        st.markdown(f"**Reaction:** {' + '.join(reactant_names)} → {product_name}")
        st.markdown(f"**Type:** {reaction_type}")
        for c in valid_children:
            cs = c.get("smiles", "")
            ci = stock_list.is_buyable(cs)
            sa = reward_calc.compute_sascore(cs)
            name = molecule_label(cs, stock_list) or cs
            status = ":green[In Stock]" if ci else ":orange[Not in stock]"
            sa_str = f"{sa:.2f}" if sa else "N/A"
            st.markdown(f"- **{name}** — SA: {sa_str} — {status}")
        if invalid_smiles:
            st.markdown("**Filtered (invalid SMILES):**")
            for inv in invalid_smiles:
                st.markdown(f"- `{inv}`")


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

    # ---- Summary + Stats (compact) ----
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{best_score:.3f}")
    c2.metric("Simulations", stats.get("simulations", "N/A"))
    c3.metric("Search Time", f"{stats.get('time_seconds', 0):.1f}s")
    c4.metric("Routes Found", stats.get("routes_found", 0))

    # Materials summary in expander
    with st.expander("Materials Summary"):
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

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Starting Materials (In Stock):**")
            for sm in starting_set:
                name = molecule_label(sm, stock_list)
                label = f"{name} (`{sm}`)" if name else f"`{sm}`"
                st.markdown(f"- :green[{label}]")
            if not starting_set:
                st.markdown("- *None found in stock*")
        with col2:
            st.markdown("**Intermediates:**")
            for im in intermediate_set:
                name = molecule_label(im, stock_list)
                label = f"{name} (`{im}`)" if name else f"`{im}`"
                st.markdown(f"- :orange[{label}]")
            if not intermediate_set:
                st.markdown("- *Direct synthesis*")

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

    backends = ["Local Model (ReactionT5)"]
    if HF_AVAILABLE:
        backends.append("HuggingFace Inference API")
    if PI_AVAILABLE:
        backends.append("Prime Intellect API")

    backend = st.sidebar.selectbox(
        "Inference Backend",
        backends,
        index=0,
        key="backend",
    )

    pi_api_key = ""
    pi_model_id = ""
    hf_model_id = ""
    if backend == "HuggingFace Inference API":
        hf_model_id = st.sidebar.text_input(
            "Model ID",
            value="rhoahndur/retrosynthesis-qwen3-4b",
            key="hf_model_id",
        )
        st.sidebar.caption("Uses the free HF Serverless Inference API (rate-limited).")
        st.sidebar.success("Ready: HuggingFace Inference API")
    elif backend == "Prime Intellect API":
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
                    if backend == "HuggingFace Inference API":
                        result = run_inference_hf(
                            smiles,
                            reward_calc,
                            stock_list,
                            model_id=hf_model_id,
                            token=os.environ.get("HF_TOKEN", None),
                        )
                    elif backend == "Prime Intellect API":
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
