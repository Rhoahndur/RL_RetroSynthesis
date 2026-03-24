"""GGUF CPU inference client for retrosynthesis.

Loads the quantized Qwen3-4B GGUF model via llama-cpp-python and runs
inference on CPU. Slow (~10-20s) but free — no GPU or API key needed.

Usage:
    # As module (from Streamlit):
    from scripts.inference_hf import run_inference_hf
    result = run_inference_hf("CC(=O)Oc1ccccc1C(=O)O", reward_calc, stock)

    # As CLI:
    python scripts/inference_hf.py --target "CC(=O)Oc1ccccc1C(=O)O"
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_cpp import Llama
from rdkit import Chem

from data.stock.loader import StockList
from env.Rewards import RewardCalculator
from scripts.inference import mol_to_base64_image

GGUF_REPO = "rhoahndur/retrosynthesis-qwen3-4b-gguf"
GGUF_FILE = "retrosynthesis-qwen3-4b-Q4_K_M.gguf"

SYSTEM_PROMPT = (
    "You are a retrosynthesis expert. Given a target molecule as a SMILES string, "
    "predict the reactant molecules that can be combined to synthesize the target.\n\n"
    "Rules:\n"
    "- Output ONLY the reactant SMILES strings separated by '.'\n"
    "- Do NOT include any explanation, reasoning, or extra text\n"
    "- Each reactant must be a valid SMILES string\n"
    "- Prefer simpler, commercially available starting materials\n"
    "- Ensure atom conservation: reactant atoms should cover the product atoms\n\n"
    "Example:\n"
    "Input: CC(=O)Oc1ccccc1C(=O)O\n"
    "Output: OC(=O)c1ccccc1O.CC(=O)OC(C)=O"
)

_llm: Llama | None = None


def _get_llm() -> Llama:
    """Lazy-load the GGUF model (downloads ~2.5GB on first call)."""
    global _llm
    if _llm is None:
        print(f"Loading GGUF model from {GGUF_REPO}...")
        _llm = Llama.from_pretrained(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            n_ctx=512,
            n_threads=4,
            verbose=False,
        )
        print("GGUF model loaded.")
    return _llm


def run_inference_hf(
    target_smiles: str,
    reward_calc: RewardCalculator,
    stock_list: StockList,
    n_candidates: int = 3,
) -> dict:
    """Run retrosynthetic inference via quantized GGUF model on CPU.

    Args:
        target_smiles: SMILES string of the target molecule.
        reward_calc: RewardCalculator instance for scoring predictions.
        stock_list: StockList instance (pre-loaded) for buyability checks.
        n_candidates: Number of candidate completions to generate.

    Returns:
        Dict matching the standard result format.
    """
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return {
            "target": target_smiles,
            "routes": [],
            "best_score": 0.0,
            "stats": {"simulations": 0, "time_seconds": 0.0, "routes_found": 0},
            "molecules": [],
            "error": f"Invalid SMILES: {target_smiles}",
        }

    start_time = time.time()
    try:
        llm = _get_llm()
    except Exception as e:
        return {
            "target": target_smiles,
            "routes": [],
            "best_score": 0.0,
            "stats": {
                "simulations": 0,
                "time_seconds": time.time() - start_time,
                "routes_found": 0,
            },
            "molecules": [],
            "error": f"Model load error: {e}",
        }

    scored_candidates: list[tuple[float, list[str]]] = []
    for _ in range(n_candidates):
        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Predict the reactants for: {target_smiles}"},
                ],
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            raw_text = resp["choices"][0]["message"]["content"]
            if not raw_text:
                continue
            # Strip thinking tags if present
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            if not raw_text:
                continue

            reactants = [r.strip() for r in raw_text.split(".") if r.strip()]
            if not reactants:
                continue

            reward = reward_calc.combined_reward(target_smiles, reactants, stock_list)
            scored_candidates.append((reward, reactants))
        except Exception:
            continue

    elapsed = time.time() - start_time

    # Rank by reward descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:3]

    # Build route trees
    routes = []
    for reward, reactants in top_candidates:
        children = []
        for r in reactants:
            buyable = stock_list.is_buyable(r)
            children.append(
                {"smiles": r, "score": 1.0 if buyable else 0.0, "in_stock": buyable, "children": []}
            )
        routes.append(
            {"smiles": target_smiles, "score": reward, "in_stock": False, "children": children}
        )

    best_score = routes[0]["score"] if routes else 0.0

    # Collect unique molecules
    seen_smiles: set[str] = set()
    all_smiles_ordered: list[str] = []
    for route in routes:
        if route["smiles"] not in seen_smiles:
            seen_smiles.add(route["smiles"])
            all_smiles_ordered.append(route["smiles"])
        for child in route.get("children", []):
            if child["smiles"] not in seen_smiles:
                seen_smiles.add(child["smiles"])
                all_smiles_ordered.append(child["smiles"])

    molecules = []
    for smi in all_smiles_ordered:
        sa = RewardCalculator.compute_sascore(smi)
        molecules.append(
            {
                "smiles": smi,
                "sascore": sa if sa is not None else 0.0,
                "in_stock": stock_list.is_buyable(smi),
                "image_b64": mol_to_base64_image(smi),
            }
        )

    return {
        "target": target_smiles,
        "routes": routes,
        "best_score": best_score,
        "stats": {
            "simulations": len(scored_candidates),
            "time_seconds": elapsed,
            "routes_found": len(routes),
        },
        "molecules": molecules,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGUF CPU retrosynthesis inference")
    parser.add_argument("--target", required=True, help="Target SMILES string")
    parser.add_argument("--n", type=int, default=3, help="Number of candidates")
    args = parser.parse_args()

    reward_calc = RewardCalculator()
    stock = StockList()
    stock.load()
    result = run_inference_hf(args.target, reward_calc, stock, n_candidates=args.n)
    print(json.dumps(result, indent=2, default=str))
