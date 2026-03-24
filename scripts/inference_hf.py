"""HuggingFace Inference API client for retrosynthesis.

Calls a PEFT LoRA model hosted on HuggingFace Hub via the free Serverless
Inference API, then scores results with RDKit rewards.

Usage:
    # As module (from Streamlit):
    from scripts.inference_hf import run_inference_hf
    result = run_inference_hf("CC(=O)Oc1ccccc1C(=O)O", reward_calc, stock)

    # As CLI:
    python scripts/inference_hf.py --target "CC(=O)Oc1ccccc1C(=O)O"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import InferenceClient
from rdkit import Chem

from data.stock.loader import StockList
from env.Rewards import RewardCalculator
from scripts.inference import mol_to_base64_image

DEFAULT_MODEL_ID = "rhoahndur/retrosynthesis-qwen3-4b"

SYSTEM_PROMPT = """You are a retrosynthesis expert. Given a target molecule as a SMILES string,
predict the reactant molecules that can be combined to synthesize the target.

Rules:
- Output ONLY the reactant SMILES strings separated by '.'
- Do NOT include any explanation, reasoning, or extra text
- Each reactant must be a valid SMILES string
- Prefer simpler, commercially available starting materials
- Ensure atom conservation: reactant atoms should cover the product atoms

Example:
Input: CC(=O)Oc1ccccc1C(=O)O
Output: OC(=O)c1ccccc1O.CC(=O)OC(C)=O"""


def run_inference_hf(
    target_smiles: str,
    reward_calc: RewardCalculator,
    stock_list: StockList,
    model_id: str = DEFAULT_MODEL_ID,
    token: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    n_candidates: int = 3,
) -> dict:
    """Run retrosynthetic inference via HuggingFace Inference API.

    Args:
        target_smiles: SMILES string of the target molecule.
        reward_calc: RewardCalculator instance for scoring predictions.
        stock_list: StockList instance (pre-loaded) for buyability checks.
        model_id: HuggingFace model/adapter repo ID.
        token: HF API token. Defaults to HF_TOKEN env var.
        system_prompt: System prompt for the chat model.
        n_candidates: Number of candidate completions to request.

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

    if token is None:
        token = os.environ.get("HF_TOKEN", None)
    client = InferenceClient(model=model_id, token=token)

    start_time = time.time()
    scored_candidates: list[tuple[float, list[str]]] = []

    # HF Inference API doesn't support n>1 in one call, so loop
    for _ in range(n_candidates):
        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Predict the reactants for: {target_smiles}"},
                ],
                max_tokens=256,
                temperature=0.7,
            )
            raw_text = response.choices[0].message.content
            if not raw_text:
                continue
            raw_text = raw_text.strip()
            if not raw_text:
                continue

            reactants = [r.strip() for r in raw_text.split(".") if r.strip()]
            if not reactants:
                continue

            reward = reward_calc.combined_reward(target_smiles, reactants, stock_list)
            scored_candidates.append((reward, reactants))
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "target": target_smiles,
                "routes": [],
                "best_score": 0.0,
                "stats": {"simulations": 0, "time_seconds": elapsed, "routes_found": 0},
                "molecules": [],
                "error": f"HF Inference API error: {e}",
            }

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
    parser = argparse.ArgumentParser(description="HuggingFace retrosynthesis inference")
    parser.add_argument("--target", required=True, help="Target SMILES string")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model/adapter repo ID")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN)")
    parser.add_argument("--n", type=int, default=3, help="Number of candidates")
    args = parser.parse_args()

    reward_calc = RewardCalculator()
    stock = StockList()
    stock.load()
    result = run_inference_hf(
        args.target, reward_calc, stock, model_id=args.model, token=args.token, n_candidates=args.n
    )
    print(json.dumps(result, indent=2, default=str))
