"""Prime Intellect inference client for retrosynthesis.

Calls a deployed LoRA adapter via OpenAI-compatible API to predict
retrosynthetic reactants, then scores results with RDKit rewards.

Usage:
    # As module (from Streamlit):
    from scripts.inference_pi import create_pi_client, run_inference_pi
    client = create_pi_client()
    result = run_inference_pi("CC(=O)Oc1ccccc1C(=O)O", client, "model-id", reward_calc, stock)

    # As CLI:
    python scripts/inference_pi.py --target "CC(=O)Oc1ccccc1C(=O)O" --model <deployment-id>
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from rdkit import Chem

from data.stock.loader import StockList
from env.Rewards import RewardCalculator
from scripts.inference import mol_to_base64_image

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


def create_pi_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """Create an OpenAI-compatible client for Prime Intellect's API.

    Args:
        api_key: API key for authentication. Defaults to PRIME_API_KEY env var.
        base_url: Base URL for the API. Defaults to Prime Intellect's endpoint.

    Returns:
        Configured OpenAI client instance.
    """
    if api_key is None:
        api_key = os.environ.get("PRIME_API_KEY", "")
    if base_url is None:
        base_url = "https://api.pinference.ai/api/v1"
    return OpenAI(base_url=base_url, api_key=api_key)


def run_inference_pi(
    target_smiles: str,
    client: OpenAI,
    model_id: str,
    reward_calc: RewardCalculator,
    stock_list: StockList,
    system_prompt: str = SYSTEM_PROMPT,
    n_candidates: int = 3,
) -> dict:
    """Run retrosynthetic inference via Prime Intellect's deployed model.

    Calls the OpenAI-compatible chat completions endpoint, parses predicted
    reactants from each candidate response, scores them with the reward
    calculator, and returns the top-3 routes in the standard result format.

    Args:
        target_smiles: SMILES string of the target molecule.
        client: OpenAI client configured for Prime Intellect.
        model_id: Deployment / model ID on Prime Intellect.
        reward_calc: RewardCalculator instance for scoring predictions.
        stock_list: StockList instance (pre-loaded) for buyability checks.
        system_prompt: System prompt for the chat model.
        n_candidates: Number of candidate completions to request.

    Returns:
        Dict matching the structure from scripts/inference.py:
        {
            "target": str,
            "routes": [...],
            "best_score": float,
            "stats": {"simulations": int, "time_seconds": float, "routes_found": int},
            "molecules": [{"smiles": str, "sascore": float, "in_stock": bool, "image_b64": str}]
        }
    """
    # Validate target SMILES
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return {
            "target": target_smiles,
            "routes": [],
            "best_score": 0.0,
            "stats": {
                "simulations": 0,
                "time_seconds": 0.0,
                "routes_found": 0,
            },
            "molecules": [],
            "error": f"Invalid SMILES: {target_smiles}",
        }

    # Call the API
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Predict the reactants for: {target_smiles}"},
            ],
            max_tokens=256,
            temperature=0.7,
            n=n_candidates,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "target": target_smiles,
            "routes": [],
            "best_score": 0.0,
            "stats": {
                "simulations": 0,
                "time_seconds": elapsed,
                "routes_found": 0,
            },
            "molecules": [],
            "error": f"API error: {e}",
        }
    elapsed = time.time() - start_time

    # Parse and score each candidate
    scored_candidates: list[tuple[float, list[str]]] = []
    for choice in response.choices:
        raw_text = choice.message.content
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

    # Rank by reward descending, pick top 3
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:3]

    # Build route trees (single-step: target -> reactants)
    routes = []
    for reward, reactants in top_candidates:
        children = []
        for r in reactants:
            buyable = stock_list.is_buyable(r)
            children.append(
                {
                    "smiles": r,
                    "score": 1.0 if buyable else 0.0,
                    "in_stock": buyable,
                    "children": [],
                }
            )
        routes.append(
            {
                "smiles": target_smiles,
                "score": reward,
                "in_stock": False,
                "children": children,
            }
        )

    best_score = routes[0]["score"] if routes else 0.0

    # Collect all unique molecules from routes for the molecules list
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
            "simulations": n_candidates,
            "time_seconds": elapsed,
            "routes_found": len(routes),
        },
        "molecules": molecules,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prime Intellect retrosynthesis inference")
    parser.add_argument("--target", required=True, help="Target SMILES string")
    parser.add_argument("--model", required=True, help="Deployment/model ID")
    parser.add_argument("--api-key", default=None, help="API key (or set PRIME_API_KEY)")
    parser.add_argument("--n", type=int, default=3, help="Number of candidates")
    args = parser.parse_args()

    client = create_pi_client(api_key=args.api_key)
    reward_calc = RewardCalculator()
    stock = StockList()
    stock.load()
    result = run_inference_pi(
        args.target, client, args.model, reward_calc, stock, n_candidates=args.n
    )
    print(json.dumps(result, indent=2, default=str))
