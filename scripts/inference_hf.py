"""GGUF CPU inference for retrosynthesis via pre-built llama.cpp binary.

Downloads the llama.cpp CLI binary and GGUF model from HuggingFace Hub,
runs inference as a subprocess. No compilation, no GPU, no API key.

Usage:
    from scripts.inference_hf import run_inference_hf
    result = run_inference_hf("CC(=O)Oc1ccccc1C(=O)O", reward_calc, stock)
"""

import json
import re
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download
from rdkit import Chem

from data.stock.loader import StockList
from env.Rewards import RewardCalculator
from scripts.inference import mol_to_base64_image

GGUF_REPO = "rhoahndur/retrosynthesis-qwen3-4b-gguf"
GGUF_FILE = "retrosynthesis-qwen3-4b-Q4_K_M.gguf"
LLAMA_RELEASE = "b5560"
LLAMA_TAR = f"llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/{LLAMA_TAR}"

CACHE_DIR = Path.home() / ".cache" / "retrosyn-llama"

SYSTEM_PROMPT = (
    "You are a retrosynthesis expert. Given a target molecule as a SMILES string, "
    "predict the reactant molecules that can be combined to synthesize the target.\n\n"
    "Rules:\n"
    "- Output ONLY the reactant SMILES strings separated by '.'\n"
    "- Do NOT include any explanation, reasoning, or extra text\n"
    "- Each reactant must be a valid SMILES string\n"
    "- Prefer simpler, commercially available starting materials\n\n"
    "Example:\n"
    "Input: CC(=O)Oc1ccccc1C(=O)O\n"
    "Output: OC(=O)c1ccccc1O.CC(=O)OC(C)=O"
)


def _get_llama_binary() -> Path:
    """Download and extract pre-built llama-cli binary."""
    binary = CACHE_DIR / "llama-cli"
    if binary.exists():
        return binary
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading llama.cpp binary ({LLAMA_RELEASE})...")
    tar_path = CACHE_DIR / LLAMA_TAR
    subprocess.check_call(
        ["curl", "-L", "-o", str(tar_path), LLAMA_URL],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if member.name.endswith("/llama-cli"):
                member.name = "llama-cli"
                tar.extract(member, CACHE_DIR)
                break
    tar_path.unlink()
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    print(f"llama-cli ready at {binary}")
    return binary


def _get_model_path() -> Path:
    """Download GGUF model from HuggingFace Hub."""
    print("Ensuring GGUF model is downloaded...")
    path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    print(f"Model at {path}")
    return Path(path)


def _run_llama(binary: Path, model_path: Path, prompt: str, temperature: float = 0.7) -> str:
    """Run a single inference call via llama-cli subprocess."""
    result = subprocess.run(
        [
            str(binary),
            "-m",
            str(model_path),
            "-c",
            "512",
            "-n",
            "256",
            "--temp",
            str(temperature),
            "--top-p",
            "0.9",
            "-p",
            prompt,
            "--no-display-prompt",
            "-t",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip()


def run_inference_hf(
    target_smiles: str,
    reward_calc: RewardCalculator,
    stock_list: StockList,
    n_candidates: int = 3,
) -> dict:
    """Run retrosynthetic inference via quantized GGUF model on CPU."""
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
        binary = _get_llama_binary()
        model_path = _get_model_path()
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
            "error": f"Setup error: {e}",
        }

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nPredict the reactants for: {target_smiles}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    scored_candidates: list[tuple[float, list[str]]] = []
    for _ in range(n_candidates):
        try:
            raw = _run_llama(binary, model_path, prompt)
            # Strip thinking tags and clean up
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"<\|im_end\|>.*", "", raw).strip()
            if not raw:
                continue
            reactants = [r.strip() for r in raw.split(".") if r.strip()]
            if not reactants:
                continue
            reward = reward_calc.combined_reward(target_smiles, reactants, stock_list)
            scored_candidates.append((reward, reactants))
        except Exception:
            continue

    elapsed = time.time() - start_time

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:3]

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
    import argparse

    parser = argparse.ArgumentParser(description="GGUF CPU retrosynthesis inference")
    parser.add_argument("--target", required=True, help="Target SMILES string")
    parser.add_argument("--n", type=int, default=3, help="Number of candidates")
    args = parser.parse_args()

    reward_calc = RewardCalculator()
    stock = StockList()
    stock.load()
    result = run_inference_hf(args.target, reward_calc, stock, n_candidates=args.n)
    print(json.dumps(result, indent=2, default=str))
