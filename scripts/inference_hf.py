"""GGUF CPU inference for retrosynthesis via llama-server.

Runs a persistent llama-server process that loads the model once and serves
requests via an OpenAI-compatible HTTP API. No compilation, no GPU, no API key.

Usage:
    from scripts.inference_hf import run_inference_hf
    result = run_inference_hf("CC(=O)Oc1ccccc1C(=O)O", reward_calc, stock)
"""

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download
from rdkit import Chem

from data.stock.loader import StockList
from env.Rewards import RewardCalculator
from scripts.inference import mol_to_base64_image

GGUF_REPO = "rhoahndur/retrosynthesis-qwen3-4b-gguf"
GGUF_FILE = "retrosynthesis-qwen3-4b-Q4_K_M.gguf"
LLAMA_RELEASE = "b8508"
LLAMA_TAR = f"llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/{LLAMA_TAR}"

CACHE_DIR = Path.home() / ".cache" / "retrosyn-llama"
SERVER_PORT = 8090

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

_server_process = None


def _get_llama_binaries() -> Path:
    """Download and extract llama-server binary + shared libs."""
    server_bin = CACHE_DIR / "llama-server"
    libs_exist = any(CACHE_DIR.glob("*.so*")) if CACHE_DIR.exists() else False
    if server_bin.exists() and libs_exist:
        return server_bin
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading llama.cpp binaries ({LLAMA_RELEASE})...")
    tar_path = CACHE_DIR / LLAMA_TAR
    subprocess.check_call(
        ["curl", "-L", "-o", str(tar_path), LLAMA_URL],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            basename = Path(member.name).name
            if basename == "llama-server" or basename.endswith(".so") or ".so." in basename:
                member.name = basename
                tar.extract(member, CACHE_DIR)
    tar_path.unlink()
    server_bin.chmod(server_bin.stat().st_mode | stat.S_IEXEC)
    print(f"llama-server ready at {server_bin}")
    return server_bin


def _get_model_path() -> Path:
    """Download GGUF model from HuggingFace Hub."""
    print("Ensuring GGUF model is downloaded...")
    path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    print(f"Model at {path}")
    return Path(path)


def _ensure_server_running():
    """Start llama-server if not already running. Model loads once, stays in memory."""
    global _server_process

    # Check if server is already responding
    try:
        resp = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
        if resp.ok:
            return
    except Exception:
        pass

    # Kill stale process if any
    if _server_process is not None:
        _server_process.kill()
        _server_process = None

    server_bin = _get_llama_binaries()
    model_path = _get_model_path()

    env = {**os.environ, "LD_LIBRARY_PATH": str(server_bin.parent)}
    print(f"[llama-server] Starting on port {SERVER_PORT}...")
    _server_process = subprocess.Popen(
        [
            str(server_bin),
            "-m",
            str(model_path),
            "-c",
            "512",
            "-t",
            "2",
            "--port",
            str(SERVER_PORT),
            "--host",
            "127.0.0.1",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready (model loading)
    print("[llama-server] Loading model (this takes ~60s on first request)...")
    for i in range(180):  # up to 3 minutes for model loading
        try:
            resp = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
            if resp.ok:
                print(f"[llama-server] Ready after {i + 1}s")
                return
        except Exception:
            pass
        # Check if process died
        if _server_process.poll() is not None:
            stderr = _server_process.stderr.read().decode()[:500]
            _server_process = None
            raise RuntimeError(f"llama-server exited: {stderr}")
        time.sleep(1)

    raise RuntimeError("llama-server failed to start within 180s")


def _query_server(target_smiles: str, temperature: float = 0.7) -> str:
    """Send a chat completion request to the local llama-server."""
    resp = requests.post(
        f"http://localhost:{SERVER_PORT}/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Predict the reactants for: {target_smiles}"},
            ],
            "max_tokens": 128,
            "temperature": temperature,
            "top_p": 0.9,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def run_inference_hf(
    target_smiles: str,
    reward_calc: RewardCalculator,
    stock_list: StockList,
    n_candidates: int = 1,
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
        _ensure_server_running()
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
            "error": f"Server error: {e}",
        }

    scored_candidates: list[tuple[float, list[str]]] = []
    for i in range(n_candidates):
        try:
            print(f"[inference] candidate {i + 1}/{n_candidates}...")
            raw = _query_server(target_smiles)
            print(f"[inference] raw output: {raw[:200]!r}")
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"<\|im_end\|>.*", "", raw).strip()
            if not raw:
                print("[inference] empty after cleanup, skipping")
                continue
            reactants = [r.strip() for r in raw.split(".") if r.strip()]
            if not reactants:
                continue
            reward = reward_calc.combined_reward(target_smiles, reactants, stock_list)
            print(f"[inference] reactants={reactants}, reward={reward:.3f}")
            scored_candidates.append((reward, reactants))
        except Exception as e:
            print(f"[inference] error: {e}")
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
    parser.add_argument("--n", type=int, default=1, help="Number of candidates")
    args = parser.parse_args()

    reward_calc = RewardCalculator()
    stock = StockList()
    stock.load()
    result = run_inference_hf(args.target, reward_calc, stock, n_candidates=args.n)
    print(json.dumps(result, indent=2, default=str))
