#!/bin/bash
# Prime Intellect pod setup script.
# Run once after provisioning an H100 pod to install all dependencies.
#
# Usage: bash scripts/setup_prime.sh

set -e

echo "=== Prime Intellect Pod Setup ==="
echo ""

# ---------------------------------------------------------------------------
# 1. apt update + install essentials
# ---------------------------------------------------------------------------
echo "[1/8] Installing system packages..."
apt-get update && apt-get install -y git wget curl build-essential
echo "  -> System packages installed."
echo ""

# ---------------------------------------------------------------------------
# 2. Install miniconda if not present, then create conda env
# ---------------------------------------------------------------------------
echo "[2/8] Setting up conda environment..."

if ! command -v conda &> /dev/null; then
    echo "  -> conda not found — installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm -f /tmp/miniconda.sh
    echo "  -> Miniconda installed."
else
    echo "  -> conda already available, skipping install."
fi

# Source conda so we can use 'conda activate' in this script
# (conda init only modifies .bashrc, which isn't re-sourced in a running script)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda.sh to source. Is conda installed?"
    exit 1
fi

# Create the env if it doesn't already exist (idempotent)
if conda env list | grep -q "retrosyn"; then
    echo "  -> Conda env 'retrosyn' already exists, skipping creation."
else
    echo "  -> Creating conda env 'retrosyn' with Python 3.10..."
    conda create -n retrosyn python=3.10 -y
fi

conda activate retrosyn
echo "  -> Activated conda env 'retrosyn' (Python $(python --version 2>&1))."
echo ""

# ---------------------------------------------------------------------------
# 3. Install PyTorch with CUDA support
# ---------------------------------------------------------------------------
echo "[3/8] Installing PyTorch with CUDA 12.1 support..."
pip install torch --index-url https://download.pytorch.org/whl/cu121
echo "  -> PyTorch installed."
echo ""

# ---------------------------------------------------------------------------
# 4. Install project requirements
# ---------------------------------------------------------------------------
echo "[4/8] Installing project requirements from requirements.txt..."
pip install -r requirements.txt
echo "  -> Requirements installed."
echo ""

# ---------------------------------------------------------------------------
# 5. Pre-download ReactionT5 model weights
# ---------------------------------------------------------------------------
echo "[5/8] Pre-downloading ReactionT5v2-retrosynthesis model weights..."
python -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
print('  -> Downloading tokenizer...')
AutoTokenizer.from_pretrained('sagawa/ReactionT5v2-retrosynthesis')
print('  -> Downloading model...')
AutoModelForSeq2SeqLM.from_pretrained('sagawa/ReactionT5v2-retrosynthesis')
print('  -> Model cached successfully.')
"
echo ""

# ---------------------------------------------------------------------------
# 6. Run data preparation
# ---------------------------------------------------------------------------
echo "[6/8] Running data preparation (USPTO-50K download + processing)..."
python scripts/prepare_data.py
echo "  -> Data preparation complete."
echo ""

# ---------------------------------------------------------------------------
# 7. Verify GPU access
# ---------------------------------------------------------------------------
echo "[7/8] Verifying GPU access..."
python -c "
import torch
if not torch.cuda.is_available():
    raise RuntimeError('NO GPU FOUND — torch.cuda.is_available() returned False')
print(f'  -> GPU detected: {torch.cuda.get_device_name(0)}')
print(f'  -> CUDA version: {torch.version.cuda}')
print(f'  -> PyTorch version: {torch.__version__}')
"
echo ""

# ---------------------------------------------------------------------------
# 8. Print system info
# ---------------------------------------------------------------------------
echo "[8/8] System info summary"
echo ""
echo "--- GPU ---"
nvidia-smi
echo ""
echo "--- Disk ---"
df -h .
echo ""
echo "--- Memory ---"
free -h
echo ""

echo "=== Setup Complete ==="
