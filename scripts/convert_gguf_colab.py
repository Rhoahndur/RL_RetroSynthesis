"""Google Colab script to convert merged model to GGUF and upload.

Steps:
  1. Go to https://colab.research.google.com
  2. Runtime → Change runtime type → T4 GPU
  3. Paste this into a cell
  4. Add HF_TOKEN as a Colab secret or environment variable
  5. Run (~15 min)
"""

# ---- CONFIG ----
MODEL_REPO = "rhoahndur/retrosynthesis-qwen3-4b"  # merged model
GGUF_REPO = "rhoahndur/retrosynthesis-qwen3-4b-gguf"  # where GGUF goes
QUANT = "Q4_K_M"  # good balance of quality vs size
# ---- END CONFIG ----

import os
import subprocess

# Install llama.cpp conversion tools
subprocess.check_call(["pip", "install", "-q", "huggingface_hub"])
subprocess.check_call(["pip", "install", "-q", "gguf", "numpy", "sentencepiece", "transformers"])

# Clone llama.cpp for the conversion script
if not os.path.exists("llama.cpp"):
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp"])
    # Build quantize tool
    subprocess.check_call(
        ["cmake", "-B", "llama.cpp/build", "-S", "llama.cpp", "-DCMAKE_BUILD_TYPE=Release"]
    )
    subprocess.check_call(
        [
            "cmake",
            "--build",
            "llama.cpp/build",
            "--config",
            "Release",
            "-t",
            "llama-quantize",
            "-j",
            "2",
        ]
    )

from huggingface_hub import HfApi, snapshot_download


def get_hf_token() -> str:
    """Read the Hugging Face write token from runtime secrets, not source code."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None

    if token:
        return token
    raise RuntimeError("Set HF_TOKEN as an environment variable or Colab secret before running.")


HF_TOKEN = get_hf_token()

# Download the merged model
print("[1/4] Downloading merged model...")
model_path = snapshot_download(MODEL_REPO, token=HF_TOKEN)
print(f"    Downloaded to: {model_path}")

# Convert to GGUF F16
print("[2/4] Converting to GGUF (F16)...")
f16_path = "/tmp/model-f16.gguf"
subprocess.check_call(
    [
        "python",
        "llama.cpp/convert_hf_to_gguf.py",
        model_path,
        "--outfile",
        f16_path,
        "--outtype",
        "f16",
    ]
)
f16_size = os.path.getsize(f16_path) / 1e9
print(f"    F16 GGUF: {f16_size:.1f} GB")

# Quantize
print(f"[3/4] Quantizing to {QUANT}...")
quant_path = f"/tmp/model-{QUANT}.gguf"
quantize_bin = "llama.cpp/build/bin/llama-quantize"
subprocess.check_call([quantize_bin, f16_path, quant_path, QUANT])
quant_size = os.path.getsize(quant_path) / 1e9
print(f"    {QUANT} GGUF: {quant_size:.1f} GB")

# Clean up F16 to free disk
os.remove(f16_path)

# Upload
print(f"[4/4] Uploading to {GGUF_REPO}...")
api = HfApi(token=HF_TOKEN)
api.create_repo(GGUF_REPO, exist_ok=True)
api.upload_file(
    path_or_fileobj=quant_path,
    path_in_repo=f"retrosynthesis-qwen3-4b-{QUANT}.gguf",
    repo_id=GGUF_REPO,
    commit_message=f"Upload {QUANT} GGUF quantization",
)

# Model card
card = f"""---
base_model: rhoahndur/retrosynthesis-qwen3-4b
tags:
  - chemistry
  - retrosynthesis
  - gguf
  - llama-cpp
license: apache-2.0
---

# Retrosynthesis Qwen3-4B GGUF

{QUANT} quantized version of [rhoahndur/retrosynthesis-qwen3-4b](https://huggingface.co/rhoahndur/retrosynthesis-qwen3-4b) for CPU inference via llama.cpp.

**Size**: {quant_size:.1f} GB
**Quantization**: {QUANT}

## Usage with llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="{GGUF_REPO}",
    filename="retrosynthesis-qwen3-4b-{QUANT}.gguf",
    n_ctx=512,
)
output = llm.create_chat_completion(
    messages=[
        {{"role": "system", "content": "You are a retrosynthesis expert. Output ONLY reactant SMILES separated by dots."}},
        {{"role": "user", "content": "Predict the reactants for: CC(=O)Oc1ccccc1C(=O)O"}}
    ],
    max_tokens=256,
    temperature=0.7,
)
print(output["choices"][0]["message"]["content"])
```

## Demo

[Retrosynthesis AI](https://huggingface.co/spaces/rhoahndur/retrosynthesis-ai)
"""
api.upload_file(
    path_or_fileobj=card.encode(),
    path_in_repo="README.md",
    repo_id=GGUF_REPO,
    commit_message="Add model card",
)

print(f"\nDone! GGUF at: https://huggingface.co/{GGUF_REPO}")
print(f"File: retrosynthesis-qwen3-4b-{QUANT}.gguf ({quant_size:.1f} GB)")
