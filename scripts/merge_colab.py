"""Google Colab script to merge LoRA adapter and push to HuggingFace Hub.

IMPORTANT: Use a GPU runtime to avoid OOM.
  Runtime → Change runtime type → T4 GPU

Steps:
  1. Go to https://colab.research.google.com
  2. Create a new notebook
  3. Runtime → Change runtime type → T4 GPU
  4. Paste this entire script into a cell
  5. Update HF_TOKEN below
  6. Run the cell (~10 min)
"""

# ---- CONFIG (edit these) ----
HF_TOKEN = "hf_YOUR_TOKEN_HERE"  # <-- paste your HF write token
REPO_ID = "rhoahndur/retrosynthesis-qwen3-4b"
ADAPTER_REPO = "rhoahndur/retrosynthesis-qwen3-4b"  # where adapter currently lives
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# ---- END CONFIG ----

# Install dependencies
import subprocess

subprocess.check_call(
    ["pip", "install", "-q", "peft", "transformers", "accelerate", "huggingface_hub"]
)

import gc

import torch
from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print(
        f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
    )
else:
    device = "cpu"
    print("WARNING: No GPU detected. Go to Runtime → Change runtime type → T4 GPU")

# Download the adapter from Hub
print("[1/6] Downloading LoRA adapter...")
adapter_path = snapshot_download(ADAPTER_REPO, token=HF_TOKEN)
print(f"    Downloaded to: {adapter_path}")

# Load base model onto GPU (keeps system RAM free for saving later)
print(f"[2/6] Loading base model {BASE_MODEL} to {device}...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map=device,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    token=HF_TOKEN,
)

print("[3/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)

# Merge LoRA
print("[4/6] Merging LoRA adapter into base model...")
model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.float16)
model = model.merge_and_unload()
param_count = sum(p.numel() for p in model.parameters())
print(f"    Merged model parameters: {param_count:,}")

# Save directly from GPU (safetensors copies tensors one at a time, no bulk CPU move)
print("[5/6] Saving merged model (from GPU, no CPU copy)...")
output_dir = "/tmp/merged_model"
model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained(output_dir)

# Free ALL model memory before upload
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Push to Hub
print(f"[6/6] Uploading to {REPO_ID}...")
api = HfApi(token=HF_TOKEN)
api.create_repo(REPO_ID, exist_ok=True)
api.upload_folder(
    folder_path=output_dir,
    repo_id=REPO_ID,
    commit_message="Upload merged Qwen3-4B + retrosynthesis LoRA adapter",
)

# Update model card
card = """---
base_model: Qwen/Qwen3-4B-Instruct-2507
tags:
  - chemistry
  - retrosynthesis
  - reinforcement-learning
license: apache-2.0
pipeline_tag: text-generation
---

# Retrosynthesis Qwen3-4B

Qwen3-4B fine-tuned for retrosynthetic route prediction via GRPO on Prime Intellect.

Given a target molecule SMILES, predicts reactant molecules that can synthesize it.

**Training**: GRPO with 6-component RDKit reward rubric (validity, SA score, stock match, atom conservation)
**Dataset**: USPTO-50K via [rhoahndur/retrosyn-targets](https://huggingface.co/datasets/rhoahndur/retrosyn-targets)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("rhoahndur/retrosynthesis-qwen3-4b")
tokenizer = AutoTokenizer.from_pretrained("rhoahndur/retrosynthesis-qwen3-4b")

messages = [
    {"role": "system", "content": "You are a retrosynthesis expert. Given a target molecule SMILES, predict reactant SMILES separated by dots. Output ONLY SMILES, no explanation."},
    {"role": "user", "content": "Predict the reactants for: CC(=O)Oc1ccccc1C(=O)O"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Demo

[Retrosynthesis AI on HuggingFace Spaces](https://huggingface.co/spaces/rhoahndur/retrosynthesis-ai)
"""
api.upload_file(
    path_or_fileobj=card.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    commit_message="Update model card for merged model",
)

print(f"\nDone! Model at: https://huggingface.co/{REPO_ID}")
print("HF Serverless Inference should be available within a few minutes.")
