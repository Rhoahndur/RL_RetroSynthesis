"""Push LoRA adapter to HuggingFace Hub for inference.

Uploads the LoRA adapter files directly — no merging needed locally.
HF Serverless Inference can load the base model + adapter on their GPUs.

If you have enough RAM (~16GB free), use --merge to merge the adapter into
the base model first, which makes inference faster on the Hub.

Usage:
    # Push adapter as-is (works on any machine)
    python scripts/merge_and_push.py

    # Merge first then push (needs ~16GB RAM)
    python scripts/merge_and_push.py --merge

    # Custom repo ID
    python scripts/merge_and_push.py --repo-id your-username/your-model-name
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def push_adapter(adapter_path: Path, repo_id: str):
    """Push LoRA adapter directly to Hub (lightweight, works on any machine)."""
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)

    # Upload adapter files
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        commit_message="Upload retrosynthesis LoRA adapter (Qwen3-4B)",
    )

    # Create a model card
    card_content = """---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
tags:
  - chemistry
  - retrosynthesis
  - reinforcement-learning
  - lora
license: apache-2.0
---

# Retrosynthesis Qwen3-4B LoRA

LoRA adapter for retrosynthetic route prediction, trained via GRPO on Prime Intellect.

**Base model**: Qwen/Qwen3-4B-Instruct-2507
**Training**: GRPO with 6-component RDKit reward rubric (validity, SA score, stock match, atom conservation)
**Dataset**: USPTO-50K via [rhoahndur/retrosyn-targets](https://huggingface.co/datasets/rhoahndur/retrosyn-targets)

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = PeftModel.from_pretrained(base, "rhoahndur/retrosynthesis-qwen3-4b")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

prompt = "Predict the reactants for: CC(=O)Oc1ccccc1C(=O)O"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Demo

[Retrosynthesis AI on HuggingFace Spaces](https://huggingface.co/spaces/rhoahndur/retrosynthesis-ai)
"""
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )

    print(f"\nDone! Adapter pushed to: https://huggingface.co/{repo_id}")


def merge_and_push(adapter_path: Path, repo_id: str):
    """Merge LoRA into base model and push (needs ~16GB RAM)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_id = "Qwen/Qwen3-4B-Instruct-2507"
    output_dir = adapter_path.parent / "merged"

    print(f"[1/4] Loading base model {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("[2/4] Applying LoRA adapter and merging...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"[3/4] Saving merged model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"    Saved: {total_bytes / 1e9:.1f} GB")

    print(f"[4/4] Pushing to {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        commit_message="Upload merged Qwen3-4B + retrosynthesis LoRA",
    )
    print(f"\nDone! Model at: https://huggingface.co/{repo_id}")
    print(f"Tip: Delete {output_dir} to free ~{total_bytes / 1e9:.0f}GB disk space.")


def main():
    parser = argparse.ArgumentParser(description="Push LoRA adapter to HuggingFace Hub")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "models" / "lora_adapter"),
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="rhoahndur/retrosynthesis-qwen3-4b",
        help="HuggingFace Hub repo ID",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge adapter into base model before pushing (needs ~16GB RAM)",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    if not (adapter_path / "adapter_config.json").exists():
        print(f"ERROR: No adapter found at {adapter_path}")
        return

    if args.merge:
        merge_and_push(adapter_path, args.repo_id)
    else:
        push_adapter(adapter_path, args.repo_id)


if __name__ == "__main__":
    main()
