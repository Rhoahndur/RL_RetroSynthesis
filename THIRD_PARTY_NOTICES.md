# Third-Party Notices

This project combines local source code with third-party chemistry libraries,
models, datasets, and generated artifacts. The root `LICENSE` applies to the
project source code in this repository. Third-party assets retain their own
upstream licenses and terms.

## Chemistry Libraries And Utilities

- **RDKit**: Used for SMILES parsing, canonicalization, fingerprints, molecular
  properties, and rendering. RDKit is distributed under a BSD-style license.
- **Synthetic Accessibility scorer**: `lib/sascorer/` and
  `environments/retrosynthesis/sascorer.py` vendor the Ertl-Schuffenhauer
  synthetic accessibility scorer and fragment score data used by RDKit
  community examples. Keep the files' upstream license expectations intact
  when redistributing.

## Models

- **ReactionT5v2 retrosynthesis model**:
  `sagawa/ReactionT5v2-retrosynthesis` is downloaded from Hugging Face at
  runtime for the local model path. Review that model card and license before
  redistribution or commercial use.
- **Qwen3 base and fine-tuned artifacts**: Prime-trained LoRA, merged model,
  and GGUF artifacts reference `Qwen/Qwen3-4B-Instruct-2507` and related
  Hugging Face repositories. Those model artifacts are governed by their
  upstream model card terms, not the repository MIT license.
- **llama.cpp / llama-server**: `scripts/inference_hf.py` downloads a pinned
  prebuilt llama.cpp release artifact and verifies its SHA-256 digest before
  extraction. llama.cpp is governed by its upstream license.

## Data

- **USPTO-50K / TDC retrosynthesis data**: `scripts/prepare_data.py` and
  `scripts/prepare_pi_dataset.py` use the TDC `USPTO-50k` retrosynthesis data
  and publish/use derived prompts through `rhoahndur/retrosyn-targets`.
  Confirm dataset terms before redistribution.
- **ASKCOS buyables**: `data/stock/buyables_full.smi.gz` and
  `environments/retrosynthesis/data/buyables.smi.gz` are derived from public
  ASKCOS buyables plus curated common building blocks. Confirm upstream data
  terms before redistribution.
- **Curated stock list**: `data/stock/buyables.csv` is a small curated set of
  common molecules used for tests and demos.

## Operational Notes

- Hugging Face and Prime Intellect credentials must be provided through
  environment variables or secret managers. Do not commit API tokens.
- Large downloaded model and binary caches are runtime artifacts and are not
  covered by this repository license file.
