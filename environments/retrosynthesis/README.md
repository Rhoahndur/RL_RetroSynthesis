# retrosynthesis

## Overview

- **Environment ID**: `retrosynthesis`
- **Type**: single-turn `vf.SingleTurnEnv`
- **Purpose**: train and evaluate a model that predicts reactant SMILES for a target product SMILES.
- **Tags**: single-turn, chemistry, retrosynthesis, train, eval

The environment entrypoint is `load_environment(...)` in `retrosynthesis.py`.
It returns a Verifiers `SingleTurnEnv` with a chemistry-specific system prompt,
a dataset of target molecules, and an async RDKit reward rubric.

## Datasets

- **Primary dataset**: `rhoahndur/retrosyn-targets` on Hugging Face Hub.
- **Source data**: USPTO-50K retrosynthesis examples prepared by the project scripts.
- **Fallback data**: 24 inline molecules for training and 4 demo molecules for test/eval when the Hugging Face dataset is unavailable.
- **Stock data**: bundled `data/buyables.smi.gz`, derived from the project buyables stock list.

Each row is converted to:

- `question`: `Predict the reactants for: <product_smiles>`
- `answer`: known reactant SMILES when available
- `info`: JSON with `product_smiles`

## Task

The model receives one target product SMILES and should output only reactant
SMILES separated by dots, for example:

```text
OC(=O)c1ccccc1O.CC(=O)OC(C)=O
```

The output should not include explanations, markdown, XML, JSON, or reasoning
text. Invalid fragments are penalized by the rubric.

## Rubric

The environment uses six weighted async reward functions:

| Reward | Weight | Meaning |
| --- | ---: | --- |
| `attempt_reward` | 0.10 | Non-empty, reasonable-length output |
| `format_reward` | 0.10 | Output contains SMILES-like characters only |
| `validity_reward` | 0.25 | Fraction of fragments RDKit can parse |
| `sascore_reward` | 0.15 | Reactants are synthetically simpler than product |
| `stock_reward` | 0.25 | Exact or Tanimoto-soft stock-list match |
| `atom_conservation_reward` | 0.15 | Product atoms are covered with byproduct-aware excess penalty |

Reward floors are intentionally used for non-empty attempts to reduce collapse
to empty completions during GRPO training.

## Quickstart

Install the environment from the lab workspace:

```bash
prime env install retrosynthesis
```

Run a small evaluation:

```bash
prime eval run retrosynthesis -m gpt-4.1-mini -n 4 -r 1
```

Run with explicit environment args:

```bash
prime eval run retrosynthesis \
  -m gpt-4.1-mini \
  -n 20 \
  -r 3 \
  -t 1024 \
  -T 0.7 \
  -a '{"split": "test", "difficulty": "all"}'
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `split` | str | `"train"` | Dataset split to load for the main dataset. Use `"test"` for held-out eval rows. |
| `difficulty` | str | `"all"` | Reserved for future filtering; currently accepted but not applied. |

`load_environment()` also accepts extra keyword arguments for Prime/Verifiers
compatibility; unknown args are ignored.

## Metrics

Verifiers reports the weighted scalar `reward` and reward-function component
metrics. Interpret higher scores as better chemical format, validity,
synthetic simplicity, stock availability, and atom conservation. This rubric is
not a substitute for expert chemistry review.

## Development Notes

- Keep this package self-contained under `environments/retrosynthesis/`.
- Use `prime eval run retrosynthesis ...` as the canonical local validation path.
- Push only after local eval behavior is verified:

```bash
prime env push --path ./environments/retrosynthesis
```
