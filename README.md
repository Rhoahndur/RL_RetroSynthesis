---
title: Retrosynthesis AI
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.50.0"
python_version: "3.10"
app_file: app/main.py
pinned: false
models:
  - sagawa/ReactionT5v2-retrosynthesis
datasets:
  - rhoahndur/retrosyn-targets
preload_from_hub:
  - sagawa/ReactionT5v2-retrosynthesis
tags:
  - chemistry
  - retrosynthesis
  - reinforcement-learning
startup_duration_timeout: "1h"
---

# Retrosynthesis AI

RL-powered retrosynthetic route prediction. Given a target molecule as a SMILES string, the system predicts commercially available starting materials and synthesis routes using reinforcement learning, Monte Carlo Tree Search, and RDKit-based reward scoring.

## How It Works

1. **Input**: A target molecule SMILES (e.g. Ibuprofen: `CC(C)Cc1ccc(cc1)C(C)C(=O)O`)
2. **Search**: MCTS explores retrosynthetic disconnections guided by the RL policy
3. **Scoring**: Multi-component rewards evaluate validity, synthetic accessibility, and stock availability
4. **Output**: Ranked synthesis routes with molecule visualizations and buyability indicators

The system supports two inference backends:
- **Prime Intellect API** (primary) — RL-trained Qwen3-4B LoRA adapter via OpenAI-compatible endpoint
- **Local ReactionT5** (fallback) — `sagawa/ReactionT5v2-retrosynthesis` with MCTS

## Project Structure

```
├── app/
│   └── main.py                  # Streamlit web application
├── configs/
│   └── rl/
│       ├── retrosynthesis.toml          # Quick validation (50 steps, Qwen3-4B)
│       ├── retrosynthesis-full.toml     # Full training (300 steps, Qwen3-30B)
│       └── retrosynthesis-continue.toml # Resume from checkpoint
├── data/
│   └── stock/
│       ├── buyables.csv             # 246 commercially available molecules
│       ├── buyables_full.smi.gz     # ~204k ASKCOS buyables (expanded stock)
│       └── loader.py                # StockList — O(1) lookup + fingerprint similarity
├── env/
│   ├── ChemEnv.py               # Gym-style step-based RL environment
│   ├── MCTS.py                  # Monte Carlo Tree Search (UCT selection)
│   └── Rewards.py               # Multi-objective reward calculator
├── environments/
│   └── retrosynthesis/
│       ├── retrosynthesis.py        # Verifiers environment for Prime Intellect RL
│       ├── pyproject.toml           # Environment package config
│       ├── sascorer.py              # Vendored Ertl-Schuffenhauer SA scorer
│       ├── fpscores.pkl.gz          # SA scorer fragment data
│       └── data/
│           └── buyables.smi.gz      # ~204k ASKCOS buyables (bundled for PI)
├── lib/
│   └── sascorer/                    # Vendored Ertl-Schuffenhauer SA scorer (BSD)
│       ├── sascorer.py
│       └── fpscores.pkl.gz
├── models/
│   ├── policy.py                    # RetroPolicy — ReactionT5 wrapper with RL interface
│   └── checkpoints/                 # Saved .pt files (gitignored)
├── scripts/
│   ├── inference.py                 # Local MCTS inference
│   ├── inference_pi.py              # Prime Intellect API inference
│   ├── train_rl.py                  # REINFORCE training loop (GRPO-style)
│   ├── eval_topk.py                 # Top-K exact match evaluation
│   ├── eval_mcts.py                 # MCTS full-route success rate evaluation
│   ├── prepare_data.py              # Download/process USPTO-50K via TDC
│   ├── prepare_pi_dataset.py        # Format dataset for HuggingFace Hub upload
│   ├── prepare_stock.py             # Download/canonicalize ASKCOS buyables
│   └── setup_prime.sh               # Prime Intellect pod provisioning script
└── tests/                           # 100 unit tests
```

## Quickstart

### Prerequisites

- Python 3.10+
- [RDKit](https://www.rdkit.org/) (installed via `rdkit-pypi`)

### Install

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app/main.py
```

The app launches with preset buttons for four demo molecules:

| Molecule | SMILES |
|---|---|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Acetaminophen | `CC(=O)Nc1ccc(O)cc1` |
| Caffeine | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` |
| Ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` |

Select "Prime Intellect API" in the sidebar and provide your API key to use the RL-trained model, or select "Local Model (ReactionT5)" to run inference locally.

### Run Inference from CLI

```bash
# Local model
python scripts/inference.py --target "CC(=O)Oc1ccccc1C(=O)O"

# Prime Intellect API
python scripts/inference_pi.py --target "CC(=O)Oc1ccccc1C(=O)O" --model <deployment-id>
```

## Reward Functions

### Local Pipeline (ReactionT5 + MCTS)

Uses 4 weighted components via `env/Rewards.py`:

| Component | Weight | Description |
|---|---|---|
| **Validity** | 0.30 | All output SMILES parse as valid RDKit molecules |
| **Plausibility** | 0.20 | Molecules pass RDKit sanitization (no valency violations) |
| **SA Score** | 0.20 | Reactants are simpler than the target (Ertl-Schuffenhauer SA score) |
| **Stock Match** | 0.30 | Reactants match buyables (exact or Tanimoto similarity ≥ 0.6) |

Atom conservation acts as a soft multiplier — bidirectional check with byproduct awareness (recognizes H2O, CO2, AcOH, etc. as legitimate leaving groups).

### Prime Intellect Environment (GRPO)

Uses 6 async reward functions via the `verifiers` rubric in `environments/retrosynthesis/`:

| Component | Weight | Description |
|---|---|---|
| **Attempt** | 0.10 | Non-empty output with reasonable structure (prevents reward collapse) |
| **Format** | 0.10 | Output contains only SMILES characters, no explanatory text |
| **Validity** | 0.25 | Fraction of reactant fragments that parse as valid SMILES |
| **SA Score** | 0.15 | Reactants are simpler than the target (Ertl-Schuffenhauer, sigmoid-mapped) |
| **Stock Match** | 0.25 | Fraction of reactants matching ~204k ASKCOS buyables (exact or Tanimoto ≥ 0.6) |
| **Atom Conservation** | 0.15 | Bidirectional atom balance with byproduct awareness (H2O, CO2, etc.) |

All functions include reward floors (0.05-0.3 minimum for non-empty output) to prevent the model from collapsing to empty responses during GRPO training.

## Training

### Prime Intellect (GRPO + LoRA)

The primary training path uses Prime Intellect's managed RL platform with a custom `verifiers` environment.

```bash
# Install the verifiers environment locally
prime env install retrosynthesis

# Validate with a quick eval
prime eval run retrosynthesis -m gpt-4.1-mini -n 4 -r 1

# Push to Hub
prime env push --path ./environments/retrosynthesis

# Launch training (quick validation)
prime rl run configs/rl/retrosynthesis.toml

# Launch training (full run with Qwen3-30B + wandb)
prime rl run configs/rl/retrosynthesis-full.toml

# Monitor
prime rl logs <run-id> -f
```

After training, deploy the LoRA adapter:

```bash
prime deployments create <adapter-id>
```

### Local REINFORCE (fallback)

```bash
# Prepare USPTO-50K training data
python scripts/prepare_data.py

# Train locally
python scripts/train_rl.py --num_steps 5000 --batch_size 16

# Resume from checkpoint
python scripts/train_rl.py --resume models/checkpoints/<checkpoint>.pt
```

## Development

### Linting

```bash
make lint          # Check
make lint-fix      # Auto-fix + format
```

Uses [Ruff](https://docs.astral.sh/ruff/) (v0.8.6) with pycodestyle, pyflakes, isort, pyupgrade, flake8-bugbear, and flake8-simplify rules.

### Tests

```bash
make test          # Fast tests only (skips model downloads)
make test-all      # All tests including slow/GPU tests
```

100 tests across 9 test files covering the stock list, rewards, policy, MCTS, ChemEnv, inference, training helpers, and both evaluation scripts.

### CI

GitHub Actions runs lint, fast tests, HuggingFace dataset verification, and eval smoke tests on every push/PR to `main`. A separate workflow auto-syncs the repo to HuggingFace Spaces on push to `main`.

## Architecture

- **StockList** (`data/stock/loader.py`) — Loads buyable molecules from CSV or gzipped SMILES (`.smi.gz`), canonicalizes all SMILES via RDKit, provides O(1) set lookup for buyability checks. Precomputes Morgan fingerprints for vectorized Tanimoto similarity via `BulkTanimotoSimilarity`. Expanded stock: ~204k ASKCOS compounds in `buyables_full.smi.gz`.
- **RewardCalculator** (`env/Rewards.py`) — Computes validity, plausibility, SA score delta (Ertl-Schuffenhauer via vendored `lib/sascorer`), stock match (with soft Tanimoto similarity), and bidirectional atom conservation with byproduct awareness. Atom conservation acts as a soft multiplier on the weighted sum.
- **RetroPolicy** (`models/policy.py`) — Wraps `sagawa/ReactionT5v2-retrosynthesis` (T5 seq2seq) with temperature sampling, log-probability computation, and checkpoint save/load for REINFORCE training.
- **MCTS** (`env/MCTS.py`) — Full Monte Carlo Tree Search with UCT selection, policy-guided expansion, reward-based simulation, backpropagation, and cycle detection. Finds multi-step routes to buyable starting materials.
- **ChemEnv** (`env/ChemEnv.py`) — Gym-style wrapper combining policy, rewards, and stock list into a step-based interface for episodic RL.
- **Verifiers Environment** (`environments/retrosynthesis/`) — Self-contained `vf.SingleTurnEnv` package for Prime Intellect hosted RL training. Loads USPTO-50K from HuggingFace Hub (`rhoahndur/retrosyn-targets`), falls back to 24 inline demo molecules. 6-component async RDKit reward rubric with ~204k bundled ASKCOS buyables, real Ertl-Schuffenhauer SA scoring, and bidirectional atom conservation.
- **eval_topk** (`scripts/eval_topk.py`) — Top-K exact match evaluation against ground-truth reactions, stratified by SA score difficulty (easy/medium/hard) with reaction type breakdown and blind spot flagging.
- **eval_mcts** (`scripts/eval_mcts.py`) — MCTS full-route success rate evaluation measuring how often complete synthesis routes (all leaves buyable) are found.

## Tech Stack

| Component | Technology |
|---|---|
| Retrosynthesis model | ReactionT5v2 (local) / Qwen3-4B (PI) |
| RL algorithm | REINFORCE (local) / GRPO (PI) |
| Chemistry engine | RDKit + vendored SA scorer (Ertl-Schuffenhauer) |
| Stock data | ASKCOS buyables (~204k compounds) |
| Training data | USPTO-50K via TDC |
| Training platform | Prime Intellect |
| Frontend | Streamlit + py3Dmol |
| Inference API | OpenAI-compatible (Prime Intellect) |
| CI/CD | GitHub Actions + ruff + pytest |
| Deployment | HuggingFace Spaces (auto-deploy from GitHub) |
| Linting | Ruff |
| Pre-commit | ruff check + ruff format |

## License

This project was built at a hackathon and is provided as-is for educational and research purposes.
