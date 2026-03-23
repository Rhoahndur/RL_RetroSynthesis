# Architecture — Retrosynthesis RL MVP

## System Overview

```mermaid
graph TB
    subgraph User["User Interface"]
        ST["Streamlit App<br/>app/main.py"]
        P3D["py3Dmol 3D Viewer"]
        RDK_DRAW["RDKit 2D Mol Images"]
    end

    subgraph Inference["Inference Layer"]
        INF["inference.py<br/>Local: ReactionT5 + MCTS"]
        INF_PI["inference_pi.py<br/>PI: OpenAI API + reward scoring"]
        MCTS["MCTS Engine<br/>env/MCTS.py"]
    end

    subgraph RL["RL Training Layer"]
        TRAIN["train_rl.py<br/>REINFORCE + GRPO advantage"]
        CKPT["Checkpoint Manager<br/>Save/load/prune .pt files"]
    end

    subgraph Eval["Evaluation Layer"]
        EVAL_TK["eval_topk.py<br/>Top-K exact match + SA buckets"]
        EVAL_MC["eval_mcts.py<br/>Full-route success rate"]
    end

    subgraph Core["Core Modules"]
        POLICY["RetroPolicy<br/>models/policy.py"]
        REWARD["RewardCalculator<br/>env/Rewards.py"]
        ENV["ChemEnv<br/>env/ChemEnv.py"]
        STOCK["StockList<br/>data/stock/loader.py"]
    end

    subgraph Model["Pre-trained Model"]
        RT5["ReactionT5v2<br/>sagawa/ReactionT5v2-retrosynthesis<br/>HuggingFace"]
    end

    subgraph Chem["Chemistry Engine"]
        RDKIT["RDKit<br/>Validity · Sanitization · SAscore<br/>Mol rendering · Canonicalization<br/>Morgan FP · Tanimoto similarity"]
    end

    subgraph Data["Data Layer"]
        CSV["buyables.csv<br/>246 common reagents"]
        TDC["TDC / USPTO-50K<br/>Training targets"]
        TARGETS["training_targets.csv<br/>Diverse product SMILES"]
        CKPT_DIR["models/checkpoints/<br/>step_N_reward_R.pt"]
    end

    subgraph Infra["Infrastructure"]
        MAC["MacBook Air M3<br/>Local dev + testing"]
        PI["Prime Intellect<br/>GRPO RL Training (managed)"]
        HFS["HuggingFace Spaces<br/>Streamlit deployment"]
    end

    %% User → Inference
    ST -->|"SMILES input"| INF
    INF -->|"route tree + images"| ST
    ST --- P3D
    ST --- RDK_DRAW

    %% Inference internals
    INF --> MCTS
    MCTS -->|"expand nodes"| POLICY
    MCTS -->|"score nodes"| REWARD
    MCTS -->|"check terminal"| STOCK

    %% Training internals
    TRAIN -->|"generate reactants"| POLICY
    TRAIN -->|"compute reward"| REWARD
    TRAIN -->|"check stock"| STOCK
    TRAIN -->|"save/load"| CKPT
    TRAIN -->|"sample targets"| TARGETS

    %% Eval internals
    EVAL_TK -->|"score predictions"| REWARD
    EVAL_MC -->|"search routes"| MCTS
    EVAL_MC -->|"check stock"| STOCK

    %% ChemEnv wiring
    ENV --> POLICY
    ENV --> REWARD
    ENV --> STOCK

    %% Core → Foundation
    POLICY -->|"wraps"| RT5
    REWARD -->|"validity + SAscore"| RDKIT
    STOCK -->|"canonicalize + fingerprints"| RDKIT
    RDK_DRAW -->|"MolToImage"| RDKIT
    STOCK -->|"loads"| CSV

    %% Checkpoints
    CKPT -->|"write"| CKPT_DIR
    INF -->|"read"| CKPT_DIR

    %% Data
    TDC -->|"prepare_data.py"| TARGETS

    %% Infra
    MAC -.->|"dev + test"| ST
    PI -.->|"train + serve"| TRAIN
    PI -.->|"SSH tunnel"| ST
```

## Data Flow: Training

```mermaid
sequenceDiagram
    participant D as Training Data
    participant T as train_rl.py
    participant P as RetroPolicy
    participant R as RewardCalculator
    participant S as StockList
    participant C as Checkpoints

    loop Every training step
        D->>T: Sample batch of target SMILES
        T->>P: predict(target, k candidates, temperature)
        P-->>T: k candidate reactant SMILES per target
        loop For each candidate
            T->>R: combined_reward(target, reactants)
            R->>S: is_buyable(reactant) + nearest_similarity()
            S-->>R: buyable / Tanimoto score
            R-->>T: reward ∈ [0, 1]
        end
        Note over T: group_mean = mean(rewards per target)
        loop For each candidate
            T->>P: log_prob(target, candidate)
            P-->>T: log π(a|s)
            Note over T: loss += -log_prob × (reward - group_mean)
        end
        T->>P: backward() + optimizer.step()
    end

    T->>C: save checkpoint every 500 steps
    T->>C: save on reward improvement
```

## Data Flow: Inference (Demo)

```mermaid
sequenceDiagram
    participant U as User (Streamlit)
    participant I as inference.py
    participant M as MCTS
    participant P as RetroPolicy
    participant R as RewardCalculator
    participant S as StockList

    U->>I: target SMILES (e.g. Ibuprofen)
    I->>M: search(target, time_budget=60s)

    loop Up to max_simulations
        M->>M: SELECT node via UCT
        M->>P: predict_greedy(node.smiles)
        P-->>M: top-K reactant sets
        M->>M: EXPAND: add children
        M->>R: score new nodes
        R->>S: check buyable status
        S-->>R: True/False
        R-->>M: node rewards
        M->>M: SIMULATE: rollout to depth
        M->>M: BACKPROPAGATE values
    end

    M-->>I: MCTSResult (top-3 routes)
    I-->>U: route tree + molecule images + scores
```

## MCTS Tree Structure

```mermaid
graph TD
    ROOT["🎯 Ibuprofen<br/>CC(C)Cc1ccc(cc1)C(C)C(=O)O<br/>SAscore: 2.8"]

    ROOT --> R1["Reactant Set A"]
    ROOT --> R2["Reactant Set B"]

    R1 --> A1["4-Isobutylacetophenone<br/>SAscore: 2.1"]
    R1 --> A2["CO<br/>methanol ✅<br/>IN STOCK"]

    R2 --> B1["p-Isobutylbenzaldehyde<br/>SAscore: 2.3"]
    R2 --> B2["CH3MgBr<br/>✅ IN STOCK"]

    A1 --> C1["Isobutylbenzene<br/>✅ IN STOCK"]
    A1 --> C2["Acetyl chloride<br/>✅ IN STOCK"]

    B1 --> D1["Isobutylbenzene<br/>✅ IN STOCK"]
    B1 --> D2["DMF + POCl3<br/>✅ IN STOCK"]

    style A2 fill:#4ade80,color:#000
    style B2 fill:#4ade80,color:#000
    style C1 fill:#4ade80,color:#000
    style C2 fill:#4ade80,color:#000
    style D1 fill:#4ade80,color:#000
    style D2 fill:#4ade80,color:#000
    style A1 fill:#fbbf24,color:#000
    style B1 fill:#fbbf24,color:#000
    style ROOT fill:#f87171,color:#000
```

## Reward Function Breakdown

```mermaid
graph LR
    subgraph Input
        PROD["Product SMILES"]
        REACT["Reactant SMILES"]
    end

    subgraph Rewards["Reward Components (weighted)"]
        V["Validity<br/>RDKit parse<br/>weight: 0.3"]
        P["Plausibility<br/>RDKit sanitize<br/>weight: 0.2"]
        SA["SAscore Δ<br/>simpler = better<br/>weight: 0.2"]
        BUY["Stock Match<br/>exact + Tanimoto ≥ 0.6<br/>weight: 0.3"]
    end

    subgraph Gate["Soft Gate (multiplier)"]
        ATOM["Atom Conservation<br/>product atoms covered?<br/>× (0.5 + 0.5 × ratio)"]
    end

    subgraph Output
        R["Combined Reward<br/>∈ [0, 1]"]
    end

    REACT --> V
    REACT --> P
    PROD --> SA
    REACT --> SA
    REACT --> BUY
    PROD --> ATOM
    REACT --> ATOM

    V -->|"× 0.3"| R
    P -->|"× 0.2"| R
    SA -->|"× 0.2"| R
    BUY -->|"× 0.3"| R
    ATOM -->|"soft gate"| R
```

## Module Dependency Graph

```mermaid
graph BT
    RDKIT["RDKit"] --> STOCK["StockList<br/>data/stock/loader.py"]
    RDKIT --> REWARD["RewardCalculator<br/>env/Rewards.py"]
    HF["HuggingFace<br/>ReactionT5v2"] --> POLICY["RetroPolicy<br/>models/policy.py"]

    STOCK --> REWARD
    STOCK --> MCTS["MCTS<br/>env/MCTS.py"]
    STOCK --> TRAIN["train_rl.py"]
    STOCK --> ENV["ChemEnv<br/>env/ChemEnv.py"]

    REWARD --> MCTS
    REWARD --> TRAIN
    REWARD --> ENV

    POLICY --> MCTS
    POLICY --> TRAIN
    POLICY --> ENV

    MCTS --> INF["inference.py<br/>(local)"]
    POLICY --> INF

    REWARD --> INF_PI["inference_pi.py<br/>(PI API)"]
    STOCK --> INF_PI

    INF --> APP["Streamlit App<br/>app/main.py"]
    INF_PI --> APP

    REWARD --> EVAL_TK["eval_topk.py"]
    MCTS --> EVAL_MC["eval_mcts.py"]
    REWARD --> EVAL_MC
    STOCK --> EVAL_MC
```

## Infrastructure Topology

```mermaid
graph LR
    subgraph Local["MacBook Air M3"]
        DEV["VS Code / Terminal"]
        BROWSER["Browser"]
    end

    subgraph HFS["HuggingFace Spaces"]
        ST_PROD["Streamlit App<br/>2 vCPU / 16GB RAM"]
        RT5_PROD["ReactionT5v2<br/>(preloaded)"]
    end

    subgraph PI["Prime Intellect (Managed)"]
        GRPO["GRPO Trainer"]
        VLLM["vLLM Inference"]
        ORCH["Orchestrator"]
        VERENV["Verifiers Env<br/>rhoahndur/retrosynthesis"]
    end

    subgraph HFHub["HuggingFace Hub"]
        DS["rhoahndur/retrosyn-targets<br/>USPTO-50K dataset"]
        MODEL["sagawa/ReactionT5v2<br/>Pre-trained model"]
    end

    DEV -->|"git push"| HFS
    BROWSER --> ST_PROD
    ST_PROD --> RT5_PROD
    DEV -->|"prime rl run"| PI
    ORCH --> VLLM
    ORCH --> GRPO
    VERENV -->|"loads dataset"| DS
    HFS -->|"preloads model"| MODEL
```

## File Map

```mermaid
graph TD
    subgraph Root["/retrosyn-rl"]
        REQ["requirements.txt"]
        PLAN["PLAN.md"]
        TASKS["TASKS.md"]
        ARCH["ARCHITECTURE.md"]
    end

    subgraph data["data/"]
        subgraph raw["raw/"]
            USPTO["USPTO-50K (via TDC)"]
        end
        subgraph processed["processed/"]
            TTARGETS["training_targets.csv"]
            VTARGETS["validation_targets.csv"]
        end
        subgraph stock["stock/"]
            BUY_CSV["buyables.csv"]
            LOADER["loader.py → StockList"]
        end
    end

    subgraph env["env/"]
        REWARDS_F["Rewards.py → RewardCalculator"]
        CHEMENV_F["ChemEnv.py → ChemEnv"]
        MCTS_F["MCTS.py → MCTS, MCTSNode"]
    end

    subgraph models["models/"]
        POLICY_F["policy.py → RetroPolicy"]
        subgraph ckpts["checkpoints/"]
            CK1["step_500_reward_0.42.pt"]
            CK2["step_1000_reward_0.58.pt"]
            BEST["best_reward_0.71.pt"]
        end
    end

    subgraph scripts["scripts/"]
        TRAIN_F["train_rl.py"]
        INF_F["inference.py"]
        INF_PI_F["inference_pi.py"]
        EVAL_TK_F["eval_topk.py"]
        EVAL_MC_F["eval_mcts.py"]
        PREP_F["prepare_data.py"]
        PREP_PI_F["prepare_pi_dataset.py"]
        SETUP_F["setup_prime.sh"]
    end

    subgraph app["app/"]
        MAIN_F["main.py (Streamlit)"]
    end

    subgraph tests["tests/ (95 tests)"]
        T_STOCK["test_stock_list.py"]
        T_REWARDS["test_rewards.py"]
        T_MCTS["test_mcts.py"]
        T_CHEMENV["test_chemenv.py"]
        T_POLICY["test_policy.py"]
        T_INF["test_inference.py"]
        T_TRAIN["test_train_helpers.py"]
        T_EVAL_TK["test_eval_topk.py"]
        T_EVAL_MC["test_eval_mcts.py"]
    end
```
