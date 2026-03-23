"""REINFORCE training loop for retrosynthesis policy fine-tuning.

Single-step REINFORCE: sample target molecules, generate reactants via policy,
score with reward function, update policy via policy gradient.

Usage:
    # Local test (tiny):
    python scripts/train_rl.py --num_steps 5 --batch_size 2

    # Full training on Prime Intellect:
    python scripts/train_rl.py --num_steps 5000 --batch_size 16 --device cuda

    # Resume from checkpoint:
    python scripts/train_rl.py --resume models/checkpoints/checkpoint_step500_reward0.4200.pt
"""

import argparse
import glob
import os
import random
import re
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from data.stock.loader import StockList
from env.Rewards import RewardCalculator
from models.policy import RetroPolicy

FALLBACK_MOLECULES = [
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CC(=O)Oc1ccccc1C(=O)O",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REINFORCE training for retrosynthesis")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    parser.add_argument("--baseline_decay", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Candidates per target for group-relative advantage (GRPO-style)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--data_path", type=str, default="data/processed/training_targets.csv")
    parser.add_argument("--stock_path", type=str, default="data/stock/buyables.csv")
    return parser.parse_args()


def load_training_data(data_path: str) -> list[str]:
    """Load target molecule SMILES from CSV.

    Args:
        data_path: Path to training_targets.csv.

    Returns:
        List of SMILES strings.
    """
    if not os.path.exists(data_path):
        print(
            f"WARNING: Training data file not found at '{data_path}'. "
            f"Using fallback demo molecules ({len(FALLBACK_MOLECULES)} molecules)."
        )
        return list(FALLBACK_MOLECULES)

    df = pd.read_csv(data_path)

    # Look for a 'smiles' column (case-insensitive)
    smiles_col = None
    for col in df.columns:
        if col.lower() == "smiles":
            smiles_col = col
            break

    if smiles_col is not None:
        smiles_list = df[smiles_col].dropna().astype(str).tolist()
    elif len(df.columns) == 1:
        # Single column -- treat it as SMILES
        smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        # Fall back to first column
        print(f"WARNING: No 'smiles' column found in {data_path}. Using first column.")
        smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()

    # Filter out empty strings
    smiles_list = [s.strip() for s in smiles_list if s.strip()]

    if not smiles_list:
        print("WARNING: CSV was empty or contained no valid SMILES. Using fallback demo molecules.")
        return list(FALLBACK_MOLECULES)

    return smiles_list


def sample_batch(data: list[str], batch_size: int) -> list[str]:
    """Sample a random batch of target molecules.

    Args:
        data: List of SMILES strings.
        batch_size: Number of molecules to sample.

    Returns:
        List of SMILES strings.
    """
    return random.choices(data, k=batch_size)


def save_checkpoint(policy, optimizer, step, reward, best_reward, checkpoint_dir):
    """Save checkpoint and prune old ones (keep last 3 + best).

    Args:
        policy: RetroPolicy instance.
        optimizer: Optimizer instance.
        step: Current training step.
        reward: Current mean reward.
        best_reward: Best reward seen so far.
        checkpoint_dir: Directory to save checkpoints.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = f"checkpoint_step{step}_reward{reward:.4f}.pt"
    path = os.path.join(checkpoint_dir, filename)

    policy.save_checkpoint(path, step, reward, optimizer)
    print(f"  Saved checkpoint: {filename}")

    # Prune old checkpoints: keep last 3 by step + the one with best reward
    existing = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step*_reward*.pt"))

    if len(existing) <= 3:
        return

    # Parse (step_num, reward_val, filepath) for each checkpoint
    parsed = []
    pattern = re.compile(r"checkpoint_step(\d+)_reward([\d.]+)\.pt$")
    for fp in existing:
        basename = os.path.basename(fp)
        m = pattern.search(basename)
        if m:
            s = int(m.group(1))
            r = float(m.group(2))
            parsed.append((s, r, fp))

    if not parsed:
        return

    # Find the checkpoint with the best reward
    best_ckpt = max(parsed, key=lambda x: x[1])

    # Sort by step number descending, keep last 3
    sorted_by_step = sorted(parsed, key=lambda x: x[0], reverse=True)
    keep_last_3 = set(x[2] for x in sorted_by_step[:3])

    # Always keep the best reward checkpoint
    keep_last_3.add(best_ckpt[2])

    # Delete everything not in the keep set
    for _, _, fp in parsed:
        if fp not in keep_last_3:
            try:
                os.remove(fp)
                print(f"  Pruned old checkpoint: {os.path.basename(fp)}")
            except OSError:
                pass


def train(args: argparse.Namespace) -> None:
    """Main training loop.

    1. Load policy, reward calculator, stock list, training data.
    2. For each step: sample targets, predict reactants, compute reward, update policy.
    3. Checkpoint every N steps + on reward improvement.

    Args:
        args: Parsed command-line arguments.
    """
    # ---- 1. Setup ----
    # Detect device
    if args.device == "auto":
        device = RetroPolicy.detect_device()
    else:
        device = args.device

    print("=" * 60)
    print("REINFORCE Training for Retrosynthesis")
    print("=" * 60)

    # Load policy
    policy = RetroPolicy(device=device)

    # Load reward calculator
    reward_calc = RewardCalculator()

    # Load stock list
    stock = StockList()
    stock.load(args.stock_path)

    # Load training data
    training_data = load_training_data(args.data_path)

    # Setup optimizer
    optimizer = torch.optim.Adam(policy.get_model().parameters(), lr=args.lr)

    # Initialize tracking variables
    baseline = 0.0
    best_reward = -float("inf")
    start_step = 0

    # Resume from checkpoint if requested
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            meta = policy.load_checkpoint(args.resume)
            start_step = meta.get("step", 0) + 1
            best_reward = meta.get("reward", -float("inf"))
            if meta.get("has_optimizer"):
                # Reload optimizer state from checkpoint
                ckpt_data = torch.load(args.resume, map_location=device)
                if "optimizer_state_dict" in ckpt_data:
                    optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            print(f"  Resumed at step {start_step}, best_reward={best_reward:.4f}")
        else:
            print(f"WARNING: Checkpoint '{args.resume}' not found. Starting from scratch.")

    # Print header
    print("  Model:          ReactionT5v2-retrosynthesis")
    print(f"  Device:         {device}")
    print(f"  Dataset size:   {len(training_data)} molecules")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Num steps:      {args.num_steps}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Num samples:    {args.num_samples} per target (group-relative advantage)")
    print(f"  Baseline decay: {args.baseline_decay}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Checkpoint every: {args.checkpoint_every}")
    print("=" * 60)

    # ---- 2. Training loop ----
    for step in range(start_step, args.num_steps):
        step_start = time.time()

        # a. Sample batch
        targets = sample_batch(training_data, args.batch_size)

        # b. Accumulate loss over batch
        total_loss = torch.tensor(0.0, device=device, requires_grad=False)
        rewards_list = []
        validity_count = 0
        stock_count = 0
        valid_samples = 0

        for target in targets:
            # Generate k candidate predictions per target
            candidates = policy.predict(
                target, num_candidates=args.num_samples, temperature=args.temperature
            )

            if not candidates:
                continue

            # Compute rewards for all candidates
            sample_rewards = []
            for candidate in candidates:
                reactant_list = candidate.split(".")
                r = reward_calc.combined_reward(target, reactant_list, stock)
                sample_rewards.append(r)

            if not sample_rewards:
                continue

            # Group-relative advantage (GRPO-style):
            # advantage_i = reward_i - mean(rewards in group)
            group_mean = sum(sample_rewards) / len(sample_rewards)

            for candidate, r in zip(candidates, sample_rewards):
                log_p = policy.log_prob(target, candidate)
                advantage = r - group_mean
                loss = -log_p * advantage
                total_loss = total_loss + loss
                valid_samples += 1

            # Track statistics using best candidate from the group
            best_reward = max(sample_rewards)
            rewards_list.append(best_reward)
            best_candidate = candidates[sample_rewards.index(best_reward)]
            best_reactant_list = best_candidate.split(".")

            all_valid = all(reward_calc.validity_reward(r) > 0.5 for r in best_reactant_list)
            if all_valid:
                validity_count += 1

            any_in_stock = any(stock.is_buyable(r) for r in best_reactant_list)
            if any_in_stock:
                stock_count += 1

        # Handle edge case: entire batch produced no valid predictions
        if valid_samples == 0:
            if step % 10 == 0:
                print(f"Step {step:5d} | No valid predictions in batch, skipping update.")
            continue

        # c. Backward + update
        optimizer.zero_grad()
        avg_loss = total_loss / len(targets)

        # Guard against NaN loss
        if torch.isnan(avg_loss):
            print(f"Step {step:5d} | WARNING: NaN loss detected, skipping update.")
            continue

        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.get_model().parameters(), max_norm=1.0)
        optimizer.step()

        # d. Update baseline
        mean_reward = sum(rewards_list) / len(rewards_list)
        baseline = args.baseline_decay * baseline + (1 - args.baseline_decay) * mean_reward

        # Compute rates
        validity_rate = validity_count / len(targets)
        stock_rate = stock_count / len(targets)

        step_time = time.time() - step_start

        # e. Logging (every 10 steps)
        if step % 10 == 0:
            print(
                f"Step {step:5d} | "
                f"reward={mean_reward:.4f} | "
                f"validity={validity_rate:.2%} | "
                f"stock={stock_rate:.2%} | "
                f"loss={avg_loss.item():.4f} | "
                f"baseline={baseline:.4f} | "
                f"time={step_time:.1f}s"
            )

        # f. Checkpointing
        if args.checkpoint_every > 0 and step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(policy, optimizer, step, mean_reward, best_reward, args.checkpoint_dir)

        if mean_reward > best_reward:
            best_reward = mean_reward
            save_checkpoint(policy, optimizer, step, mean_reward, best_reward, args.checkpoint_dir)
            print(f"  New best reward: {best_reward:.4f}")

    # ---- 3. Final checkpoint ----
    print("=" * 60)
    print("Training complete.")
    final_step = args.num_steps - 1 if args.num_steps > 0 else 0
    save_checkpoint(policy, optimizer, final_step, best_reward, best_reward, args.checkpoint_dir)
    print(f"Final best reward: {best_reward:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    train(args)
