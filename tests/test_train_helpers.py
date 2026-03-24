"""Tests for helper functions in scripts/train_rl.py."""

import sys
import types
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Stub out 'transformers' before importing scripts.train_rl so that
# models.policy can be imported without the HuggingFace library installed.
# ---------------------------------------------------------------------------
if "transformers" not in sys.modules:
    _mock = types.ModuleType("transformers")
    _mock.AutoModelForSeq2SeqLM = type(
        "AutoModelForSeq2SeqLM",
        (),
        {"from_pretrained": classmethod(lambda cls, *a, **kw: None)},
    )
    _mock.AutoTokenizer = type(
        "AutoTokenizer",
        (),
        {"from_pretrained": classmethod(lambda cls, *a, **kw: None)},
    )
    sys.modules["transformers"] = _mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_rl import FALLBACK_MOLECULES, load_training_data, sample_batch, save_checkpoint

# ---------------------------------------------------------------------------
# load_training_data
# ---------------------------------------------------------------------------


def test_load_training_data_missing_file():
    result = load_training_data("/nonexistent/path.csv")
    assert result == list(FALLBACK_MOLECULES)


def test_load_training_data_valid_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("smiles\nCCO\nCO\nc1ccccc1\n")
    result = load_training_data(str(csv_file))
    assert len(result) == 3
    assert "CCO" in result
    assert "CO" in result
    assert "c1ccccc1" in result


# ---------------------------------------------------------------------------
# sample_batch
# ---------------------------------------------------------------------------


def test_sample_batch_correct_size():
    data = ["A", "B", "C"]
    result = sample_batch(data, 5)
    assert len(result) == 5


def test_sample_batch_from_data():
    data = ["A", "B", "C"]
    result = sample_batch(data, 10)
    for item in result:
        assert item in data


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


def test_save_checkpoint_creates_file(tmp_path, mock_policy):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    save_checkpoint(
        policy=mock_policy,
        optimizer=optimizer,
        step=100,
        reward=0.5,
        best_reward=0.5,
        checkpoint_dir=str(tmp_path),
    )
    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1
    assert "checkpoint_step100" in files[0].name


def test_save_checkpoint_prunes_old(tmp_path, mock_policy):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)

    # Create 5 checkpoints; step 2 has the highest reward (0.9000)
    configs = [
        (1, 0.1),
        (2, 0.9),  # best reward — should survive pruning
        (3, 0.3),
        (4, 0.4),
        (5, 0.5),
    ]
    for step, reward in configs:
        save_checkpoint(
            policy=mock_policy,
            optimizer=optimizer,
            step=step,
            reward=reward,
            best_reward=max(r for _, r in configs[: configs.index((step, reward)) + 1]),
            checkpoint_dir=str(tmp_path),
        )

    remaining = list(tmp_path.glob("checkpoint_step*_reward*.pt"))
    remaining_names = sorted(f.name for f in remaining)

    # Should keep last 3 by step (steps 3, 4, 5) plus best reward (step 2)
    assert len(remaining) == 4, f"Expected 4 checkpoints, got {len(remaining)}: {remaining_names}"

    # The best-reward checkpoint (step 2) must survive
    assert any("step2" in f.name for f in remaining), (
        f"Best-reward checkpoint (step2) was pruned. Remaining: {remaining_names}"
    )

    # The last 3 by step (3, 4, 5) must survive
    for step_num in (3, 4, 5):
        assert any(f"step{step_num}" in f.name for f in remaining), (
            f"Checkpoint step{step_num} was pruned. Remaining: {remaining_names}"
        )

    # Step 1 (lowest step, not best reward) should be pruned
    assert not any("step1" in f.name for f in remaining), (
        f"Checkpoint step1 should have been pruned. Remaining: {remaining_names}"
    )
