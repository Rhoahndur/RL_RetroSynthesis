"""Tests for scripts/inference.py helper functions."""

import base64
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stub out 'transformers' before importing scripts.inference so that
# models.policy can be imported without the HuggingFace library installed.
# ---------------------------------------------------------------------------
if "transformers" not in sys.modules:
    _mock = types.ModuleType("transformers")
    _mock.AutoModelForSeq2SeqLM = type(
        "AutoModelForSeq2SeqLM", (),
        {"from_pretrained": classmethod(lambda cls, *a, **kw: None)},
    )
    _mock.AutoTokenizer = type(
        "AutoTokenizer", (),
        {"from_pretrained": classmethod(lambda cls, *a, **kw: None)},
    )
    sys.modules["transformers"] = _mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inference import mol_to_base64_image, flatten_route_molecules, run_inference


# ---------------------------------------------------------------------------
# mol_to_base64_image
# ---------------------------------------------------------------------------

def test_mol_to_base64_image_valid():
    result = mol_to_base64_image("CCO")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_mol_to_base64_image_is_base64():
    result = mol_to_base64_image("CCO")
    assert result is not None
    # Should decode without error and produce non-empty bytes (PNG data)
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_mol_to_base64_image_invalid():
    result = mol_to_base64_image("invalid")
    assert result is None


def test_mol_to_base64_image_empty():
    # RDKit treats "" as a valid (empty) molecule, so the function renders a
    # blank image and returns a base64 string rather than None.
    result = mol_to_base64_image("")
    assert isinstance(result, str)
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


# ---------------------------------------------------------------------------
# flatten_route_molecules
# ---------------------------------------------------------------------------

def test_flatten_route_molecules_simple():
    tree = {"smiles": "CCO", "children": []}
    result = flatten_route_molecules(tree)
    assert result == ["CCO"]


def test_flatten_route_molecules_nested():
    tree = {
        "smiles": "A",
        "children": [
            {
                "smiles": "B",
                "children": [
                    {"smiles": "C", "children": []},
                ],
            },
            {
                "smiles": "D",
                "children": [],
            },
        ],
    }
    result = flatten_route_molecules(tree)
    assert result == ["A", "B", "C", "D"]


def test_flatten_route_molecules_deduplication():
    tree = {
        "smiles": "A",
        "children": [
            {"smiles": "B", "children": []},
            {"smiles": "A", "children": []},  # duplicate of root
            {"smiles": "B", "children": []},  # duplicate of first child
        ],
    }
    result = flatten_route_molecules(tree)
    assert result == ["A", "B"]


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

def test_run_inference_invalid_smiles(mock_policy, reward_calc, stock_list):
    result = run_inference(
        "invalid",
        mock_policy,
        reward_calc,
        stock_list,
        max_simulations=5,
        time_budget=3,
    )
    assert "error" in result
    assert result["routes"] == []


def test_run_inference_valid_smiles(mock_policy, reward_calc, stock_list):
    result = run_inference(
        "CC(=O)Oc1ccccc1C(=O)O",
        mock_policy,
        reward_calc,
        stock_list,
        max_simulations=5,
        time_budget=3,
    )
    for key in ("target", "routes", "best_score", "stats", "molecules"):
        assert key in result, f"Missing key: {key}"
    assert result["target"] == "CC(=O)Oc1ccccc1C(=O)O"


def test_run_inference_stats_keys(mock_policy, reward_calc, stock_list):
    result = run_inference(
        "CC(=O)Oc1ccccc1C(=O)O",
        mock_policy,
        reward_calc,
        stock_list,
        max_simulations=5,
        time_budget=3,
    )
    stats = result["stats"]
    for key in ("simulations", "time_seconds", "routes_found"):
        assert key in stats, f"Missing stats key: {key}"
