"""Tests for scripts/eval_topk.py — top-K exact match evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_topk import canonicalize_reaction, compute_sascore_bucket, evaluate

# ── canonicalize_reaction ───────────────────────────────────────────────


def test_canonicalize_reaction_basic():
    result = canonicalize_reaction("CCO.CC(=O)O")
    assert isinstance(result, frozenset)
    assert len(result) == 2


def test_canonicalize_reaction_order_independent():
    a = canonicalize_reaction("CCO.CC(=O)O")
    b = canonicalize_reaction("CC(=O)O.CCO")
    assert a == b


def test_canonicalize_reaction_invalid_fragment():
    result = canonicalize_reaction("CCO.invalid_xyz.CC(=O)O")
    assert len(result) == 2  # invalid fragment skipped


def test_canonicalize_reaction_empty():
    assert canonicalize_reaction("") == frozenset()


def test_canonicalize_reaction_none():
    assert canonicalize_reaction(None) == frozenset()


# ── compute_sascore_bucket ──────────────────────────────────────────────


def test_compute_sascore_bucket_easy():
    assert compute_sascore_bucket("C") == "easy"


def test_compute_sascore_bucket_medium_or_hard():
    bucket = compute_sascore_bucket("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    assert bucket in ("medium", "hard")


def test_compute_sascore_bucket_invalid():
    assert compute_sascore_bucket("not_a_smiles") == "unknown"


# ── evaluate ────────────────────────────────────────────────────────────


def test_evaluate_with_mock_policy_match(mock_policy):
    """MockPolicy returns correct aspirin reactants, so top-1 should match."""
    examples = [
        {
            "product": "CC(=O)Oc1ccccc1C(=O)O",
            "ground_truth": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
        }
    ]
    results = evaluate(examples, mock_policy, top_k=10)
    assert results["overall"]["top_1"] > 0


def test_evaluate_no_match(mock_policy):
    """MockPolicy returns default (ethanol + acetic acid) which won't match."""
    examples = [
        {
            "product": "c1ccccc1",  # benzene — MockPolicy gives default response
            "ground_truth": "C1=CC=CC=C1.O",  # made-up ground truth
        }
    ]
    results = evaluate(examples, mock_policy, top_k=10)
    assert results["overall"]["top_1"] == 0.0


def test_evaluate_result_structure(mock_policy):
    examples = [
        {
            "product": "CC(=O)Oc1ccccc1C(=O)O",
            "ground_truth": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
        }
    ]
    results = evaluate(examples, mock_policy, top_k=10)
    assert "total_examples" in results
    assert "overall" in results
    assert "by_bucket" in results
    assert "top_1" in results["overall"]


# ── by_reaction_type ────────────────────────────────────────────────


def test_evaluate_by_reaction_type(mock_policy):
    """When examples have reaction_type, results include per-type breakdown."""
    examples = [
        {
            "product": "CC(=O)Oc1ccccc1C(=O)O",
            "ground_truth": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
            "reaction_type": "Acylation",
        },
        {
            "product": "CC(=O)Nc1ccc(O)cc1",
            "ground_truth": "Nc1ccc(O)cc1.CC(=O)OC(C)=O",
            "reaction_type": "Acylation",
        },
    ]
    results = evaluate(examples, mock_policy, top_k=10, by_reaction_type=True)
    assert "by_reaction_type" in results
    assert "Acylation" in results["by_reaction_type"]
    assert results["by_reaction_type"]["Acylation"]["count"] == 2


def test_evaluate_no_reaction_type_column(mock_policy):
    """Without reaction_type in examples, no by_reaction_type key in results."""
    examples = [
        {
            "product": "CC(=O)Oc1ccccc1C(=O)O",
            "ground_truth": "OC(=O)c1ccccc1O.CC(=O)OC(C)=O",
        },
    ]
    results = evaluate(examples, mock_policy, top_k=10, by_reaction_type=True)
    assert "by_reaction_type" not in results
