"""Tests for scripts/eval_mcts.py — MCTS full-route success rate evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.MCTS import MCTS
from scripts.eval_mcts import DEMO_MOLECULES, compute_route_depth, evaluate_molecule

# ── DEMO_MOLECULES ──────────────────────────────────────────────────────


def test_demo_molecules_list():
    assert len(DEMO_MOLECULES) == 4
    for mol in DEMO_MOLECULES:
        assert "smiles" in mol
        assert "name" in mol


# ── compute_route_depth ─────────────────────────────────────────────────


def test_compute_route_depth_leaf():
    route = {"smiles": "C", "score": 1.0, "in_stock": True, "children": []}
    assert compute_route_depth(route) == 0


def test_compute_route_depth_one_level():
    route = {
        "smiles": "CC(=O)O",
        "score": 0.8,
        "in_stock": False,
        "children": [
            {"smiles": "CCO", "score": 1.0, "in_stock": True, "children": []},
            {"smiles": "C", "score": 1.0, "in_stock": True, "children": []},
        ],
    }
    assert compute_route_depth(route) == 1


def test_compute_route_depth_nested():
    route = {
        "smiles": "A",
        "score": 0.5,
        "in_stock": False,
        "children": [
            {
                "smiles": "B",
                "score": 0.7,
                "in_stock": False,
                "children": [
                    {"smiles": "C", "score": 1.0, "in_stock": True, "children": []},
                ],
            }
        ],
    }
    assert compute_route_depth(route) == 2


# ── evaluate_molecule ───────────────────────────────────────────────────


def test_evaluate_molecule_buyable_target(mock_policy, reward_calc, stock_list):
    """Ethanol is buyable — MCTS should succeed immediately."""
    mcts = MCTS(mock_policy, reward_calc, stock_list, max_simulations=5)
    result = evaluate_molecule(mcts, "CCO", "Ethanol")
    assert result["success"] is True
    assert result["best_score"] == 1.0


def test_evaluate_molecule_aspirin(mock_policy, reward_calc, stock_list):
    """Aspirin with MockPolicy should return a valid result dict."""
    mcts = MCTS(mock_policy, reward_calc, stock_list, max_simulations=10)
    result = evaluate_molecule(mcts, "CC(=O)Oc1ccccc1C(=O)O", "Aspirin")
    assert "success" in result
    assert "num_routes" in result
    assert "num_complete" in result
    assert "best_score" in result
    assert "route_depth" in result
    assert "simulations" in result
    assert "time_seconds" in result


def test_evaluate_molecule_result_types(mock_policy, reward_calc, stock_list):
    """Check that result values have correct types."""
    mcts = MCTS(mock_policy, reward_calc, stock_list, max_simulations=5)
    result = evaluate_molecule(mcts, "CCO", "Ethanol")
    assert isinstance(result["success"], bool)
    assert isinstance(result["num_routes"], int)
    assert isinstance(result["num_complete"], int)
    assert isinstance(result["best_score"], float)
    assert isinstance(result["route_depth"], int)
    assert isinstance(result["simulations"], int)
    assert isinstance(result["time_seconds"], float)
