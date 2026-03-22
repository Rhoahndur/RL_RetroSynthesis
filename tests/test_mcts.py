"""Unit tests for MCTS search engine (MCTSNode, MCTSResult, MCTS)."""

import math

import pytest

from env.MCTS import MCTS, MCTSNode, MCTSResult


# ---------------------------------------------------------------------------
# MCTSNode tests
# ---------------------------------------------------------------------------

class TestMCTSNode:

    def test_node_value_no_visits(self):
        """value property returns 0.0 when visit_count is 0."""
        node = MCTSNode(smiles="C")
        assert node.visit_count == 0
        assert node.value == 0.0

    def test_node_value_with_visits(self):
        """Node with visit_count=4, total_value=2.0 has value 0.5."""
        node = MCTSNode(smiles="C", visit_count=4, total_value=2.0)
        assert node.value == pytest.approx(0.5)

    def test_uct_unvisited_is_infinity(self):
        """uct_score() returns inf for an unvisited node."""
        node = MCTSNode(smiles="C", visit_count=0)
        assert node.uct_score() == float("inf")

    def test_uct_root_returns_value(self):
        """Root node (no parent) returns self.value from uct_score()."""
        node = MCTSNode(smiles="C", visit_count=3, total_value=1.5, parent=None)
        assert node.uct_score() == pytest.approx(node.value)


# ---------------------------------------------------------------------------
# MCTSResult tests
# ---------------------------------------------------------------------------

class TestMCTSResult:

    def test_result_defaults(self):
        """MCTSResult() has None best_route, 0.0 score, and empty lists."""
        result = MCTSResult()
        assert result.best_route is None
        assert result.score == 0.0
        assert result.all_routes == []
        assert "simulations" in result.stats
        assert "time_seconds" in result.stats
        assert "routes_found" in result.stats


# ---------------------------------------------------------------------------
# MCTS search tests
# ---------------------------------------------------------------------------

class TestMCTS:

    def test_mcts_init(self, mock_policy, reward_calc, stock_list):
        """MCTS can be instantiated without errors."""
        mcts = MCTS(mock_policy, reward_calc, stock_list)
        assert mcts.policy is mock_policy
        assert mcts.stock_list is stock_list

    def test_search_returns_result(self, mock_policy, reward_calc, stock_list):
        """search() on aspirin returns an MCTSResult."""
        mcts = MCTS(mock_policy, reward_calc, stock_list,
                     max_simulations=10)
        result = mcts.search("CC(=O)Oc1ccccc1C(=O)O", time_budget=5)
        assert isinstance(result, MCTSResult)

    def test_search_respects_max_simulations(self, mock_policy, reward_calc, stock_list):
        """With max_simulations=5, stats['simulations'] does not exceed 5."""
        mcts = MCTS(mock_policy, reward_calc, stock_list,
                     max_simulations=5)
        result = mcts.search("CC(=O)Oc1ccccc1C(=O)O", time_budget=5)
        assert result.stats["simulations"] <= 5

    def test_search_buyable_target(self, mock_policy, reward_calc, stock_list):
        """Searching for ethanol (already buyable) gives score 1.0 and a route."""
        mcts = MCTS(mock_policy, reward_calc, stock_list,
                     max_simulations=10)
        result = mcts.search("CCO", time_budget=5)
        assert result.score == pytest.approx(1.0)
        assert result.best_route is not None

    def test_search_result_has_stats(self, mock_policy, reward_calc, stock_list):
        """result.stats contains required keys."""
        mcts = MCTS(mock_policy, reward_calc, stock_list,
                     max_simulations=10)
        result = mcts.search("CC(=O)Oc1ccccc1C(=O)O", time_budget=5)
        for key in ("simulations", "time_seconds", "routes_found"):
            assert key in result.stats, f"Missing stats key: {key}"

    def test_extract_routes_format(self, mock_policy, reward_calc, stock_list):
        """If routes are found, each has smiles, score, in_stock, children keys."""
        mcts = MCTS(mock_policy, reward_calc, stock_list,
                     max_simulations=10)
        result = mcts.search("CC(=O)Oc1ccccc1C(=O)O", time_budget=5)
        if result.all_routes:
            for route in result.all_routes:
                assert "smiles" in route
                assert "score" in route
                assert "in_stock" in route
                assert "children" in route
