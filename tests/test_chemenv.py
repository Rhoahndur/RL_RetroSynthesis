"""Unit tests for ChemEnv retrosynthesis environment."""

import pytest

from env.ChemEnv import ChemEnv


class TestChemEnv:
    @pytest.fixture(autouse=True)
    def _make_env(self, mock_policy, reward_calc, stock_list):
        """Create a ChemEnv instance available as self.env for every test."""
        self.env = ChemEnv(mock_policy, reward_calc, stock_list)

    # ------------------------------------------------------------------
    # reset() tests
    # ------------------------------------------------------------------

    def test_reset_returns_state(self):
        """reset() returns a dict with the expected keys."""
        state = self.env.reset("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(state, dict)
        for key in ("molecules", "depths", "in_stock", "route_tree", "done"):
            assert key in state, f"Missing state key: {key}"

    def test_reset_state_has_target(self):
        """state['molecules'] equals [target_smiles] after reset."""
        state = self.env.reset("CC(=O)Oc1ccccc1C(=O)O")
        assert state["molecules"] == ["CC(=O)Oc1ccccc1C(=O)O"]

    def test_reset_buyable_target_is_done(self):
        """Resetting with ethanol (buyable) sets done=True immediately."""
        state = self.env.reset("CCO")
        assert state["done"] is True

    # ------------------------------------------------------------------
    # step() tests
    # ------------------------------------------------------------------

    def test_step_returns_tuple(self):
        """step() returns a 4-tuple (state, reward, done, info)."""
        self.env.reset("CC(=O)Oc1ccccc1C(=O)O")
        result = self.env.step()
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_step_reward_is_float(self):
        """The reward from step() is a float."""
        self.env.reset("CC(=O)Oc1ccccc1C(=O)O")
        _state, reward, _done, _info = self.env.step()
        assert isinstance(reward, float)

    def test_step_done_when_already_done(self):
        """After done=True, another step() returns done=True with 0 reward."""
        state = self.env.reset("CCO")  # buyable -> immediately done
        assert state["done"] is True

        _state, reward, done, _info = self.env.step()
        assert done is True
        assert reward == 0.0

    # ------------------------------------------------------------------
    # get_route() tests
    # ------------------------------------------------------------------

    def test_get_route_returns_dict(self):
        """get_route() returns a dict with a 'smiles' key."""
        self.env.reset("CC(=O)Oc1ccccc1C(=O)O")
        route = self.env.get_route()
        assert isinstance(route, dict)
        assert "smiles" in route

    # ------------------------------------------------------------------
    # Episode termination test
    # ------------------------------------------------------------------

    def test_episode_terminates(self):
        """Running step() in a loop eventually sets done=True (max_depth)."""
        self.env.reset("CC(=O)Oc1ccccc1C(=O)O")
        done = False
        steps = 0
        max_steps = 50  # safety limit

        while not done and steps < max_steps:
            _state, _reward, done, _info = self.env.step()
            steps += 1

        assert done is True, "Episode did not terminate within max_steps"
