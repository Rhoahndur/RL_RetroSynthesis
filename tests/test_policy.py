"""Unit tests for models.policy — lightweight checks that avoid model downloads."""

import pytest

from models.policy import RetroPolicy, DEFAULT_MODEL_NAME


# ---------------------------------------------------------------------------
# Fast tests (no model download)
# ---------------------------------------------------------------------------


def test_detect_device_returns_string():
    device = RetroPolicy.detect_device()
    assert isinstance(device, str)
    assert device in ("cuda", "mps", "cpu")


def test_detect_device_is_cpu_or_mps_on_test():
    """On a typical test machine (no NVIDIA GPU) this should not crash."""
    device = RetroPolicy.detect_device()
    assert device in ("cpu", "mps")


def test_default_model_name():
    assert DEFAULT_MODEL_NAME == "sagawa/ReactionT5v2-retrosynthesis"


def test_format_input():
    """_format_input is a pure string helper — test without loading a model."""
    # Call the unbound method directly via the class
    result = RetroPolicy._format_input(None, "CC(=O)O")
    assert result == "REACTANT:CC(=O)O"


# ---------------------------------------------------------------------------
# Slow tests (require model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_policy_loads_model():
    policy = RetroPolicy(device="cpu")
    assert hasattr(policy, "model")
    assert hasattr(policy, "tokenizer")
    assert policy.device == "cpu"


@pytest.mark.slow
def test_policy_predict_returns_list():
    policy = RetroPolicy(device="cpu")
    results = policy.predict("CC(=O)O", num_candidates=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
