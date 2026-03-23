"""Unit tests for RewardCalculator."""

# SMILES constants
ETHANOL = "CCO"
ACETIC_ACID = "CC(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALICYLIC_ACID = "OC(=O)c1ccccc1O"
ACETIC_ANHYDRIDE = "CC(=O)OC(C)=O"


# ── validity_reward ──────────────────────────────────────────────────────────


def test_validity_valid_smiles(reward_calc):
    assert reward_calc.validity_reward(ETHANOL) == 1.0


def test_validity_invalid_smiles(reward_calc):
    assert reward_calc.validity_reward("not_valid") == 0.0


def test_validity_empty_string(reward_calc):
    assert reward_calc.validity_reward("") == 0.0


def test_validity_none_input(reward_calc):
    assert reward_calc.validity_reward(None) == 0.0


# ── plausibility_reward ──────────────────────────────────────────────────────


def test_plausibility_valid(reward_calc):
    assert reward_calc.plausibility_reward(ETHANOL) == 1.0


def test_plausibility_invalid(reward_calc):
    assert reward_calc.plausibility_reward("xyz") == 0.0


# ── compute_sascore ──────────────────────────────────────────────────────────


def test_sascore_ethanol(reward_calc):
    score = reward_calc.compute_sascore(ETHANOL)
    assert isinstance(score, float)
    assert 1.0 <= score <= 10.0


def test_sascore_ibuprofen_higher(reward_calc):
    ethanol_sa = reward_calc.compute_sascore(ETHANOL)
    ibuprofen_sa = reward_calc.compute_sascore(IBUPROFEN)
    assert ibuprofen_sa > ethanol_sa


def test_sascore_invalid(reward_calc):
    assert reward_calc.compute_sascore("invalid") is None


def test_sascore_none(reward_calc):
    assert reward_calc.compute_sascore(None) is None


# ── sascore_reward ───────────────────────────────────────────────────────────


def test_sascore_reward_simpler_reactants(reward_calc):
    reward = reward_calc.sascore_reward(IBUPROFEN, [ETHANOL, ACETIC_ACID])
    assert reward > 0.5


def test_sascore_reward_empty_reactants(reward_calc):
    assert reward_calc.sascore_reward(IBUPROFEN, []) == 0.0


def test_sascore_reward_invalid_reactant(reward_calc):
    assert reward_calc.sascore_reward(IBUPROFEN, ["invalid"]) == 0.0


# ── stock_reward ─────────────────────────────────────────────────────────────


def test_stock_reward_buyable(reward_calc, stock_list):
    assert reward_calc.stock_reward(ETHANOL, stock_list) == 1.0


def test_stock_reward_not_buyable(reward_calc, stock_list):
    # Ibuprofen is not exactly buyable but may get partial credit
    # via fingerprint similarity to buyable substructures
    assert reward_calc.stock_reward(IBUPROFEN, stock_list) < 1.0


# ── atom_conservation_reward ─────────────────────────────────────────────────


def test_atom_conservation_perfect(reward_calc):
    reward = reward_calc.atom_conservation_reward(ASPIRIN, [SALICYLIC_ACID, ACETIC_ANHYDRIDE])
    assert reward == 1.0


def test_atom_conservation_missing_atoms(reward_calc):
    reward = reward_calc.atom_conservation_reward(ASPIRIN, [ETHANOL])
    assert reward < 1.0


def test_atom_conservation_invalid(reward_calc):
    assert reward_calc.atom_conservation_reward("invalid", [ETHANOL]) == 0.0


# ── combined_reward ──────────────────────────────────────────────────────────


def test_combined_reward_in_range(reward_calc, stock_list):
    reward = reward_calc.combined_reward(ASPIRIN, [ETHANOL, ACETIC_ACID], stock_list)
    assert 0.0 <= reward <= 1.0


def test_combined_reward_all_valid_buyable(reward_calc, stock_list):
    reward = reward_calc.combined_reward(ASPIRIN, [SALICYLIC_ACID, ACETIC_ANHYDRIDE], stock_list)
    assert reward > 0.0


def test_combined_reward_empty_input(reward_calc, stock_list):
    assert reward_calc.combined_reward("", [ETHANOL], stock_list) == 0.0


def test_combined_reward_custom_weights(reward_calc, stock_list):
    default_reward = reward_calc.combined_reward(
        ASPIRIN, [SALICYLIC_ACID, ACETIC_ANHYDRIDE], stock_list
    )
    custom_weights = {"validity": 0.0, "plausibility": 0.0, "sascore": 0.0, "stock": 1.0}
    custom_reward = reward_calc.combined_reward(
        ASPIRIN, [SALICYLIC_ACID, ACETIC_ANHYDRIDE], stock_list, weights=custom_weights
    )
    assert custom_reward != default_reward
