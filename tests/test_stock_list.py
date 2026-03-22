"""Unit tests for StockList (data.stock.loader)."""

from data.stock.loader import StockList


def test_load_populates_set(stock_list):
    assert len(stock_list) > 200


def test_load_returns_self():
    sl = StockList()
    result = sl.load()
    assert result is sl


def test_ethanol_is_buyable(stock_list):
    assert stock_list.is_buyable("CCO") is True


def test_methanol_is_buyable(stock_list):
    assert stock_list.is_buyable("CO") is True


def test_ibuprofen_not_buyable(stock_list):
    assert stock_list.is_buyable("CC(C)Cc1ccc(cc1)C(C)C(=O)O") is False


def test_invalid_smiles_returns_false(stock_list):
    assert stock_list.is_buyable("not_a_smiles") is False


def test_empty_string_returns_false(stock_list):
    assert stock_list.is_buyable("") is False


def test_canonical_matching(stock_list):
    assert stock_list.is_buyable("C(O)") is True
    assert stock_list.is_buyable("CO") is True


def test_contains_operator(stock_list):
    assert "CCO" in stock_list


def test_canonicalize_valid():
    assert StockList.canonicalize("c1ccccc1") == "c1ccccc1"


def test_canonicalize_invalid():
    assert StockList.canonicalize("xyz") is None


def test_known_precursors_present(stock_list):
    assert stock_list.is_buyable("OC(=O)c1ccccc1O") is True
    assert stock_list.is_buyable("Nc1ccc(O)cc1") is True
    assert stock_list.is_buyable("CC(=O)OC(C)=O") is True
