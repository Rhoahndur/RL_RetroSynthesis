"""Unit tests for StockList (data.stock.loader)."""

import gzip
import os
import tempfile

from data.stock.loader import StockList


def test_load_populates_set(stock_list):
    assert len(stock_list) > 100


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


def test_load_smi_gz():
    """Loading from a .smi.gz file works correctly."""
    smiles = ["CCO", "CO", "c1ccccc1", "CC(=O)O"]
    with tempfile.NamedTemporaryFile(suffix=".smi.gz", delete=False) as tmp:
        tmp_path = tmp.name
        with gzip.open(tmp_path, "wt") as gz:
            for smi in smiles:
                gz.write(smi + "\n")
    try:
        sl = StockList()
        sl.load_smi_gz(tmp_path)
        assert len(sl) == len(smiles)
        assert sl.is_buyable("CCO") is True
        assert sl.is_buyable("c1ccccc1") is True
        assert sl.is_buyable("INVALID") is False
    finally:
        os.unlink(tmp_path)


def test_load_auto_detects_smi_gz():
    """The load() method auto-detects .smi.gz format by extension."""
    smiles = ["CCO", "CO", "N"]
    with tempfile.NamedTemporaryFile(suffix=".smi.gz", delete=False) as tmp:
        tmp_path = tmp.name
        with gzip.open(tmp_path, "wt") as gz:
            for smi in smiles:
                gz.write(smi + "\n")
    try:
        sl = StockList()
        result = sl.load(tmp_path)
        assert result is sl
        assert len(sl) == len(smiles)
        assert sl.is_buyable("CCO") is True
    finally:
        os.unlink(tmp_path)


def test_load_smi_gz_filters_invalid():
    """Invalid SMILES in .smi.gz are silently skipped."""
    lines = ["CCO", "NOT_VALID_SMILES", "CO", ""]
    with tempfile.NamedTemporaryFile(suffix=".smi.gz", delete=False) as tmp:
        tmp_path = tmp.name
        with gzip.open(tmp_path, "wt") as gz:
            for line in lines:
                gz.write(line + "\n")
    try:
        sl = StockList()
        sl.load_smi_gz(tmp_path)
        # Only CCO and CO should survive
        assert len(sl) == 2
    finally:
        os.unlink(tmp_path)
