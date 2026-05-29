"""
Tests for the Insightly loader module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.loader import (
    get_file_info,
    detect_format,
    infer_separator,
    SUPPORTED_FORMATS,
)


class TestGetFileInfo:

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
            "C": [1.0, 2.0, 3.0],
        })

    def test_basic_info(self, df):
        info = get_file_info(df)
        assert info["rows"] == 3
        assert info["columns"] == 3
        assert info["numeric_cols"] == 2
        assert info["categorical_cols"] == 1
        assert info["missing_cells"] == 0
        assert info["duplicate_rows"] == 0

    def test_with_missing(self):
        df = pd.DataFrame({"A": [1, None, 3], "B": ["x", None, "z"]})
        info = get_file_info(df)
        assert info["missing_cells"] == 2
        assert info["rows"] == 3

    def test_with_duplicates(self):
        df = pd.DataFrame({"A": [1, 1, 2], "B": ["x", "x", "y"]})
        info = get_file_info(df)
        assert info["duplicate_rows"] == 1

    def test_empty_df(self):
        info = get_file_info(None)
        assert info == {}

    def test_memory_format(self, df):
        info = get_file_info(df)
        assert "KB" in info["memory_human"] or "B" in info["memory_human"]


class TestDetectFormat:

    def test_csv(self):
        assert detect_format("data.csv") == "CSV (Comma-separated)"

    def test_excel(self):
        assert detect_format("data.xlsx") == "Excel (.xlsx)"
        assert detect_format("data.xls") == "Excel (.xls)"

    def test_json(self):
        assert detect_format("data.json") == "JSON"

    def test_unknown(self):
        assert detect_format("data.txt") == "Unknown"


class TestInferSeparator:

    def test_tsv(self):
        assert infer_separator("data.tsv") == "\t"

    def test_csv(self):
        assert infer_separator("data.csv") == ","

    def test_unknown(self):
        assert infer_separator("data.txt") == ","
