"""
Tests for the Insightly profiler module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.profiler import profile_dataframe, profile_column


class TestProfileDataframe:

    @pytest.fixture
    def mixed_df(self):
        return pd.DataFrame({
            "numeric": [1.0, 2.0, 3.0, 4.0, 100.0],
            "categorical": ["a", "b", "a", "b", "c"],
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "with_null": [1.0, None, 3.0, None, 5.0],
        })

    def test_overall_metrics(self, mixed_df):
        profile = profile_dataframe(mixed_df)
        overall = profile["overall"]
        assert overall["rows"] == 5
        assert overall["columns"] == 4
        assert overall["missing_cells"] == 2
        assert overall["total_cells"] == 20

    def test_column_profiles_present(self, mixed_df):
        profile = profile_dataframe(mixed_df)
        assert len(profile["columns"]) == 4
        assert "numeric" in profile["columns"]
        assert "categorical" in profile["columns"]

    def test_empty_df(self):
        df = pd.DataFrame()
        profile = profile_dataframe(df)
        assert profile["overall"] == {}
        assert profile["columns"] == {}

    def test_numeric_stats(self, mixed_df):
        col = profile_dataframe(mixed_df)["columns"]["numeric"]
        assert col["mean"] == 22.0
        assert col["min"] == 1.0
        assert col["max"] == 100.0
        assert col["outliers_iqr"] > 0  # 100 is an outlier

    def test_categorical_stats(self, mixed_df):
        col = profile_dataframe(mixed_df)["columns"]["categorical"]
        assert col["unique"] == 3
        assert col["top_value"] == "a" or col["top_value"] == "b"
        assert "entropy" in col


class TestProfileColumn:

    def test_numeric(self):
        # 20 unique values so it's not classified as Categorical (Numeric)
        s = pd.Series(list(range(1, 21)), dtype=float, name="test")
        info = profile_column(s)
        assert info["inferred_type"] == "Numeric"
        assert info["mean"] == 10.5
        assert info["nulls"] == 0

    def test_categorical(self):
        s = pd.Series(["a", "b", "a", "c", "a"], name="test")
        info = profile_column(s)
        assert "Categorical" in info["inferred_type"]
        assert info["top_value"] == "a"
        assert info["empty_strings"] == 0

    def test_with_nulls(self):
        s = pd.Series([1.0, None, 3.0, None, None], name="test")
        info = profile_column(s)
        assert info["nulls"] == 3
        assert info["null_pct"] == 60.0
        assert info["count"] == 2

    def test_empty_series(self):
        s = pd.Series([], dtype=object, name="test")
        info = profile_column(s)
        assert info["count"] == 0
        assert info["nulls"] == 0
