"""
Tests for the Insightly cleaner module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.cleaner import (
    detect_issues,
    clean_dataframe,
    fill_missing_options,
)


class TestDetectIssues:

    @pytest.fixture
    def dirty_df(self):
        return pd.DataFrame({
            "A": [1.0, 2.0, None, 4.0, 100.0],
            "B": ["x", "y", "z", "x", "x"],
            "C": [10, 20, 30, 40, 50],
        })

    def test_detects_missing(self, dirty_df):
        issues = detect_issues(dirty_df)
        assert len(issues["missing"]) == 1  # column A has 1 null

    def test_detects_outliers(self, dirty_df):
        issues = detect_issues(dirty_df)
        assert len(issues["outliers"]) > 0  # 100 is outlier in A

    def test_clean_data_no_issues(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        issues = detect_issues(df)
        assert sum(len(v) for v in issues.values()) == 0

    def test_detects_duplicates(self):
        df = pd.DataFrame({"A": [1, 1, 2], "B": ["x", "x", "y"]})
        issues = detect_issues(df)
        assert len(issues["duplicates"]) == 1


class TestCleanDataframe:

    @pytest.fixture
    def dirty_df(self):
        return pd.DataFrame({
            "A": [1.0, 2.0, None, 4.0, 100.0],
            "B": ["x", "y", "z", "x", "x"],
            "C": [10, 20, 30, 40, 50],
        })

    def test_drop_missing(self, dirty_df):
        cleaned, log = clean_dataframe(dirty_df, fill_missing="drop")
        assert len(cleaned) == 4  # dropped 1 row with NaN
        assert any("Dropped" in entry for entry in log)

    def test_fill_mean(self, dirty_df):
        cleaned, log = clean_dataframe(dirty_df, fill_missing="mean")
        assert cleaned["A"].isnull().sum() == 0
        assert any("mean" in entry.lower() for entry in log)

    def test_drop_duplicates(self):
        df = pd.DataFrame({"A": [1, 1, 2], "B": ["x", "x", "y"]})
        cleaned, log = clean_dataframe(df, drop_duplicates=True)
        assert len(cleaned) == 2
        assert any("duplicate" in entry.lower() for entry in log)

    def test_remove_outliers(self, dirty_df):
        cleaned, log = clean_dataframe(dirty_df, remove_outliers=True)
        # 100 is an outlier in column A
        if len(cleaned) < len(dirty_df):
            assert any("outlier" in entry.lower() for entry in log)

    def test_no_changes(self, dirty_df):
        cleaned, log = clean_dataframe(dirty_df)
        assert len(cleaned) == len(dirty_df)
        assert log == []

    def test_fix_types(self):
        df = pd.DataFrame({
            "num_str": ["1", "2", "3"],
            "text": ["a", "b", "c"],
        })
        cleaned, log = clean_dataframe(df, fix_types=True)
        assert pd.api.types.is_numeric_dtype(cleaned["num_str"])

    def test_returns_copy(self, dirty_df):
        cleaned, _ = clean_dataframe(dirty_df)
        assert cleaned is not dirty_df


class TestFillMissingOptions:

    def test_returns_list(self):
        options = fill_missing_options()
        assert isinstance(options, list)
        assert "none" in options
        assert "drop" in options
        assert "mean" in options
        assert "median" in options
        assert "mode" in options
