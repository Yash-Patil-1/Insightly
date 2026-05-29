"""
Tests for the Insightly reporter module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.reporter import generate_report
from src.profiler import profile_dataframe


class TestGenerateReport:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "Chicago"],
            "score": [90.0, 85.0, 95.0],
        })

    def test_returns_string(self, sample_df):
        profile = profile_dataframe(sample_df)
        report = generate_report(sample_df, profile)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_includes_dataset_overview(self, sample_df):
        profile = profile_dataframe(sample_df)
        report = generate_report(sample_df, profile)
        assert "Dataset Overview" in report or "Overview" in report

    def test_includes_column_details(self, sample_df):
        profile = profile_dataframe(sample_df)
        report = generate_report(sample_df, profile)
        assert "name" in report
        assert "age" in report
        assert "city" in report
        assert "score" in report

    def test_includes_recommendations(self, sample_df):
        profile = profile_dataframe(sample_df)
        report = generate_report(sample_df, profile)
        assert "Recommendation" in report or "recommendation" in report

    def test_empty_df(self):
        df = pd.DataFrame()
        profile = profile_dataframe(df)
        report = generate_report(df, profile)
        assert "No data loaded" in report

    def test_with_missing_data(self):
        df = pd.DataFrame({
            "A": [1.0, None, 3.0],
            "B": ["x", "y", None],
        })
        profile = profile_dataframe(df)
        report = generate_report(df, profile)
        assert "missing" in report.lower() or "Missing" in report
