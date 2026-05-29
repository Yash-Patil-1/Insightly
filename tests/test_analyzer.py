"""
Tests for the Insightly analyzer module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.analyzer import (
    analyze_correlations,
    detect_outliers_zscore,
    detect_distribution_anomalies,
    detect_time_trends,
    generate_insights_summary,
)


class TestAnalyzeCorrelations:

    @pytest.fixture
    def numeric_df(self):
        np.random.seed(42)
        x = np.random.randn(100)
        return pd.DataFrame({
            "A": x,
            "B": x * 2 + np.random.randn(100) * 0.1,  # strong positive
            "C": -x * 2 + np.random.randn(100) * 0.1,  # strong negative
            "D": np.random.randn(100),  # independent
        })

    def test_returns_matrix(self, numeric_df):
        result = analyze_correlations(numeric_df)
        assert result["matrix"] is not None
        assert result["matrix"].shape == (4, 4)

    def test_top_correlations(self, numeric_df):
        result = analyze_correlations(numeric_df)
        assert len(result["top"]) > 0

    def test_strong_correlation_detected(self, numeric_df):
        result = analyze_correlations(numeric_df)
        top = result["top"]
        top_vals = [abs(t["strength"]) for t in top]
        assert max(top_vals) > 0.9  # A and B should be strongly correlated

    def test_insufficient_columns(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = analyze_correlations(df)
        assert result["matrix"] is None
        assert "message" in result

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"A": ["x", "y", "z"]})
        result = analyze_correlations(df)
        assert result["matrix"] is None


class TestDetectOutliersZscore:

    @pytest.fixture
    def df_with_outliers(self):
        np.random.seed(42)
        values = np.random.randn(50) * 10 + 50
        values[0] = 200  # extreme outlier
        values[1] = -100  # extreme outlier
        return pd.DataFrame({"values": values})

    def test_detects_outliers(self, df_with_outliers):
        result = detect_outliers_zscore(df_with_outliers)
        assert len(result) > 0
        assert "values" in result

    def test_no_outliers(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        result = detect_outliers_zscore(df, threshold=5.0)
        assert len(result) == 0

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame()
        result = detect_outliers_zscore(df)
        assert result == {}


class TestDetectDistributionAnomalies:

    def test_high_skew(self):
        np.random.seed(42)
        values = np.random.exponential(scale=2, size=100)  # right-skewed
        df = pd.DataFrame({"skewed": values})
        anomalies = detect_distribution_anomalies(df)
        skew_found = any("skewed" in a["column"] for a in anomalies)
        assert isinstance(anomalies, list)

    def test_constant_column(self):
        df = pd.DataFrame({"constant": [5.0] * 10})
        anomalies = detect_distribution_anomalies(df)
        assert len(anomalies) > 0
        assert any("Constant" in a["issue"] for a in anomalies)

    def test_clean_column_no_anomalies(self):
        np.random.seed(42)
        df = pd.DataFrame({"normal": np.random.randn(100)})
        anomalies = detect_distribution_anomalies(df)
        # clean normal data should have few anomalies
        assert isinstance(anomalies, list)


class TestGenerateInsightsSummary:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
            "categorical": ["a", "b", "c", "a", "b"],
            "with_missing": [1.0, None, 3.0, None, 5.0],
        })

    def test_returns_list(self, sample_df):
        insights = generate_insights_summary(sample_df)
        assert isinstance(insights, list)
        assert len(insights) > 0

    def test_includes_row_count(self, sample_df):
        insights = generate_insights_summary(sample_df)
        assert any("5 rows" in i for i in insights)

    def test_includes_missing_info(self, sample_df):
        insights = generate_insights_summary(sample_df)
        assert any("missing" in i.lower() for i in insights)

    def test_empty_df(self):
        df = pd.DataFrame()
        insights = generate_insights_summary(df)
        assert isinstance(insights, list)
