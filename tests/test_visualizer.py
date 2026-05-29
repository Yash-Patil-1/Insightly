"""
Tests for the Insightly visualizer module.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.visualizer import (
    suggest_chart_type,
    create_chart,
    create_correlation_heatmap,
    create_missing_heatmap,
    auto_gallery,
)


class TestSuggestChartType:

    def test_numeric_single(self):
        s = pd.Series([1, 2, 3, 4, 100])
        chart = suggest_chart_type(s)
        assert chart in ("histogram", "bar")

    def test_categorical_single(self):
        s = pd.Series(["a", "b", "c", "a", "b"])
        chart = suggest_chart_type(s)
        assert chart in ("bar", "pie", "histogram")

    def test_two_numeric(self):
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([4, 5, 6])
        chart = suggest_chart_type(s1, secondary=s2)
        assert chart == "scatter"

    def test_numeric_and_categorical(self):
        s1 = pd.Series([1, 2, 3, 4])
        s2 = pd.Series(["a", "b", "a", "b"])
        chart = suggest_chart_type(s1, secondary=s2)
        assert chart in ("bar", "box", "violin")


class TestCreateChart:

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["a", "b", "a", "b", "c"],
            "num2": [5.0, 4.0, 3.0, 2.0, 1.0],
        })

    def test_single_chart(self, df):
        fig = create_chart(df, "num")
        assert fig is not None

    def test_two_column_chart(self, df):
        fig = create_chart(df, "num", secondary="num2", chart_type="scatter")
        assert fig is not None

    def test_specific_chart_type(self, df):
        fig = create_chart(df, "cat", chart_type="bar")
        assert fig is not None

    def test_with_color(self, df):
        fig = create_chart(df, "num", secondary="num2", chart_type="scatter", color_col="cat")
        assert fig is not None


class TestCreateCorrelationHeatmap:

    def test_with_matrix(self):
        corr = pd.DataFrame({
            "A": [1.0, 0.5, -0.3],
            "B": [0.5, 1.0, 0.1],
            "C": [-0.3, 0.1, 1.0],
        }, index=["A", "B", "C"])
        fig = create_correlation_heatmap(corr)
        assert fig is not None

    def test_empty_matrix(self):
        fig = create_correlation_heatmap(pd.DataFrame())
        assert fig is not None


class TestCreateMissingHeatmap:

    def test_with_data(self):
        df = pd.DataFrame({"A": [1, None, 3], "B": [None, 2, 3]})
        fig = create_missing_heatmap(df)
        assert fig is not None


class TestAutoGallery:

    @pytest.fixture
    def df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "num1": np.random.randn(50),
            "num2": np.random.randn(50),
            "cat": np.random.choice(["a", "b", "c"], 50),
        })

    def test_returns_charts(self, df):
        gallery = auto_gallery(df, max_charts=4)
        assert len(gallery) > 0
        for name, chart_type, fig in gallery:
            assert name is not None
            assert chart_type is not None
            assert fig is not None

    def test_max_charts_respected(self, df):
        gallery = auto_gallery(df, max_charts=2)
        assert len(gallery) <= 2
