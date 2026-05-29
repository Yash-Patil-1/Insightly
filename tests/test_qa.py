"""
Tests for the Insightly QA engine.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.qa import (
    answer_query,
    get_suggestions,
    _match_pattern,
    _resolve_col,
    _handle_missing,
    _handle_missing_all,
    _handle_filter,
    _handle_filter_simple,
    _handle_between,
    _handle_show_rows,
    _handle_mean,
    _handle_sum,
    _handle_correlation_pair,
    _handle_correlation_all,
    _handle_outliers,
    _handle_outliers_all,
)


# ── Fixtures ──

@pytest.fixture
def finance_df():
    """Sample finance DataFrame for testing."""
    return pd.read_csv(Path(__file__).parent.parent / "sample_data" / "personal_finance.csv")


@pytest.fixture
def survey_df():
    """Sample survey DataFrame for testing."""
    return pd.read_json(Path(__file__).parent.parent / "sample_data" / "survey_results.json")


@pytest.fixture
def missing_df():
    """DataFrame with known missing values."""
    np.random.seed(42)
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, np.nan, 35, np.nan, 45],
        "score": [90.0, 85.0, np.nan, 95.0, 88.0],
        "city": ["NYC", "LA", np.nan, "Chicago", "NYC"],
    })


# ═══════════════════════════════════════════════════════════════
# BUG REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════

class TestBugFixes:
    """Regression tests for the bugs that were fixed."""

    def test_missing_values_query(self, finance_df):
        """'missing values' should not be caught by filter_simple pattern."""
        result = answer_query(finance_df, "missing values")
        text = result["text"]
        assert "Could not parse" not in text
        assert "Got an error" not in text
        assert "Missing Values" in text or "No missing values" in text

    def test_how_many_missing_values(self, finance_df):
        """'how many missing values' should work."""
        result = answer_query(finance_df, "how many missing values")
        text = result["text"]
        assert "Could not parse" not in text
        assert "missing" in text.lower()

    def test_missing_values_in_column(self, finance_df):
        """'missing values in Category' should work."""
        result = answer_query(finance_df, "missing values in Category")
        text = result["text"]
        assert "Could not parse" not in text
        assert "Got an error" not in text

    def test_amount_between(self, finance_df):
        """'Amount between 100 and 300' should work without 'is' prefix."""
        result = answer_query(finance_df, "Amount between 100 and 300")
        text = result["text"]
        assert "Could not parse" not in text
        assert "Found" in text or "No rows found" in text

    def test_amount_is_between(self, finance_df):
        """'Amount is between 100 and 300' should work with 'is' prefix."""
        result = answer_query(finance_df, "Amount is between 100 and 300")
        text = result["text"]
        assert "Could not parse" not in text

    def test_filter_where_amount(self, finance_df):
        """'filter where Amount > 500' should work (operator preserved)."""
        result = answer_query(finance_df, "filter where Amount > 500")
        text = result["text"]
        assert "Could not parse" not in text
        assert "Got an error" not in text

    def test_category_equals_food(self, finance_df):
        """'Category equals Food' should work."""
        result = answer_query(finance_df, "Category equals Food")
        text = result["text"]
        assert "Could not parse" not in text
        assert "Got an error" not in text

    def test_filter_simple_operator_preserved(self, finance_df):
        """'Amount > 200' should preserve the operator."""
        result = answer_query(finance_df, "filter Amount > 200")
        text = result["text"]
        assert "Could not parse" not in text
        assert "row(s)" in text or "Found" in text

    def test_show_rows_where_filter(self, finance_df):
        """'show rows where Amount > 200' should work."""
        result = answer_query(finance_df, "show rows where Amount > 200")
        text = result["text"]
        assert "Could not parse" not in text

    def test_basic_queries(self, finance_df):
        """Basic queries still work."""
        qs = [
            "show me top 5 rows",
            "how many rows",
            "average of Amount",
            "unique values in Category",
        ]
        for q in qs:
            result = answer_query(finance_df, q)
            text = result["text"]
            assert "Could not parse" not in text, f"Failed on: {q}"
            assert "Got an error" not in text, f"Failed on: {q}"


# ═══════════════════════════════════════════════════════════════
# HANDLER UNIT TESTS
# ═══════════════════════════════════════════════════════════════

class TestHandlers:

    def test_handle_show_rows(self, finance_df):
        result = _handle_show_rows(finance_df, "5")
        assert result["text"] is not None
        assert "Showing top 5" in result["text"]
        assert "48" in result["text"]  # total rows

    def test_handle_missing_all(self, missing_df):
        result = _handle_missing_all(missing_df)
        text = result["text"]
        assert "Missing Values" in text
        assert "3" in text or "2" in text  # missing across columns

    def test_handle_missing_column(self, missing_df):
        result = _handle_missing(missing_df, "age")
        text = result["text"]
        assert "age" in text
        assert "2" in text  # 2 missing in age

    def test_handle_missing_no_col(self, missing_df):
        """With no col name, should fall back to all."""
        result = _handle_missing(missing_df)
        text = result["text"]
        assert "Missing Values" in text or "No missing" in text

    def test_handle_between(self, finance_df):
        result = _handle_between(finance_df, "Amount", "100", "300")
        text = result["text"]
        assert "row(s)" in text or "No rows" in text

    def test_handle_mean(self, finance_df):
        result = _handle_mean(finance_df, "Amount")
        text = result["text"]
        assert "Average" in text
        assert "739" in text

    def test_handle_sum(self, finance_df):
        result = _handle_sum(finance_df, "Amount")
        text = result["text"]
        assert "Total" in text

    def test_handle_correlation_pair(self, finance_df):
        result = _handle_correlation_pair(finance_df, "Amount", "Balance")
        text = result["text"]
        assert "Correlation" in text
        assert "r" in text or "r =" in text

    def test_handle_correlation_all(self, finance_df):
        result = _handle_correlation_all(finance_df)
        text = result["text"]
        assert "Correlation" in text or "Top" in text

    def test_handle_outliers_column(self, finance_df):
        result = _handle_outliers(finance_df, "Amount")
        text = result["text"]
        assert "Outliers" in text or "outlier" in text.lower()

    def test_handle_outliers_all(self, finance_df):
        result = _handle_outliers_all(finance_df)
        text = result["text"]
        assert "Outlier" in text or "outlier" in text.lower()


class TestColumnResolution:

    def test_exact_match(self, finance_df):
        col = _resolve_col(finance_df, "Amount")
        assert col == "Amount"

    def test_case_insensitive(self, finance_df):
        col = _resolve_col(finance_df, "amount")
        assert col == "Amount"

    def test_partial_match(self, finance_df):
        col = _resolve_col(finance_df, "Categ")
        assert col is not None

    def test_no_match(self, finance_df):
        col = _resolve_col(finance_df, "zzzznotacolumn")
        assert col is None


class TestSuggestions:

    def test_get_suggestions_finance(self, finance_df):
        suggestions = get_suggestions(finance_df)
        assert len(suggestions) > 0
        assert len(suggestions) <= 10

    def test_get_suggestions_survey(self, survey_df):
        suggestions = get_suggestions(survey_df)
        assert len(suggestions) > 0
        assert len(suggestions) <= 10

    def test_get_suggestions_includes_missing(self, finance_df):
        suggestions = get_suggestions(finance_df)
        assert any("missing" in s.lower() for s in suggestions)
