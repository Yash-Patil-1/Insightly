"""
Insightly — Auto-Profiler
Generates per-column statistics, missing value analysis, and data type summaries.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profile a full DataFrame, returning column-level stats and overall metrics.
    """
    if df is None or df.empty:
        return {"overall": {}, "columns": {}}

    overall = {
        "rows": len(df),
        "columns": len(df.columns),
        "total_cells": len(df) * len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
        if len(df) * len(df.columns) > 0
        else 0,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round(df.duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
        "memory_used": _format_bytes(df.memory_usage(deep=True).sum()),
    }

    columns = {}
    for col in df.columns:
        columns[col] = profile_column(df[col])

    return {"overall": overall, "columns": columns}


def profile_column(series: pd.Series) -> Dict[str, Any]:
    """Profile a single column."""
    info = {
        "name": series.name,
        "dtype": str(series.dtype),
        "inferred_type": _infer_type(series),
        "count": int(series.count()),
        "nulls": int(series.isnull().sum()),
        "null_pct": round(series.isnull().sum() / max(len(series), 1) * 100, 2),
        "unique": int(series.nunique()),
        "unique_pct": round(series.nunique() / max(series.count(), 1) * 100, 2),
    }

    # Numeric stats
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            info.update({
                "min": _safe_val(non_null.min()),
                "max": _safe_val(non_null.max()),
                "mean": _safe_val(non_null.mean()),
                "median": _safe_val(non_null.median()),
                "std": _safe_val(non_null.std()),
                "var": _safe_val(non_null.var()),
                "skew": _safe_val(non_null.skew()),
                "kurtosis": _safe_val(non_null.kurtosis()),
                "q25": _safe_val(non_null.quantile(0.25)),
                "q75": _safe_val(non_null.quantile(0.75)),
                "iqr": _safe_val(non_null.quantile(0.75) - non_null.quantile(0.25)),
                "zeros": int((non_null == 0).sum()),
                "negatives": int((non_null < 0).sum()),
                "outliers_iqr": _count_outliers_iqr(non_null),
            })

    # Categorical stats
    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        non_null = series.dropna()
        if len(non_null) > 0:
            value_counts = non_null.value_counts()
            top_n = value_counts.head(5).to_dict()
            info.update({
                "top_values": {str(k): int(v) for k, v in top_n.items()},
                "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "top_freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_pct": round(value_counts.iloc[0] / len(non_null) * 100, 2)
                if len(non_null) > 0 and len(value_counts) > 0
                else 0,
                "entropy": _entropy(non_null),
                "empty_strings": int((non_null.astype(str).str.strip() == "").sum()),
            })

    # Datetime stats
    if pd.api.types.is_datetime64_any_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            info.update({
                "min_date": str(non_null.min()),
                "max_date": str(non_null.max()),
                "range_days": (non_null.max() - non_null.min()).days,
                "most_frequent_date": str(non_null.mode().iloc[0]) if len(non_null.mode()) > 0 else None,
            })

    return info


def _infer_type(series: pd.Series) -> str:
    """Infer the semantic type of a column."""
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Datetime"
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10:
            return "Categorical (Numeric)"
        return "Numeric"
    if pd.api.types.is_object_dtype(series):
        nunique = series.nunique()
        total = series.count()
        if total > 0 and nunique / total < 0.05:
            return "Categorical (Low Cardinality)"
        if nunique > 100:
            return "Text / High Cardinality"
        return "Categorical"
    return str(series.dtype)


def _count_outliers_iqr(series: pd.Series) -> int:
    """Count outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def _entropy(series: pd.Series) -> float:
    """Compute Shannon entropy of a categorical series."""
    probs = series.value_counts(normalize=True)
    return round(-(probs * np.log2(probs + 1e-10)).sum(), 4)


def _safe_val(val: Any) -> Any:
    """Safely convert a numpy value to native Python type."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if np.isnan(val) or np.isinf(val):
            return None
        return round(float(val), 6)
    if isinstance(val, np.bool_):
        return bool(val)
    return val


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
