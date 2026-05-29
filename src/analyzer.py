"""
Insightly — Analysis Engine
Correlations, outlier detection, distribution analysis, trend detection, and key insights.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from scipy import stats as scipy_stats


def analyze_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute correlation matrix and highlight top correlations."""
    numeric_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric_df.shape[1] < 2:
        return {"matrix": None, "top": [], "message": "Need at least 2 numeric columns for correlation."}

    corr = numeric_df.corr(method="pearson")

    # Extract top positive and negative correlations
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                pairs.append({
                    "col1": corr.columns[i],
                    "col2": corr.columns[j],
                    "correlation": round(val, 4),
                    "strength": abs(val),
                })

    pairs.sort(key=lambda x: x["strength"], reverse=True)
    top = pairs[:10]

    # p-values for top correlations
    for pair in top:
        c1, c2 = pair["col1"], pair["col2"]
        clean = numeric_df[[c1, c2]].dropna()
        if len(clean) > 3:
            _, p_val = scipy_stats.pearsonr(clean[c1], clean[c2])
            pair["p_value"] = round(p_val, 6)
        else:
            pair["p_value"] = None

    return {
        "matrix": corr,
        "top": top,
        "message": None,
    }


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
    """Detect outliers using Z-score method."""
    numeric_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
    results = {}

    for col in numeric_df.columns:
        clean = numeric_df[col].dropna()
        if len(clean) < 4:
            continue
        z = np.abs(scipy_stats.zscore(clean, nan_policy="omit"))
        outlier_mask = z > threshold
        count = int(outlier_mask.sum())
        if count > 0:
            results[col] = {
                "count": count,
                "pct": round(count / len(clean) * 100, 2),
                "indices": clean.index[outlier_mask].tolist()[:20],  # limit to 20
                "threshold": threshold,
            }

    return results


def detect_distribution_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Flag columns with unusual distributions (high skew, bimodal, constant)."""
    anomalies = []

    for col in df.select_dtypes(include="number").columns:
        clean = df[col].dropna()
        if len(clean) < 5:
            continue

        # Constant column
        if clean.nunique() == 1:
            anomalies.append({
                "column": col,
                "issue": "Constant value",
                "detail": f"All {len(clean)} values are {clean.iloc[0]:.4f}",
            })
            continue

        # High skew
        skew = clean.skew()
        if abs(skew) > 2:
            anomalies.append({
                "column": col,
                "issue": "Highly skewed",
                "detail": f"Skewness = {skew:.3f} (|skew| > 2 indicates strong asymmetry)",
            })

        # High kurtosis
        kurt = clean.kurtosis()
        if kurt > 5:
            anomalies.append({
                "column": col,
                "issue": "Heavy tails (leptokurtic)",
                "detail": f"Kurtosis = {kurt:.3f} (kurtosis > 5 indicates extreme outliers)",
            })

        # High zero ratio
        zero_ratio = (clean == 0).sum() / len(clean)
        if zero_ratio > 0.5:
            anomalies.append({
                "column": col,
                "issue": "High zero-inflation",
                "detail": f"{zero_ratio:.0%} of values are zero",
            })

    # Categorical anomalies
    for col in df.select_dtypes(include="object").columns:
        clean = df[col].dropna()
        if len(clean) > 0:
            vc = clean.value_counts(normalize=True)
            if len(vc) > 0 and vc.iloc[0] > 0.95:
                anomalies.append({
                    "column": col,
                    "issue": "Dominant category",
                    "detail": f"'{vc.index[0]}' appears in {vc.iloc[0]:.0%} of rows",
                })

    return anomalies


def detect_time_trends(df: pd.DataFrame, date_col: str, value_col: str) -> Optional[Dict[str, Any]]:
    """Detect trends in a time series column pair."""
    if date_col not in df.columns or value_col not in df.columns:
        return None

    try:
        temp = df[[date_col, value_col]].dropna().copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce", format="mixed")
        temp = temp.sort_values(date_col)

        if len(temp) < 5:
            return None

        # Simple linear regression for trend
        x = np.arange(len(temp))
        y = temp[value_col].values
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)

        direction = "upward" if slope > 0 else "downward"
        strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.4 else "weak"

        return {
            "date_col": date_col,
            "value_col": value_col,
            "direction": direction,
            "strength": strength,
            "slope": round(slope, 4),
            "r_squared": round(r_value ** 2, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
            "points": len(temp),
            "date_range": f"{temp[date_col].min().date()} to {temp[date_col].max().date()}",
        }
    except Exception:
        return None


def generate_insights_summary(df: pd.DataFrame) -> List[str]:
    """Generate a list of plain-English insight statements."""
    insights = []
    profile = _quick_profile(df)

    # Size insight
    insights.append(
        f"Dataset has **{profile['rows']:,} rows** and **{profile['cols']} columns** "
        f"({profile['numeric']} numeric, {profile['categorical']} categorical, "
        f"{profile['datetime']} datetime)"
    )

    # Missing data insight
    if profile["missing_total"] > 0:
        insights.append(
            f"**{profile['missing_total']:,} missing values** found ({profile['missing_pct']:.1f}% of all cells) "
            f"across {profile['cols_with_missing']} column(s)"
        )
    else:
        insights.append("✅ **No missing values** — dataset is fully populated")

    # Duplicate insight
    if profile["duplicates"] > 0:
        insights.append(
            f"**{profile['duplicates']:,} duplicate rows** detected ({profile['dup_pct']:.1f}% of data)"
        )
    else:
        insights.append("✅ **No duplicate rows** found")

    # Column-level insights
    for col in profile["high_missing"]:
        insights.append(
            f"⚠️ Column **'{col['col']}'** has **{col['pct']:.1f}% missing** values — consider imputation or review"
        )

    for col in profile["high_cardinality"]:
        insights.append(
            f"🔠 Column **'{col['col']}'** has **{col['unique']:,} unique values** "
            f"({col['unique_pct']:.1f}% of rows) — high cardinality"
        )

    for col in profile["low_cardinality"]:
        insights.append(
            f"🏷️ Column **'{col['col']}'** has only **{col['unique']} unique values** — good for grouping/filtering"
        )

    # Correlation insight
    if profile["top_corr"]:
        c = profile["top_corr"]
        direction = "positive" if c['r'] > 0 else "negative"
        insights.append(
            f"📊 **Top correlation**: '{c['col1']}' and '{c['col2']}' "
            f"(r = {abs(c['r']):.3f}, {direction}, {'strong' if abs(c['r']) > 0.7 else 'moderate'})"
        )

    return insights


def _quick_profile(df: pd.DataFrame) -> dict:
    """Quick profile for insight generation."""
    numeric = list(df.select_dtypes(include="number").columns)
    categorical = list(df.select_dtypes(include=["object", "category"]).columns)
    datetime = list(df.select_dtypes(include="datetime").columns)

    missing_total = int(df.isnull().sum().sum())
    missing_pct = missing_total / (len(df) * len(df.columns)) * 100 if len(df) * len(df.columns) > 0 else 0

    cols_with_missing = [c for c in df.columns if df[c].isnull().sum() > 0]
    duplicates = int(df.duplicated().sum())

    high_missing = []
    high_cardinality = []
    low_cardinality = []

    for col in df.columns:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            pct = nulls / len(df) * 100
            if pct > 10:
                high_missing.append({"col": col, "pct": pct})

        if pd.api.types.is_object_dtype(df[col]):
            unique = df[col].nunique()
            if unique > 100:
                high_cardinality.append({"col": col, "unique": unique, "unique_pct": unique / len(df) * 100})
            elif unique <= 10 and unique > 0:
                low_cardinality.append({"col": col, "unique": unique})

    # Top correlation
    top_corr = None
    numeric_df = df[numeric].dropna(axis=1, how="all")
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        max_val = 0
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = abs(corr.iloc[i, j])
                if not np.isnan(val) and val > max_val:
                    max_val = val
                    top_corr = {
                        "col1": corr.columns[i],
                        "col2": corr.columns[j],
                        "r": corr.iloc[i, j],
                    }

    return {
        "rows": len(df),
        "cols": len(df.columns),
        "numeric": len(numeric),
        "categorical": len(categorical),
        "datetime": len(datetime),
        "missing_total": missing_total,
        "missing_pct": missing_pct,
        "cols_with_missing": len(cols_with_missing),
        "duplicates": duplicates,
        "dup_pct": duplicates / len(df) * 100 if len(df) > 0 else 0,
        "high_missing": high_missing,
        "high_cardinality": high_cardinality,
        "low_cardinality": low_cardinality,
        "top_corr": top_corr,
    }
