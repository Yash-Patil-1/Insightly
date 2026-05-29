"""
Insightly — Report Generator
Generates a plain-English narrative report about the dataset.
"""

import pandas as pd
from typing import Any, Dict, List
from datetime import datetime


def generate_report(df: pd.DataFrame, profile: Dict[str, Any]) -> str:
    """
    Generate a comprehensive plain-English report from a DataFrame and its profile.
    """
    if df is None or df.empty:
        return "No data loaded."

    overall = profile.get("overall", {})
    columns = profile.get("columns", {})

    lines = []
    lines.append("# 📊 Insightly Data Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Rows:** {overall.get('rows', 0):,} | **Columns:** {overall.get('columns', 0)}")
    lines.append("")

    # ── Section 1: Overview ──
    lines.append("---\n")
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"This dataset contains **{overall.get('rows', 0):,} rows** and **{overall.get('columns', 0)} columns**.")
    lines.append(f"Total memory usage: **{overall.get('memory_used', 'N/A')}**.")

    missing = overall.get('missing_cells', 0)
    if missing > 0:
        lines.append(f"There are **{missing:,} missing values** ({overall.get('missing_pct', 0):.1f}% of all cells).")
    else:
        lines.append("✅ The dataset is **fully populated** with no missing values.")

    dups = overall.get('duplicate_rows', 0)
    if dups > 0:
        lines.append(f"**{dups:,} duplicate rows** detected ({overall.get('duplicate_pct', 0):.1f}% of data).")
    else:
        lines.append("✅ No duplicate rows found.")
    lines.append("")

    # ── Section 2: Column Details ──
    lines.append("---\n")
    lines.append("## 2. Column-by-Column Analysis\n")

    for col_name, info in columns.items():
        inferred = info.get("inferred_type", "Unknown")
        dtype = info.get("dtype", "?")
        lines.append(f"\n### {col_name}")
        lines.append(f"- **Type:** {inferred} (`{dtype}`)")
        lines.append(f"- **Non-null count:** {info.get('count', 0):,} / {info.get('count', 0) + info.get('nulls', 0):,}")
        lines.append(f"- **Missing:** {info.get('nulls', 0):,} ({info.get('null_pct', 0):.1f}%)")
        lines.append(f"- **Unique values:** {info.get('unique', 0):,} ({info.get('unique_pct', 0):.1f}%)")

        # Numeric details
        if "mean" in info:
            lines.append(f"- **Range:** {_fmt(info.get('min'))} to {_fmt(info.get('max'))}")
            lines.append(f"- **Mean:** {_fmt(info.get('mean'))} | **Median:** {_fmt(info.get('median'))}")
            lines.append(f"- **Std Dev:** {_fmt(info.get('std'))} | **IQR:** {_fmt(info.get('iqr'))}")
            lines.append(f"- **Skewness:** {_fmt(info.get('skew'))} | **Zeros:** {info.get('zeros', 0):,}")
            if info.get('outliers_iqr', 0) > 0:
                lines.append(f"- ⚠️ **{info['outliers_iqr']} outlier(s)** detected via IQR")

        # Categorical details
        if "top_value" in info:
            lines.append(f"- **Most common:** '{info['top_value']}' ({info['top_pct']:.1f}% of non-null values)")
            top_vals = info.get("top_values", {})
            if top_vals:
                items = list(top_vals.items())[:3]
                lines.append(f"- **Top values:** {', '.join(f'{k} ({v})' for k, v in items)}")
            if info.get("empty_strings", 0) > 0:
                lines.append(f"- ⚠️ **{info['empty_strings']} empty string(s)** found")

        # Datetime details
        if "min_date" in info:
            lines.append(f"- **Date range:** {info['min_date']} to {info['max_date']} ({info['range_days']:,} days)")

    # ── Section 3: Data Quality ──
    lines.append("\n---\n")
    lines.append("## 3. Data Quality Assessment\n")

    # Count issues
    high_missing = [c for c, info in columns.items() if info.get("null_pct", 0) > 10]
    outlier_cols = [c for c, info in columns.items() if info.get("outliers_iqr", 0) > 0]
    high_cardinality = [c for c, info in columns.items()
                        if info.get("unique_pct", 0) > 50 and info.get("inferred_type") == "Text / High Cardinality"]

    if high_missing:
        lines.append(f"- ⚠️ **High missing rates** in: {', '.join(high_missing)}")
    if outlier_cols:
        lines.append(f"- ⚠️ **Outliers detected** in: {', '.join(outlier_cols)}")
    if high_cardinality:
        lines.append(f"- 🔠 **High cardinality columns** (may need feature engineering): {', '.join(high_cardinality)}")
    if not high_missing and not outlier_cols:
        lines.append("- ✅ Data quality looks **good** — no major issues detected")

    # ── Section 4: Recommendations ──
    lines.append("\n---\n")
    lines.append("## 4. Recommendations\n")

    recs = _generate_recommendations(df, columns)
    if recs:
        for rec in recs:
            lines.append(f"- {rec}")
    else:
        lines.append("- No specific recommendations — data looks clean and ready for analysis.")

    lines.append("\n---\n")
    lines.append(f"*Report generated by Insightly on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines)


def _fmt(val: Any) -> str:
    """Format a value nicely."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:,.4f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _generate_recommendations(df: pd.DataFrame, columns: Dict[str, Any]) -> List[str]:
    """Generate data-specific recommendations."""
    recs = []

    for col, info in columns.items():
        # Recommendation: fill missing values
        if info.get("null_pct", 0) > 20:
            if "mean" in info:
                recs.append(f"Consider imputing **'{col}'** with the **mean** ({_fmt(info['mean'])}) or **median** ({_fmt(info['median'])})")
            else:
                recs.append(f"Consider filling or dropping rows with missing **'{col}'** values")

        # Recommendation: remove outliers
        if info.get("outliers_iqr", 0) > 0 and info.get("count", 0) > 0:
            outlier_pct = info["outliers_iqr"] / info["count"] * 100
            if outlier_pct > 5:
                recs.append(f"**{col}** has {info['outliers_iqr']} outliers ({outlier_pct:.1f}%) — consider winsorization or review")

        # Recommendation: cardinality
        if info.get("unique_pct", 0) > 80 and info.get("inferred_type") == "Text / High Cardinality":
            recs.append(f"**'{col}'** is high-cardinality text — consider feature extraction (TF-IDF, embeddings)")

        # Recommendation: skew
        if "skew" in info and info["skew"] is not None:
            abs_skew = abs(info["skew"])
            if abs_skew > 2:
                recs.append(f"**'{col}'** is highly skewed (skew={info['skew']:.2f}) — log or Box-Cox transformation may help")

        # Recommendation: zeros
        if info.get("zeros", 0) > 0 and info.get("count", 0) > 0:
            zero_pct = info["zeros"] / info["count"] * 100
            if zero_pct > 30:
                recs.append(f"**'{col}'** is {zero_pct:.0f}% zeros — consider zero-inflated models or indicator flag")

    # Deduplicate
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            unique_recs.append(r)

    return unique_recs[:10]  # limit to top 10
