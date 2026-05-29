"""
Insightly — Data Cleaner
One-click operations: drop duplicates, handle missing values, fix types, remove outliers.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple


def detect_issues(df: pd.DataFrame) -> Dict[str, list]:
    """Scan DataFrame and return a list of detected quality issues."""
    issues = {"missing": [], "duplicates": [], "types": [], "outliers": []}

    # Missing values
    for col in df.columns:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            pct = nulls / len(df) * 100
            issues["missing"].append({
                "column": col,
                "count": int(nulls),
                "pct": round(pct, 2),
            })

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues["duplicates"].append({
            "count": int(dup_count),
            "pct": round(dup_count / len(df) * 100, 2),
        })

    # Type issues (mixed types, object columns that look numeric)
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            numeric_ratio = df[col].dropna().astype(str).str.match(r"^-?\d+\.?\d*$").mean()
            if numeric_ratio > 0.8 and numeric_ratio < 1.0:
                issues["types"].append({
                    "column": col,
                    "issue": f"Mixed types — {numeric_ratio:.0%} numeric, rest text",
                })

    # Outliers (IQR)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        clean = df[col].dropna()
        if len(clean) > 0:
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((clean < lower) | (clean > upper)).sum()
            if outliers > 0:
                issues["outliers"].append({
                    "column": col,
                    "count": int(outliers),
                    "pct": round(outliers / len(clean) * 100, 2),
                    "bounds": {"lower": round(lower, 4), "upper": round(upper, 4)},
                })

    return issues


def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = False,
    fill_missing: str = "none",
    fix_types: bool = False,
    remove_outliers: bool = False,
    outlier_method: str = "iqr",
    columns_to_clean: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply selected cleaning operations to a DataFrame.
    Returns (cleaned_df, log_of_changes).
    """
    cleaned = df.copy()
    log = []

    cols = columns_to_clean if columns_to_clean else list(df.columns)

    # 1. Drop duplicates
    if drop_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        removed = before - len(cleaned)
        if removed > 0:
            log.append(f"Dropped {removed} duplicate row(s)")

    # 2. Fill/remove missing values
    if fill_missing in ("mean", "median", "mode", "drop"):
        if fill_missing == "drop":
            before = len(cleaned)
            cleaned = cleaned.dropna(subset=cols)
            removed = before - len(cleaned)
            if removed > 0:
                log.append(f"Dropped {removed} row(s) with missing values")
        else:
            for col in cols:
                if cleaned[col].isnull().sum() > 0:
                    if fill_missing == "mean" and pd.api.types.is_numeric_dtype(cleaned[col]):
                        val = cleaned[col].mean()
                        cleaned[col] = cleaned[col].fillna(val)
                        log.append(f"Filled '{col}' missing with mean ({val:.2f})")
                    elif fill_missing == "median" and pd.api.types.is_numeric_dtype(cleaned[col]):
                        val = cleaned[col].median()
                        cleaned[col] = cleaned[col].fillna(val)
                        log.append(f"Filled '{col}' missing with median ({val:.2f})")
                    elif fill_missing == "mode":
                        mode_vals = cleaned[col].mode()
                        if len(mode_vals) > 0:
                            val = mode_vals[0]
                            cleaned[col] = cleaned[col].fillna(val)
                            log.append(f"Filled '{col}' missing with mode ({val})")

    # 3. Fix types (convert object -> numeric/datetime where possible)
    if fix_types:
        for col in cols:
            if pd.api.types.is_object_dtype(cleaned[col]) or pd.api.types.is_string_dtype(cleaned[col]):
                # Try numeric
                try:
                    converted = pd.to_numeric(cleaned[col], errors="coerce")
                    if converted.notna().sum() > cleaned[col].notna().sum() * 0.8:
                        cleaned[col] = converted
                        log.append(f"Converted '{col}' to numeric")
                        continue
                except Exception:
                    pass
                # Try datetime
                try:
                    converted = pd.to_datetime(cleaned[col], errors="coerce", format="mixed")
                    if converted.notna().sum() > cleaned[col].notna().sum() * 0.8:
                        cleaned[col] = converted
                        log.append(f"Converted '{col}' to datetime")
                except Exception:
                    pass

    # 4. Remove outliers
    if remove_outliers:
        numeric_cols = cleaned.select_dtypes(include="number").columns
        mask = pd.Series(True, index=cleaned.index)
        for col in numeric_cols:
            if outlier_method == "iqr":
                clean = cleaned[col].dropna()
                if len(clean) > 0:
                    q1 = clean.quantile(0.25)
                    q3 = clean.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    col_mask = (cleaned[col] >= lower) & (cleaned[col] <= upper)
                    mask = mask & col_mask.fillna(True)
        before = len(cleaned)
        cleaned = cleaned[mask]
        removed = before - len(cleaned)
        if removed > 0:
            log.append(f"Removed {removed} outlier row(s) using IQR method")

    return cleaned, log


def fill_missing_options() -> List[str]:
    """Return available fill strategies."""
    return ["none", "drop", "mean", "median", "mode"]
