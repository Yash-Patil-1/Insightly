"""
Insightly — Multi-Format Data Loader
Supports CSV, Excel (.xlsx/.xls), JSON, Parquet, TSV, and clipboard paste.
"""

import io
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


SUPPORTED_FORMATS = {
    ".csv": "CSV (Comma-separated)",
    ".tsv": "TSV (Tab-separated)",
    ".xlsx": "Excel (.xlsx)",
    ".xls": "Excel (.xls)",
    ".json": "JSON",
    ".parquet": "Parquet",
    ".feather": "Feather",
}


def detect_format(file_name: str) -> str:
    """Detect file format from extension."""
    ext = Path(file_name).suffix.lower()
    return SUPPORTED_FORMATS.get(ext, "Unknown")


def infer_separator(file_name: str) -> str:
    """Infer separator from file extension."""
    ext = Path(file_name).suffix.lower()
    if ext == ".tsv":
        return "\t"
    return ","


def load_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load an uploaded file into a pandas DataFrame.
    Supports CSV, Excel, JSON, Parquet, Feather, TSV.
    """
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name
    ext = Path(file_name).suffix.lower()
    raw = uploaded_file.getvalue()

    try:
        if ext == ".csv":
            return _load_csv(raw, file_name)
        elif ext == ".tsv":
            return pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False)
        elif ext in (".xlsx", ".xls"):
            return _load_excel(raw, file_name)
        elif ext == ".json":
            return _load_json(raw)
        elif ext == ".parquet":
            return pd.read_parquet(io.BytesIO(raw))
        elif ext == ".feather":
            return pd.read_feather(io.BytesIO(raw))
        else:
            st.warning(f"Unsupported format: {ext}")
            return None
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return None


def _load_csv(raw: bytes, file_name: str) -> pd.DataFrame:
    """Load CSV with encoding detection and delimiter sniffing."""
    # Try common encodings
    for enc in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
        try:
            # Try comma first
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
            except Exception:
                # Try semicolon (common in European CSVs)
                return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=";", low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    # Last resort
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="replace", low_memory=False)


def _load_excel(raw: bytes, file_name: str) -> Optional[pd.DataFrame]:
    """Load Excel file, showing sheet selector if multiple sheets."""
    try:
        xls = pd.ExcelFile(io.BytesIO(raw))
        if len(xls.sheet_names) == 1:
            return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
        # Multiple sheets — let user pick
        sheet = st.selectbox("Select sheet:", xls.sheet_names, key="excel_sheet")
        if sheet:
            return pd.read_excel(xls, sheet_name=sheet)
        return None
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None


def _load_json(raw: bytes) -> Optional[pd.DataFrame]:
    """Load JSON (records array or record-oriented dict)."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        return None

    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        # Try orient='records' or just normalize
        try:
            return pd.json_normalize(data)
        except Exception:
            return pd.DataFrame([data])
    else:
        st.error("JSON must be an array of objects or an object.")
        return None


def load_clipboard(text: str) -> Optional[pd.DataFrame]:
    """Load data pasted from clipboard (tab-separated)."""
    if not text or not text.strip():
        return None
    try:
        return pd.read_csv(io.StringIO(text), sep="\t", low_memory=False)
    except Exception as e:
        st.error(f"Could not parse clipboard data: {e}")
        return None


def get_file_info(df: pd.DataFrame) -> dict:
    """Return summary info about the loaded DataFrame."""
    if df is None:
        return {}
    mem = df.memory_usage(deep=True).sum()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "size": _format_bytes(mem),
        "numeric_cols": df.select_dtypes(include="number").shape[1],
        "categorical_cols": df.select_dtypes(include="object").shape[1],
        "datetime_cols": df.select_dtypes(include="datetime").shape[1],
        "bool_cols": df.select_dtypes(include="bool").shape[1],
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_human": _format_bytes(mem),
    }


def _format_bytes(size: int) -> str:
    """Format byte count to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
