"""
Insightly — Natural Language Q&A Engine
Answers plain-English questions about loaded data.
Works out-of-the-box with pattern matching, optionally uses OpenAI for complex queries.
"""

import re
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.visualizer import create_chart, create_correlation_heatmap
from src.analyzer import analyze_correlations, detect_outliers_zscore


# ────────────────────────────────────────────────────────────────
#  PATTERN-BASED QUERY ENGINE
# ────────────────────────────────────────────────────────────────

QUESTION_PATTERNS = {
    # ── Row display ──
    r"(?:show|display|list|view|get|find)\s+(?:me\s+)?(?:the\s+)?(?:top\s+)?(\d+)?\s*(?:rows?|records?|entries?|transactions?|items?)\s*(.*)": "show_rows",
    r"(?:show|display|list|view|get|find)\s+(?:me\s+)?(?:all|every)\s+(?:rows?|records?|entries?|transactions?|items?|data)\s*(.*)": "show_rows",
    r"^(?:show|display|list|view)\s+(?:me\s+)?(?:the\s+)?(?:top\s+)?(\d+)?\s*(.*)": "show_rows",

    # ── Shape / size ──
    r"(?:how many|count|number of)\s+(?:rows|records|entries)\b": "row_count",
    r"(?:how many|count|number of)\s+(?:columns|cols|fields)\b": "col_count",
    r"(?:what is|tell me)\s+(?:the\s+)?(?:shape|size|dimensions?)\b": "shape",
    r"(?:how many|total)\s+(?:rows|records)\s+(?:and|&)\s+(?:columns|cols)": "shape",

    # ── Statistics ──
    r"(?:average|mean|avg)\s+(?:of|for|value of)\s+(.+?)(?:\s+by\s+(.+))?$": "mean",
    r"(?:sum|total)\s+(?:of|for)\s+(.+?)(?:\s+by\s+(.+))?$": "sum",
    r"(?:median|midpoint)\s+(?:of|for|value of)\s+(.+?)(?:\s+by\s+(.+))?$": "median",
    r"(?:min|minimum|lowest|smallest)\s+(?:of|for|value of|value in|)\s*(.+?)(?:\s+by\s+(.+))?$": "min",
    r"(?:max|maximum|highest|largest|biggest)\s+(?:of|for|value of|value in|)\s*(.+?)(?:\s+by\s+(.+))?$": "max",
    r"(?:std|standard deviation|variance|spread)\s+(?:of|for)\s+(.+?)$": "std",
    r"(?:describe|summary|stats|statistics)\s+(?:of|for|on)\s+(.+?)$": "describe_col",
    r"(?:describe|summary|stats|statistics)\s+(?:the\s+)?(?:data|dataset|table)\s*$": "describe_all",
    r"(?:what is|get)\s+(?:the\s+)?(?:range|min and max)\s+(?:of|for)\s+(.+?)$": "range",

    # ── Filtering ──
    r"(?:filter|where|only|shows?)\s+(.+?)\s*(is|are|equals?|==|=|!=|not equal|>|>=|<|<=|greater|less|above|below|more than|less than)\s*(.+?)(?:\s+(?:and|&)\s+(.+?)\s*(is|are|equals?|==|=|!=|>|>=|<|<=|greater|less|above|below)\s*(.+))?$": "filter",

    # ── Missing values ──
    r"(?:missing|null|na|nan)\s+(?:values?|data|entries?|cells?)?\s*(?:in\s+)?(.+?)?$": "missing",
    r"(?:how many|count)\s+(?:missing|null|na|nan)\s+(?:values?|data)": "missing_all",

    # ── Between ──
    r"(.+?)\s+(?:is|are|value|values\s+)?(?:between|from|in range)\s+(\d+\.?\d*)\s+(?:and|to|-)\s+(\d+\.?\d*)": "between",
    r"(.+?)\s*(is|are|equals?|==|=|!=|not equal|>|>=|<|<=|greater than|less than|above|below|more than|less than)\s*(.+?)$": "filter_simple",

    # ── Grouping / aggregation ──
    r"(?:group|grouped)\s+(?:by|per)\s+(.+?)\s+(?:and\s+)?(?:get|show|calculate|find|compute)?\s*(?:the\s+)?(?:average|mean|sum|total|count|max|min|std)\s+(?:of\s+)?(.+?)$": "groupby",
    r"(?:total|sum|count|average|mean)\s+(?:by|per|for each|grouped by)\s+(.+?)$": "groupby_simple",
    r"(?:breakdown|distribution)\s+(?:by|of|per|across)\s+(.+?)$": "distribution",

    # ── Correlation ──
    r"(?:correlation|relationship)\s+(?:between|of)\s+(.+?)\s+(?:and|&|vs\.?|versus)\s+(.+?)$": "correlation_pair",
    r"(?:correlation|relationship|correlations?)\s*(?:matrix|table|analysis)?\s*$": "correlation_all",
    r"(?:how are|does)\s+(.+?)\s+(?:and|&|relate to|correlate with|compare to)\s+(.+?)$": "correlation_pair",

    # ── Outliers ──
    r"(?:outliers?|anomal(?:y|ies))\s+(?:in|for|detected in|of)\s+(.+?)$": "outliers",
    r"(?:outliers?|anomal(?:y|ies))\s*$": "outliers_all",
    r"(?:detect|find|show)\s+(?:outliers?|anomal(?:y|ies))\s*(?:in\s+)?(.+?)?$": "outliers",

    # ── Unique values / cardinality ──
    r"(?:unique|distinct|different)\s+(?:values?|items?|categories?|entries?)\s+(?:of|in|for)\s+(.+?)$": "unique",
    r"(?:how many|count|number of)\s+(?:unique|distinct|different)\s+(?:values?|items?|categories?)\s+(?:in|of|for)\s+(.+?)$": "unique_count",
    r"(?:list|show|what are)\s+(?:all\s+)?(?:the\s+)?(?:unique|distinct|different)\s+(?:values?)\s+(?:of|in|for)\s+(.+?)$": "unique",
    r"(?:list|show|what are)\s+(?:the\s+)?(?:categories?|types?)\s+(?:of|in|for)\s+(.+?)$": "unique",

    # ── Time / trends ──
    r"(?:trend|change|pattern|movement)\s+(?:of|in|for|over)\s+(.+?)\s+(?:over|by|across|throughout)\s+(.+?)$": "trend",
    r"(?:trend|change|pattern)\s+(?:over|across|by|throughout)\s+(.+?)\s+(?:of|in|for)\s+(.+?)$": "trend",
    r"(?:time series|timeline|over time)\s+(?:of|for|of)\s+(.+?)$": "time_series",

    # ── Comparisons ──
    r"(?:compare|comparison|difference)\s+(.+?)\s+(?:and|&|vs\.?|versus|between)\s+(.+?)$": "compare",
    r"(?:compare|comparison)\s+(.+?)\s+(?:by|across|per|grouped by)\s+(.+?)$": "compare_by",

    # ── Chart requests ──
    r"(?:plot|chart|graph|visualize|viz)\s+(?:of|for|a\s+)?(.+?)(?:\s+(?:vs\.?|versus|against|and|by)\s+(.+))?$": "chart",
    r"(?:show|draw|display|generate)\s+(?:a|an|me)\s+(?:(.+?)\s+)?(?:chart|graph|plot|histogram|bar|pie|scatter|line|box)\s*(?:of|for|of|showing)?\s*(.+?)?(?:\s+(?:vs\.?|versus|against|and|by)\s+(.+))?$": "chart_type",

    # ── Data quality ──
    r"(?:data quality|quality|cleanliness|dirty)\s*(?:score|assessment|check|analysis)?\s*$": "data_quality",
    r"(?:duplicate|duplicates|duplicated rows?)\s*(?:rows?|values?|data)?\s*$": "duplicates",
    r"(?:what data|what columns|column names|schema)\s*(?:do|are|is)?\s*(?:i have|available|present|in the dataset)?\s*$": "columns",

    # ── General ──
    r"(?:help|what can i ask|what questions|examples|suggestions)\s*$": "help",
    r"(?:thank|thanks|got it|ok|bye|done)\s*$": "acknowledge",
}


def answer_query(df: pd.DataFrame, question: str, llm_func=None) -> Dict[str, Any]:
    """
    Main entry point. Takes a DataFrame and a plain-English question.
    Returns a dict with 'text' (markdown answer) and optionally 'figure' (plotly chart).

    If llm_func is provided, it's called for complex questions that the
    pattern engine cannot handle. llm_func(question, df_context) should
    return a string answer.
    """
    question = question.strip()
    if not question or not df_loaded(df):
        return {"text": "No data loaded. Please upload or select a dataset first."}

    # Try pattern matching first
    result = _match_pattern(df, question)

    if result and result.get("text"):
        return result

    # Fallback to LLM if available
    if llm_func:
        try:
            answer = llm_func(question)
            if answer:
                return {"text": answer}
        except Exception as e:
            return {"text": f"LLM error: {e}\n\nTry rephrasing your question."}

    # Last resort
    return {
        "text": (
            "I couldn't understand that question. Here are some things you can ask:\n\n"
            "- *\"Show me the top 10 rows\"*\n"
            "- *\"Average of Amount\"*\n"
            "- *\"Total by Category\"*\n"
            "- *\"Filter where Amount > 500\"*\n"
            "- *\"Correlation between Amount and Balance\"*\n"
            "- *\"Missing values in any column\"*\n"
            "- *\"Outliers in Amount\"*\n"
            "- *\"Unique values in Category\"*\n"
            "- *\"Plot Amount by Category\"*\n"
            "- *\"How many rows and columns?\"*"
        )
    }


def df_loaded(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty


def _match_pattern(df: pd.DataFrame, question: str) -> Optional[Dict[str, Any]]:
    """Try to match the question against known patterns."""
    q = question.lower().strip().rstrip("?.")

    for pattern, handler_name in QUESTION_PATTERNS.items():
        match = re.search(pattern, q)
        if match:
            handler = _HANDLERS.get(handler_name)
            if handler:
                try:
                    groups = match.groups()
                    return handler(df, *groups)
                except Exception as e:
                    return {"text": f"Got an error processing that: {e}\n\nTry rephrasing."}

    return None


# ────────────────────────────────────────────────────────────────
#  COLUMN NAME RESOLUTION
# ────────────────────────────────────────────────────────────────

def _resolve_col(df: pd.DataFrame, name: str) -> Optional[str]:
    """Find the actual column name from a fuzzy input."""
    if not name:
        return None
    name = name.strip().strip("'\"")

    # Exact match
    if name in df.columns:
        return name

    # Case-insensitive
    for col in df.columns:
        if col.lower() == name.lower():
            return col

    # Contains
    for col in df.columns:
        if name.lower() in col.lower() or col.lower() in name.lower():
            return col

    # Word-by-word partial
    words = set(name.lower().split())
    best, best_score = None, 0
    for col in df.columns:
        col_words = set(col.lower().replace("_", " ").replace("-", " ").split())
        score = len(words & col_words)
        if score > best_score:
            best_score = score
            best = col

    if best_score > 0:
        return best

    return None


def _resolve_numeric_cols(df: pd.DataFrame, *names) -> List[str]:
    """Resolve column names and filter to numeric only."""
    resolved = []
    for name in names:
        if name:
            col = _resolve_col(df, name)
            if col and pd.api.types.is_numeric_dtype(df[col]):
                resolved.append(col)
    return resolved


def _resolve_date_col(df: pd.DataFrame) -> Optional[str]:
    """Find a datetime or date-like column."""
    for col in df.select_dtypes(include="datetime").columns:
        return col
    # Try object columns that look like dates
    for col in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[col].dropna().head(5), errors="coerce", format="mixed")
            if df[col].dropna().apply(lambda x: bool(re.match(r"\d{4}[-/]\d{2}[-/]\d{2}", str(x)))).mean() > 0.5:
                return col
        except Exception:
            pass
    return None


def _get_numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _get_cat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.select_dtypes(include=["object", "category"]).columns]


def _fmt_val(x: Any) -> str:
    """Format a value nicely for display."""
    if pd.isna(x):
        return "N/A"
    if isinstance(x, float):
        if abs(x) >= 1e6:
            return f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"
        return f"{x:,.4f}".rstrip("0").rstrip(".")
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, datetime):
        return str(x.date()) if hasattr(x, "date") else str(x)
    return str(x)


def _fmt_int(x: Any) -> str:
    return f"{int(x):,}" if pd.notna(x) else "N/A"


def _maybe_chart(df: pd.DataFrame, col_x: str, col_y: str = None, chart_type: str = None) -> Any:
    """Try to create a chart, returning None if it fails."""
    try:
        return create_chart(df, col_x, secondary=col_y, chart_type=chart_type)
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────
#  HANDLER FUNCTIONS
# ────────────────────────────────────────────────────────────────

def _handle_show_rows(df: pd.DataFrame, limit: str = None, extra: str = None) -> Dict[str, Any]:
    """Show top N rows."""
    n = int(limit) if limit and limit.isdigit() else 10
    n = min(n, len(df))
    result = df.head(n)

    text = f"**Showing top {n} of {len(df):,} rows:**\n\n"
    if extra and extra.strip():
        col = _resolve_col(df, extra)
        if col:
            result = result.sort_values(col, ascending=False).head(n)
            text = f"**Top {n} rows sorted by '{col}':**\n\n"

    # Render as markdown table or string
    try:
        table_str = result.to_markdown(index=False)
    except (ImportError, Exception):
        table_str = result.to_string(index=False)
    if len(table_str) > 3000:
        table_str = table_str[:3000] + "\n... (truncated)"
    return {"text": text + table_str}


def _handle_row_count(df: pd.DataFrame, *args) -> Dict[str, Any]:
    return {"text": f"**Total rows:** {len(df):,}"}


def _handle_col_count(df: pd.DataFrame, *args) -> Dict[str, Any]:
    return {"text": f"**Total columns:** {len(df.columns)}"}


def _handle_shape(df: pd.DataFrame, *args) -> Dict[str, Any]:
    return {"text": f"**Shape:** {len(df):,} rows × {len(df.columns)} columns"}


def _handle_mean(df: pd.DataFrame, col_name: str, group_by: str = None) -> Dict[str, Any]:
    col = _resolve_col(df, col_name) if col_name else None
    if not col or not pd.api.types.is_numeric_dtype(df[col]):
        # Try the first numeric column if not specified
        numeric_cols = _get_numeric_cols(df)
        if not numeric_cols:
            return {"text": "No numeric columns found in the dataset."}
        col = numeric_cols[0]
        col_name_used = f"(using '{col}')"
    else:
        col_name_used = f"'{col}'"

    val = df[col].mean()
    text = f"**Average of {col_name_used}:** {_fmt_val(val)}"

    if group_by:
        group_col = _resolve_col(df, group_by)
        if group_col:
            grouped = df.groupby(group_col)[col].mean().sort_values(ascending=False)
            text += f"\n\n**By {group_col}:**\n" + "\n".join(f"- **{k}:** {_fmt_val(v)}" for k, v in grouped.items())
            fig = _maybe_chart(df, group_col, col, "bar")
            if fig:
                return {"text": text, "figure": fig}

    return {"text": text}


def _handle_sum(df: pd.DataFrame, col_name: str, group_by: str = None) -> Dict[str, Any]:
    col = _resolve_col(df, col_name) if col_name else None
    if not col or not pd.api.types.is_numeric_dtype(df[col]):
        numeric_cols = _get_numeric_cols(df)
        if not numeric_cols:
            return {"text": "No numeric columns found."}
        col = numeric_cols[0]

    val = df[col].sum()
    text = f"**Total of '{col}':** {_fmt_val(val)}"

    if group_by:
        group_col = _resolve_col(df, group_by)
        if group_col:
            grouped = df.groupby(group_col)[col].sum().sort_values(ascending=False)
            text += f"\n\n**By {group_col}:**\n" + "\n".join(f"- **{k}:** {_fmt_val(v)}" for k, v in grouped.head(15).items())
            fig = _maybe_chart(df, group_col, col, "bar")
            if fig:
                return {"text": text, "figure": fig}

    return {"text": text}


def _handle_median(df: pd.DataFrame, col_name: str, *args) -> Dict[str, Any]:
    col = _resolve_col(df, col_name) if col_name else _get_numeric_cols(df)
    if isinstance(col, list) and col:
        col = col[0]
    if not col or not pd.api.types.is_numeric_dtype(df[col]):
        return {"text": f"No numeric column found matching '{col_name}'."}
    return {"text": f"**Median of '{col}':** {_fmt_val(df[col].median())}"}


def _handle_min(df: pd.DataFrame, col_name: str, group_by: str = None) -> Dict[str, Any]:
    col = _resolve_col(df, col_name) if col_name else (_get_numeric_cols(df)[0] if _get_numeric_cols(df) else None)
    if not col or not pd.api.types.is_numeric_dtype(df[col]):
        # Try as categorical
        col = _resolve_col(df, col_name)
        if col:
            val = df[col].min()
            return {"text": f"**Minimum of '{col}':** {_fmt_val(val)}"}
        return {"text": f"Could not find column '{col_name}'."}
    val = df[col].min()
    text = f"**Minimum of '{col}':** {_fmt_val(val)}"
    if group_by:
        group_col = _resolve_col(df, group_by)
        if group_col:
            grouped = df.groupby(group_col)[col].min()
            text += "\n\n**By " + group_col + ":**\n" + "\n".join(f"- **{k}:** {_fmt_val(v)}" for k, v in grouped.items())
    return {"text": text}


def _handle_max(df: pd.DataFrame, col_name: str, group_by: str = None) -> Dict[str, Any]:
    col = _resolve_col(df, col_name) if col_name else (_get_numeric_cols(df)[0] if _get_numeric_cols(df) else None)
    if not col or not pd.api.types.is_numeric_dtype(df[col]):
        col = _resolve_col(df, col_name)
        if col:
            val = df[col].max()
            return {"text": f"**Maximum of '{col}':** {_fmt_val(val)}"}
        return {"text": f"Could not find column '{col_name}'."}
    val = df[col].max()
    text = f"**Maximum of '{col}':** {_fmt_val(val)}"
    if group_by:
        group_col = _resolve_col(df, group_by)
        if group_col:
            grouped = df.groupby(group_col)[col].max()
            text += "\n\n**By " + group_col + ":**\n" + "\n".join(f"- **{k}:** {_fmt_val(v)}" for k, v in grouped.items())
    return {"text": text}


def _handle_std(df: pd.DataFrame, col_name: str, *args) -> Dict[str, Any]:
    cols = _resolve_numeric_cols(df, col_name) if col_name else _get_numeric_cols(df)
    if not cols:
        return {"text": "No numeric columns found."}
    text = f"**Standard Deviation of '{cols[0]}':** {_fmt_val(df[cols[0]].std())}"
    return {"text": text}


def _handle_range(df: pd.DataFrame, col_name: str, *args) -> Dict[str, Any]:
    col = _resolve_col(df, col_name) if col_name else (_get_numeric_cols(df)[0] if _get_numeric_cols(df) else None)
    if not col:
        return {"text": f"Could not find column '{col_name}'."}
    if pd.api.types.is_numeric_dtype(df[col]):
        return {"text": f"**Range of '{col}':** {_fmt_val(df[col].min())} to {_fmt_val(df[col].max())}"}
    return {"text": f"**Range of '{col}':** {df[col].min()} to {df[col].max()}"}


def _handle_describe_col(df: pd.DataFrame, col_name: str, *args) -> Dict[str, Any]:
    col = _resolve_col(df, col_name) if col_name else None
    if not col:
        return {"text": f"Could not find column '{col_name}'. Available columns: {', '.join(df.columns[:20])}"}
    s = df[col]
    text = f"### Summary of '{col}'\n\n"
    text += f"- **Type:** {s.dtype}\n"
    text += f"- **Count:** {_fmt_int(s.count())}\n"
    text += f"- **Missing:** {_fmt_int(s.isnull().sum())} ({s.isnull().mean() * 100:.1f}%)\n"
    text += f"- **Unique:** {_fmt_int(s.nunique())}\n"
    if pd.api.types.is_numeric_dtype(s):
        text += f"- **Mean:** {_fmt_val(s.mean())} | **Median:** {_fmt_val(s.median())}\n"
        text += f"- **Std:** {_fmt_val(s.std())} | **IQR:** {_fmt_val(s.quantile(0.75) - s.quantile(0.25))}\n"
        text += f"- **Range:** {_fmt_val(s.min())} to {_fmt_val(s.max())}\n"
        text += f"- **Skew:** {_fmt_val(s.skew())} | **Outliers (IQR):** {_fmt_int(_iqr_outliers(s))}\n"
    else:
        top = s.value_counts().head(5)
        text += "- **Top values:**\n"
        for val, cnt in top.items():
            text += f"  - {val}: {_fmt_int(cnt)} ({cnt / s.count() * 100:.1f}%)\n"
    fig = _maybe_chart(df, col)
    if fig:
        return {"text": text, "figure": fig}
    return {"text": text}


def _handle_describe_all(df: pd.DataFrame, *args) -> Dict[str, Any]:
    text = "### Dataset Summary\n\n"
    for col in df.columns:
        s = df[col]
        text += f"**{col}** ({s.dtype}) — {_fmt_int(s.count())} values, {s.isnull().sum()} missing, {_fmt_int(s.nunique())} unique\n"
        if pd.api.types.is_numeric_dtype(s):
            text += f"  Mean={_fmt_val(s.mean())}, Min={_fmt_val(s.min())}, Max={_fmt_val(s.max())}\n"
        else:
            top = s.value_counts().index[0] if s.count() > 0 else "—"
            text += f"  Most common: {top}\n"
    return {"text": text}


def _handle_filter(df: pd.DataFrame, *groups) -> Dict[str, Any]:
    """Handle filter patterns with 1-2 conditions."""
    # Reconstruct condition with captured operators
    col1 = groups[0] if len(groups) > 0 and groups[0] else ""
    op1 = groups[1] if len(groups) > 1 and groups[1] else ""
    val1 = groups[2] if len(groups) > 2 and groups[2] else ""
    q = f"{col1} {op1} {val1}".strip()
    # Handle AND clause (groups 3, 4, 5 = second column, operator, value)
    if len(groups) > 5 and groups[3] and groups[4] and groups[5]:
        q += f" and {groups[3]} {groups[4]} {groups[5]}"
    return _parse_and_apply_filters(df, q)


def _handle_filter_simple(df: pd.DataFrame, col_part: str, operator: str, val_part: str, *rest) -> Dict[str, Any]:
    """Handle simple 'Column > value' patterns."""
    q = f"{col_part} {operator} {val_part}".strip()
    return _parse_and_apply_filters(df, q)



def _handle_between(df: pd.DataFrame, col_part: str, low_val: str, high_val: str, *args) -> Dict[str, Any]:
    """Handle 'Amount between 100 and 1000' patterns."""
    col = _resolve_col(df, col_part)
    if not col:
        return {"text": f"Could not find column matching '{col_part}'."}
    if not pd.api.types.is_numeric_dtype(df[col]):
        return {"text": f"'{col}' is not numeric. Between filter only works on numeric columns."}

    try:
        low = float(low_val)
        high = float(high_val)
    except ValueError:
        return {"text": f"Could not parse bounds: '{low_val}' or '{high_val}'."}

    if low > high:
        low, high = high, low

    mask = (df[col] >= low) & (df[col] <= high)
    filtered = df[mask]
    n = len(filtered)

    if n == 0:
        return {"text": f"No rows found where **{col}** is between **{_fmt_val(low)}** and **{_fmt_val(high)}**."}

    text = f"Found **{n:,} row(s)** where **{col}** is between **{_fmt_val(low)}** and **{_fmt_val(high)}** ({(n / len(df)) * 100:.1f}% of data):\n\n"
    if n <= 100:
        try:
            text += filtered.to_markdown(index=False)
        except (ImportError, Exception):
            text += filtered.to_string(index=False)
    else:
        try:
            text += filtered.head(20).to_markdown(index=False) + f"\n\n... and {n - 20} more rows."
        except (ImportError, Exception):
            text += filtered.head(20).to_string(index=False) + f"\n\n... and {n - 20} more rows."

    return {"text": text}


def _parse_and_apply_filters(df: pd.DataFrame, condition_str: str) -> Dict[str, Any]:
    """Parse condition string and apply filters."""
    ops = [
        (">=", ">="), ("<=", "<="), ("!=", "!="),
        (">", ">"), ("<", "<"),
        ("==", "=="), ("=", "=="),
        ("greater than or equal to", ">="), ("less than or equal to", "<="),
        ("greater than", ">"), ("less than", "<"),
        ("more than", ">"), ("less than", "<"),
        ("above", ">"), ("below", "<"),
        ("not equal to", "!="), ("not equal", "!="),
        ("equals", "=="), ("is equal to", "=="),
        ("contains", "contains"),
    ]

    for op_text, op_symbol in ops:
        pattern = re.compile(
            r"(.+?)\s*" + re.escape(op_text) + r"\s*(.+)",
            re.IGNORECASE
        )
        match = pattern.search(condition_str)
        if match:
            col_part = match.group(1).strip()
            val_part = match.group(2).strip().strip("'\"")

            col = _resolve_col(df, col_part)
            if not col:
                return {"text": f"Could not find column matching '{col_part}'."}

            # Parse value
            try:
                val = float(val_part)
            except ValueError:
                val = val_part

            try:
                if op_symbol == "contains":
                    mask = df[col].astype(str).str.contains(val_part, case=False, na=False)
                elif op_symbol == "==":
                    mask = df[col] == val
                elif op_symbol == "!=":
                    mask = df[col] != val
                elif op_symbol == ">":
                    mask = df[col] > val
                elif op_symbol == "<":
                    mask = df[col] < val
                elif op_symbol == ">=":
                    mask = df[col] >= val
                elif op_symbol == "<=":
                    mask = df[col] <= val
                else:
                    return {"text": f"Unsupported operator: {op_symbol}"}

                filtered = df[mask]
                n = len(filtered)

                if n == 0:
                    return {"text": f"No rows found where **{col}** {op_text} **{val_part}**."}

                text = f"Found **{n:,} row(s)** where **{col}** {op_text} **{val_part}**:\n\n"
                if n <= 100:
                    text += filtered.to_markdown(index=False) if hasattr(filtered, 'to_markdown') else str(filtered.head(50).to_string(index=False))
                else:
                    text += filtered.head(20).to_markdown(index=False) + f"\n\n... and {n - 20} more rows."

                return {"text": text}
            except Exception as e:
                return {"text": f"Error applying filter: {e}"}

    return {"text": f"Could not parse filter condition from: '{condition_str}'"}


def _handle_groupby(df: pd.DataFrame, group_col_name: str, agg_col_name: str) -> Dict[str, Any]:
    """Group by a column and aggregate another."""
    group_col = _resolve_col(df, group_col_name) if group_col_name else None
    if not group_col:
        return {"text": f"Could not find column for grouping: '{group_col_name}'."}

    agg_col = _resolve_col(df, agg_col_name) if agg_col_name else None
    if not agg_col:
        # Use first numeric
        numeric_cols = _get_numeric_cols(df)
        agg_col = numeric_cols[0] if numeric_cols else None

    if not agg_col:
        return {"text": "Could not find a numeric column to aggregate."}

    grouped = df.groupby(group_col)[agg_col].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False).head(15)
    text = f"**'{agg_col}' grouped by '{group_col}':**\n\n"
    try:
        table_str = grouped.to_markdown()
    except (ImportError, Exception):
        table_str = grouped.to_string()
    text += table_str

    fig = _maybe_chart(df, group_col, agg_col, "bar")
    if fig:
        return {"text": text, "figure": fig}
    return {"text": text}


def _handle_groupby_simple(df: pd.DataFrame, group_col_name: str, *args) -> Dict[str, Any]:
    """Simple aggregation by category (e.g., 'total by category')."""
    group_col = _resolve_col(df, group_col_name)
    if not group_col:
        return {"text": f"Could not find column: '{group_col_name}'."}

    # Detect if question implies a specific aggregation
    q = " ".join(str(a) for a in args if a).lower()

    if "count" in q:
        result = df[group_col].value_counts().head(20)
        text = f"**Count by '{group_col}':**\n" + "\n".join(f"- **{k}:** {_fmt_int(v)} ({v / len(df) * 100:.1f}%)" for k, v in result.items())
        fig = _maybe_chart(df, group_col)
        if fig:
            return {"text": text, "figure": fig}
        return {"text": text}

    # Default: find numeric cols and sum them
    numeric_cols = _get_numeric_cols(df)
    if numeric_cols:
        result = df.groupby(group_col)[numeric_cols[0]].sum().sort_values(ascending=False).head(20)
        text = f"**Total {numeric_cols[0]} by '{group_col}':**\n" + "\n".join(f"- **{k}:** {_fmt_val(v)}" for k, v in result.items())
        fig = _maybe_chart(df, group_col, numeric_cols[0], "bar")
        if fig:
            return {"text": text, "figure": fig}
        return {"text": text}

    return {"text": f"**Count by '{group_col}':**\n" + "\n".join(f"- **{k}:** {v}" for k, v in df[group_col].value_counts().head(20).items())}


def _handle_distribution(df: pd.DataFrame, col_name: str, *args) -> Dict[str, Any]:
    col = _resolve_col(df, col_name)
    if not col:
        return {"text": f"Could not find column: '{col_name}'."}
    vc = df[col].value_counts(normalize=True).head(20)
    text = f"**Distribution of '{col}':**\n"
    for k, v in vc.items():
        text += f"- **{k}:** {v * 100:.1f}%\n"
    fig = _maybe_chart(df, col)
    if fig:
        return {"text": text, "figure": fig}
    return {"text": text}


def _handle_correlation_pair(df: pd.DataFrame, col1_name: str, col2_name: str) -> Dict[str, Any]:
    col1 = _resolve_col(df, col1_name)
    col2 = _resolve_col(df, col2_name)
    if not col1 or not col2:
        return {"text": f"Could not find columns. Tried: '{col1_name}', '{col2_name}'."}
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return {"text": f"Both columns need to be numeric. '{col1}' is {df[col1].dtype}, '{col2}' is {df[col2].dtype}."}
    corr = df[[col1, col2]].dropna().corr().iloc[0, 1]
    text = f"**Correlation between '{col1}' and '{col2}':** {corr:.4f}\n\n"
    if abs(corr) > 0.7:
        text += f"This is a **strong {'positive' if corr > 0 else 'negative'}** relationship."
    elif abs(corr) > 0.4:
        text += f"This is a **moderate {'positive' if corr > 0 else 'negative'}** relationship."
    else:
        text += f"This is a **weak {'positive' if corr > 0 else 'negative'}** relationship."
    fig = _maybe_chart(df, col1, col2, "scatter")
    if fig:
        return {"text": text, "figure": fig}
    return {"text": text}


def _handle_correlation_all(df: pd.DataFrame, *args) -> Dict[str, Any]:
    result = analyze_correlations(df)
    if result["matrix"] is None:
        return {"text": result.get("message", "Not enough numeric columns for correlation analysis.")}
    top = result.get("top", [])
    if not top:
        return {"text": "No significant correlations found."}
    text = "**Top Correlations:**\n\n"
    for i, pair in enumerate(top[:5], 1):
        text += f"{i}. **{pair['col1']}** & **{pair['col2']}**: r = {pair['correlation']:.4f}\n"
        if pair.get("p_value") is not None:
            sig = "significant" if pair["p_value"] < 0.05 else "not significant"
            text += f"   p = {pair['p_value']} ({sig})\n"
    fig = create_correlation_heatmap(result["matrix"])
    return {"text": text, "figure": fig}


def _handle_outliers(df: pd.DataFrame, col_name: str = None, *args) -> Dict[str, Any]:
    if col_name:
        col = _resolve_col(df, col_name)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            text = f"**Outliers in '{col}':** {len(outliers)} ({len(outliers) / df[col].notna().sum() * 100:.1f}%)\n"
            text += f"- IQR bounds: {_fmt_val(lower)} to {_fmt_val(upper)}\n"
            if len(outliers) > 0 and len(outliers) <= 20:
                text += f"- Values: {', '.join(_fmt_val(v) for v in outliers.head(20))}"
            fig = _maybe_chart(df, col, chart_type="box")
            if fig:
                return {"text": text, "figure": fig}
            return {"text": text}
        return {"text": f"Could not find numeric column '{col_name}'."}
    return _handle_outliers_all(df)


def _handle_outliers_all(df: pd.DataFrame, *args) -> Dict[str, Any]:
    result = detect_outliers_zscore(df)
    if not result:
        return {"text": "No significant outliers detected (|z| > 3)."}
    text = "**Outlier Detection (Z-score) Results:**\n\n"
    for col, info in result.items():
        text += f"- **{col}**: {info['count']} outliers ({info['pct']:.1f}%)\n"
    return {"text": text}


def _handle_missing(df: pd.DataFrame, col_name: str = None, *args) -> Dict[str, Any]:
    if col_name and col_name.strip():
        col = _resolve_col(df, col_name)
        if col:
            n = df[col].isnull().sum()
            return {"text": f"**'{col}'** — {_fmt_int(n)} missing values ({n / len(df) * 100:.1f}%)"}
        return {"text": f"Could not find column '{col_name}'."}
    return _handle_missing_all(df)


def _handle_missing_all(df: pd.DataFrame, *args) -> Dict[str, Any]:
    total = int(df.isnull().sum().sum())
    if total == 0:
        return {"text": "✅ **No missing values** in the entire dataset."}
    cols_with_missing = [(c, int(df[c].isnull().sum())) for c in df.columns if df[c].isnull().sum() > 0]
    text = f"**Missing Values:** {_fmt_int(total)} total across {len(cols_with_missing)} column(s)\n\n"
    for col, n in sorted(cols_with_missing, key=lambda x: -x[1]):
        text += f"- **{col}**: {_fmt_int(n)} ({n / len(df) * 100:.1f}%)\n"
    return {"text": text}


def _handle_unique(df: pd.DataFrame, col_name: str, *args) -> Dict[str, Any]:
    col = _resolve_col(df, col_name)
    if not col:
        return {"text": f"Could not find column: '{col_name}'."}
    vals = df[col].dropna().unique()
    text = f"**Unique values in '{col}':** {len(vals):,}\n\n"
    if len(vals) <= 50:
        text += ", ".join(str(v) for v in vals[:50])
    else:
        text += f"Showing first 50 of {len(vals):,}: " + ", ".join(str(v) for v in vals[:50])
    return {"text": text}


def _handle_unique_count(df: pd.DataFrame, col_name: str, *args) -> Dict[str, Any]:
    col = _resolve_col(df, col_name)
    if not col:
        return {"text": f"Could not find column: '{col_name}'."}
    return {"text": f"**Unique values in '{col}':** {_fmt_int(df[col].nunique())}"}


def _handle_trend(df: pd.DataFrame, val_col_name: str, time_col_name: str, *args) -> Dict[str, Any]:
    val_col = _resolve_col(df, val_col_name) if val_col_name else (_get_numeric_cols(df)[0] if _get_numeric_cols(df) else None)
    time_col = _resolve_col(df, time_col_name) if time_col_name else _resolve_date_col(df)

    if not time_col or not val_col:
        return {"text": "Could not find a date column or value column for trend analysis."}
    if not pd.api.types.is_numeric_dtype(df[val_col]):
        return {"text": f"'{val_col}' is not numeric. Trend analysis requires a numeric column."}

    # Convert to datetime if needed
    try:
        temp = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(temp[time_col]):
            temp[time_col] = pd.to_datetime(temp[time_col], errors="coerce", format="mixed")
        temp = temp.dropna(subset=[time_col, val_col]).sort_values(time_col)
        text = f"**Trend of '{val_col}' over '{time_col}':**\n"
        text += f"- **Period:** {temp[time_col].min().date()} to {temp[time_col].max().date()}\n"
        text += f"- **Start value:** {_fmt_val(temp[val_col].iloc[0])}\n"
        text += f"- **End value:** {_fmt_val(temp[val_col].iloc[-1])}\n"
        text += f"- **Overall change:** {_fmt_val(temp[val_col].iloc[-1] - temp[val_col].iloc[0])}\n"
        fig = _maybe_chart(df, time_col, val_col, "line")
        if fig:
            return {"text": text, "figure": fig}
        return {"text": text}
    except Exception as e:
        return {"text": f"Error analyzing trend: {e}"}


def _handle_time_series(df: pd.DataFrame, col_name: str, *args) -> Dict[str, Any]:
    col = _resolve_col(df, col_name) if col_name else None
    time_col = _resolve_date_col(df)
    if not time_col:
        return {"text": "No date/time column found in the dataset."}
    if not col:
        numeric_cols = _get_numeric_cols(df)
        col = numeric_cols[0] if numeric_cols else None
    if not col:
        return {"text": "Could not find a numeric column for time series."}
    fig = _maybe_chart(df, time_col, col, "line")
    if fig:
        return {"text": f"**Time series of '{col}' over '{time_col}'**", "figure": fig}
    return {"text": f"Could not create time series chart for '{col}'."}


def _handle_compare(df: pd.DataFrame, col1_name: str, col2_name: str, *args) -> Dict[str, Any]:
    col1 = _resolve_col(df, col1_name)
    col2 = _resolve_col(df, col2_name)
    if not col1 or not col2:
        return {"text": f"Could not find columns: '{col1_name}', '{col2_name}'."}
    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
        text = f"**Comparison: '{col1}' vs '{col2}'**\n\n"
        text += f"- **{col1}** — Mean: {_fmt_val(df[col1].mean())}, Median: {_fmt_val(df[col1].median())}, Std: {_fmt_val(df[col1].std())}\n"
        text += f"- **{col2}** — Mean: {_fmt_val(df[col2].mean())}, Median: {_fmt_val(df[col2].median())}, Std: {_fmt_val(df[col2].std())}\n"
        fig = _maybe_chart(df, col1, col2, "scatter")
        if fig:
            return {"text": text, "figure": fig}
        return {"text": text}
    text = f"**'{col1}'** values: {', '.join(str(v) for v in df[col1].unique()[:10])}\n\n"
    text += f"**'{col2}'** values: {', '.join(str(v) for v in df[col2].unique()[:10])}"
    fig = _maybe_chart(df, col1, col2)
    if fig:
        return {"text": text, "figure": fig}
    return {"text": text}


def _handle_compare_by(df: pd.DataFrame, col_name: str, group_name: str, *args) -> Dict[str, Any]:
    col = _resolve_col(df, col_name)
    group_col = _resolve_col(df, group_name)
    if not col or not group_col:
        return {"text": f"Could not find columns. Tried: '{col_name}', '{group_name}'."}
    if pd.api.types.is_numeric_dtype(df[col]):
        grouped = df.groupby(group_col)[col].agg(["mean", "median", "std"]).round(2)
        text = f"**'{col}' by '{group_col}':**\n\n" + grouped.to_markdown() if hasattr(grouped, 'to_markdown') else str(grouped.to_string())
        fig = _maybe_chart(df, group_col, col, "box")
        if fig:
            return {"text": text, "figure": fig}
        return {"text": text}
    return {"text": f"'{col}' is not numeric. Try comparing categorical values using 'distribution of {col}'."}


def _handle_chart(df: pd.DataFrame, col_name: str, col2_name: str = None, *args) -> Dict[str, Any]:
    col = _resolve_col(df, col_name)
    if not col:
        numeric_cols = _get_numeric_cols(df)
        cat_cols = _get_cat_cols(df)
        candidates = numeric_cols + cat_cols
        col = candidates[0] if candidates else df.columns[0]

    col2 = _resolve_col(df, col2_name) if col2_name else None

    fig = _maybe_chart(df, col, col2)
    if fig:
        title = f"Chart of '{col}'" + (f" vs '{col2}'" if col2 else "")
        return {"text": f"**{title}**", "figure": fig}
    return {"text": f"Could not create chart for '{col}'."}


def _handle_chart_type(df: pd.DataFrame, *groups) -> Dict[str, Any]:
    """Handle 'show a bar chart of X' or 'pie chart of Y'."""
    q = " ".join(str(g) for g in groups if g)
    chart_keywords = {
        "histogram": "histogram", "bar": "bar", "pie": "pie",
        "scatter": "scatter", "line": "line", "box": "box",
        "violin": "violin", "kde": "kde", "heatmap": "heatmap"
    }

    detected_type = None
    for keyword, chart_type in chart_keywords.items():
        if keyword in q.lower():
            detected_type = chart_type
            break

    # Find column names mentioned
    col_names = []
    for col in df.columns:
        if col.lower() in q.lower():
            col_names.append(col)

    if not col_names:
        col_names = _get_numeric_cols(df)[:2] if _get_numeric_cols(df) else [df.columns[0]]

    col = col_names[0]
    col2 = col_names[1] if len(col_names) > 1 else None

    fig = _maybe_chart(df, col, col2, detected_type)
    if fig:
        title = f"{detected_type.title() if detected_type else 'Chart'} of '{col}'" + (f" vs '{col2}'" if col2 else "")
        return {"text": f"**{title}**", "figure": fig}
    return {"text": f"Could not create {detected_type or 'chart'} for '{col}'."}


def _handle_data_quality(df: pd.DataFrame, *args) -> Dict[str, Any]:
    total_missing = int(df.isnull().sum().sum())
    total_dupes = int(df.duplicated().sum())
    total_cells = len(df) * len(df.columns)
    total_outliers = sum(
        _iqr_outliers(df[col]) for col in _get_numeric_cols(df)
    )

    text = "### Data Quality Assessment\n\n"
    text += f"- **Missing cells:** {_fmt_int(total_missing)} ({total_missing / total_cells * 100:.1f}%)\n"
    text += f"- **Duplicate rows:** {_fmt_int(total_dupes)} ({total_dupes / len(df) * 100:.1f}%)\n"
    text += f"- **Outliers (IQR):** {_fmt_int(total_outliers)}\n"
    text += f"- **Data types:** {len(_get_numeric_cols(df))} numeric, {len(_get_cat_cols(df))} categorical, {len(df.select_dtypes(include='datetime').columns)} datetime\n"

    score = 100
    score -= min(30, total_missing / max(total_cells, 1) * 100)
    score -= min(20, total_dupes / len(df) * 100) if len(df) > 0 else 0
    score -= min(20, total_outliers / max(total_cells, 1) * 100)
    score = max(0, int(score))
    text += f"\n**Overall Quality Score: {score}/100** — {'Good' if score >= 80 else 'Fair' if score >= 50 else 'Poor'}"

    return {"text": text}


def _handle_duplicates(df: pd.DataFrame, *args) -> Dict[str, Any]:
    n = df.duplicated().sum()
    if n == 0:
        return {"text": "✅ **No duplicate rows** found."}
    return {"text": f"**{_fmt_int(n)} duplicate row(s)** found ({n / len(df) * 100:.1f}% of data)."}


def _handle_columns(df: pd.DataFrame, *args) -> Dict[str, Any]:
    text = "**Available Columns:**\n\n"
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()
        nulls = df[col].isnull().sum()
        text += f"- **{col}** (`{dtype}`) — {_fmt_int(nunique)} unique, {_fmt_int(nulls)} missing\n"
    return {"text": text}


def _handle_help(df: pd.DataFrame, *args) -> Dict[str, Any]:
    return {
        "text": (
            "### 💬 Ask Me Anything About Your Data\n\n"
            "Here are examples of questions you can ask:\n\n"
            "**🔍 Explore**\n"
            "- *\"Show me the top 10 rows\"*\n"
            "- *\"How many rows and columns?\"*\n"
            "- *\"What columns do I have?\"*\n\n"
            "**📊 Statistics**\n"
            "- *\"Average of Amount\"*\n"
            "- *\"Sum of Amount by Category\"*\n"
            "- *\"Describe the Age column\"*\n"
            "- *\"Min and max of Balance\"*\n\n"
            "**🔎 Filter**\n"
            "- *\"Where Amount > 500\"*\n"
            "- *\"Category equals Food\"*\n"
            "- *\"Filter Amount between 100 and 1000\"*\n\n"
            "**📈 Analysis**\n"
            "- *\"Correlation between Amount and Balance\"*\n"
            "- *\"Outliers in Amount\"*\n"
            "- *\"Missing values\"*\n"
            "- *\"Trend of Balance over Date\"*\n\n"
            "**🎨 Visualize**\n"
            "- *\"Plot Amount by Category\"*\n"
            "- *\"Histogram of Age\"*\n"
            "- *\"Show a scatter plot of Amount vs Balance\"*\n\n"
            "**📋 Group & Compare**\n"
            "- *\"Total by Category\"*\n"
            "- *\"Compare Amount between Male and Female\"*\n"
            "- *\"Distribution of Payment Method\"*"
        )
    }


def _handle_acknowledge(df: pd.DataFrame, *args) -> Dict[str, Any]:
    return {"text": "You're welcome! 😊 Feel free to ask any other questions about your data."}


# ────────────────────────────────────────────────────────────────
#  UTILITY
# ────────────────────────────────────────────────────────────────

def _iqr_outliers(series: pd.Series) -> int:
    if len(series) < 4:
        return 0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


# ────────────────────────────────────────────────────────────────
#  HANDLER REGISTRY
# ────────────────────────────────────────────────────────────────

_HANDLERS = {
    "show_rows": _handle_show_rows,
    "row_count": _handle_row_count,
    "col_count": _handle_col_count,
    "shape": _handle_shape,
    "mean": _handle_mean,
    "sum": _handle_sum,
    "median": _handle_median,
    "min": _handle_min,
    "max": _handle_max,
    "std": _handle_std,
    "range": _handle_range,
    "describe_col": _handle_describe_col,
    "describe_all": _handle_describe_all,
    "filter": _handle_filter,
    "filter_simple": _handle_filter_simple,
    "between": _handle_between,
    "groupby": _handle_groupby,
    "groupby_simple": _handle_groupby_simple,
    "distribution": _handle_distribution,
    "correlation_pair": _handle_correlation_pair,
    "correlation_all": _handle_correlation_all,
    "outliers": _handle_outliers,
    "outliers_all": _handle_outliers_all,
    "missing": _handle_missing,
    "missing_all": _handle_missing_all,
    "unique": _handle_unique,
    "unique_count": _handle_unique_count,
    "trend": _handle_trend,
    "time_series": _handle_time_series,
    "compare": _handle_compare,
    "compare_by": _handle_compare_by,
    "chart": _handle_chart,
    "chart_type": _handle_chart_type,
    "data_quality": _handle_data_quality,
    "duplicates": _handle_duplicates,
    "columns": _handle_columns,
    "help": _handle_help,
    "acknowledge": _handle_acknowledge,
}


# ────────────────────────────────────────────────────────────────
#  SUGGESTED QUESTIONS
# ────────────────────────────────────────────────────────────────

def get_suggestions(df: pd.DataFrame) -> List[str]:
    """Generate context-aware question suggestions based on the loaded data."""
    suggestions = []
    numeric_cols = _get_numeric_cols(df)
    cat_cols = _get_cat_cols(df)
    date_col = _resolve_date_col(df)

    suggestions.append("Show me the top 10 rows")
    suggestions.append("How many rows and columns?")

    if numeric_cols:
        suggestions.append(f"Average of {numeric_cols[0]}")
        suggestions.append(f"Sum of {numeric_cols[0]}")
        if len(numeric_cols) >= 2:
            suggestions.append(f"Correlation between {numeric_cols[0]} and {numeric_cols[1]}")
        suggestions.append(f"Outliers in {numeric_cols[0]}")

    if cat_cols:
        suggestions.append(f"Total by {cat_cols[0]}")
        suggestions.append(f"Distribution of {cat_cols[0]}")
        if numeric_cols:
            suggestions.append(f"Plot {numeric_cols[0]} by {cat_cols[0]}")

    suggestions.append("Missing values")
    suggestions.append("Describe the dataset")

    if date_col and numeric_cols:
        suggestions.append(f"Trend of {numeric_cols[0]} over time")

    return suggestions[:10]
