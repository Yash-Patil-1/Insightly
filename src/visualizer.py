"""
Insightly — Smart Visualizer
Auto-selects the best chart type based on data types and column combinations.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Tuple


def suggest_chart_type(col_data: pd.Series, secondary: Optional[pd.Series] = None) -> str:
    """
    Suggest the best chart type based on column data type and cardinality.
    """
    if secondary is not None:
        # Two-column visualization
        if pd.api.types.is_numeric_dtype(col_data) and pd.api.types.is_numeric_dtype(secondary):
            return "scatter"
        if pd.api.types.is_numeric_dtype(col_data) and _is_categorical(secondary):
            return "box"
        if _is_categorical(col_data) and pd.api.types.is_numeric_dtype(secondary):
            return "bar"
        return "scatter"

    # Single-column visualization
    if pd.api.types.is_numeric_dtype(col_data):
        if col_data.nunique() <= 5:
            return "bar"
        return "histogram"

    if pd.api.types.is_datetime64_any_dtype(col_data):
        return "histogram"

    if _is_categorical(col_data):
        nunique = col_data.nunique()
        if nunique <= 10:
            return "pie"
        elif nunique <= 30:
            return "bar"
        return "bar"  # top 20 bar

    return "bar"


def _is_categorical(series: pd.Series) -> bool:
    """Check if a series should be treated as categorical."""
    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) and series.nunique() <= 10:
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    return False


def create_chart(
    df: pd.DataFrame,
    col: str,
    secondary: Optional[str] = None,
    chart_type: Optional[str] = None,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Create a plotly figure for the specified column(s) and chart type."""
    if chart_type is None:
        chart_type = suggest_chart_type(
            df[col], df[secondary] if secondary else None
        )

    title = title or f"{col}" + (f" vs {secondary}" if secondary else "")

    try:
        if secondary:
            return _create_multi_chart(df, col, secondary, chart_type, color_col, title)
        return _create_single_chart(df, col, chart_type, title)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Could not render chart: {e}", showarrow=False)
        return fig


def _create_single_chart(df: pd.DataFrame, col: str, chart_type: str, title: str) -> go.Figure:
    """Create a single-column chart."""
    clean = df[col].dropna()

    if chart_type == "histogram":
        fig = px.histogram(
            clean, x=col,
            title=title,
            labels={col: col},
            color_discrete_sequence=["#2563eb"],
            marginal="box",
            nbins=min(50, max(10, int(clean.nunique()))),
        )
        fig.update_layout(bargap=0.05)

    elif chart_type == "bar":
        vc = clean.value_counts().head(20).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(
            vc, x=col, y="count",
            title=title,
            labels={"count": "Count"},
            color="count",
            color_continuous_scale="Blues",
        )

    elif chart_type == "pie":
        vc = clean.value_counts().reset_index()
        vc.columns = [col, "count"]
        fig = px.pie(
            vc, names=col, values="count",
            title=title,
            hole=0.4,
        )

    elif chart_type == "box":
        fig = px.box(
            clean, y=col,
            title=title,
            points="outliers",
            color_discrete_sequence=["#2563eb"],
        )

    else:
        fig = px.histogram(clean, x=col, title=title)

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    return fig


def _create_multi_chart(
    df: pd.DataFrame, col: str, secondary: str, chart_type: str,
    color_col: Optional[str], title: str,
) -> go.Figure:
    """Create a two-column chart."""
    plot_df = df[[col, secondary]].dropna()

    if color_col and color_col in df.columns:
        plot_df[color_col] = df.loc[plot_df.index, color_col]

    if chart_type == "scatter":
        fig = px.scatter(
            plot_df, x=col, y=secondary,
            color=color_col,
            title=title,
            opacity=0.7,
            trendline="ols" if len(plot_df) > 10 else None,
            labels={col: col, secondary: secondary},
        )

    elif chart_type == "box":
        cat_col = col if _is_categorical(df[col]) else secondary
        num_col = secondary if cat_col == col else col
        fig = px.box(
            plot_df, x=cat_col, y=num_col,
            color=cat_col,
            title=title,
            points="outliers",
        )

    elif chart_type == "bar":
        cat_col = col if _is_categorical(df[col]) else secondary
        num_col = secondary if cat_col == col else col
        aggr = plot_df.groupby(cat_col)[num_col].mean().reset_index().sort_values(num_col, ascending=False).head(20)
        fig = px.bar(
            aggr, x=cat_col, y=num_col,
            title=title,
            color=num_col,
            color_continuous_scale="Blues",
            labels={cat_col: cat_col, num_col: f"Mean {num_col}"},
        )

    elif chart_type == "line":
        fig = px.line(
            plot_df.sort_values(col), x=col, y=secondary,
            color=color_col,
            title=title,
            markers=True,
        )

    else:
        fig = px.scatter(plot_df, x=col, y=secondary, title=title)

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create a correlation heatmap figure."""
    if corr_matrix is None or corr_matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="No correlation data available", showarrow=False)
        return fig

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Correlation Matrix",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=80, r=20, t=40, b=80),
        height=max(400, len(corr_matrix.columns) * 50),
    )
    return fig


def create_missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a missing value heatmap (light = present, dark = missing)."""
    missing = df.isnull().astype(int)
    fig = px.imshow(
        missing.T,
        color_continuous_scale="gray",
        aspect="auto",
        title="Missing Values (white = present, black = missing)",
        labels={"x": "Row index", "y": "Column"},
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=80, r=20, t=40, b=20),
        height=max(200, len(df.columns) * 30),
    )
    fig.update_xaxes(showticklabels=False)
    return fig


def auto_gallery(df: pd.DataFrame, max_charts: int = 8) -> List[Tuple[str, str, go.Figure]]:
    """Auto-generate a gallery of the most interesting charts."""
    charts = []
    numeric_cols = df.select_dtypes(include="number").columns[:5]
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if df[c].nunique() <= 30][:3]
    datetime_cols = df.select_dtypes(include="datetime").columns[:2]

    # Numeric histograms
    for col in numeric_cols:
        if len(charts) >= max_charts:
            break
        fig = create_chart(df, col, chart_type="histogram", title=f"Distribution of {col}")
        charts.append((col, "histogram", fig))

    # Categorical pie/bar
    for col in cat_cols:
        if len(charts) >= max_charts:
            break
        fig = create_chart(df, col, chart_type="pie", title=f"Breakdown of {col}")
        charts.append((col, "pie", fig))

    # Scatter for top numeric pair
    if len(numeric_cols) >= 2 and len(charts) < max_charts:
        fig = create_chart(df, numeric_cols[0], secondary=numeric_cols[1], chart_type="scatter")
        charts.append((f"{numeric_cols[0]} vs {numeric_cols[1]}", "scatter", fig))

    # Box plots for categorical vs numeric
    if len(cat_cols) > 0 and len(numeric_cols) > 0 and len(charts) < max_charts:
        fig = create_chart(df, cat_cols[0], secondary=numeric_cols[0], chart_type="box")
        charts.append((f"{numeric_cols[0]} by {cat_cols[0]}", "box", fig))

    # Time series
    if len(datetime_cols) > 0 and len(numeric_cols) > 0 and len(charts) < max_charts:
        fig = create_chart(df, datetime_cols[0], secondary=numeric_cols[0], chart_type="line")
        charts.append((f"{numeric_cols[0]} over time", "line", fig))

    return charts
