"""
Insightly — Swiss Army Knife for Data Analysis
Drop any structured data file and get instant insights.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

# ---- Path setup ----
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import load_file, load_clipboard, get_file_info, SUPPORTED_FORMATS
from src.profiler import profile_dataframe
from src.cleaner import clean_dataframe, detect_issues, fill_missing_options
from src.analyzer import (
    analyze_correlations,
    detect_outliers_zscore,
    detect_distribution_anomalies,
    generate_insights_summary,
)
from src.visualizer import (
    create_chart,
    create_correlation_heatmap,
    create_missing_heatmap,
    auto_gallery,
    suggest_chart_type,
)
from src.reporter import generate_report
from src.theme import apply_theme, theme_selector
from src.qa import answer_query, get_suggestions

# ── Page Config ──
st.set_page_config(
    page_title="Insightly",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apply Theme ──
apply_theme()

# ── Session State ──
if "df" not in st.session_state:
    st.session_state.df = None
if "df_name" not in st.session_state:
    st.session_state.df_name = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "cleaning_log" not in st.session_state:
    st.session_state.cleaning_log = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = []
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""


# ── Helpers ──
def reset_data():
    st.session_state.df = None
    st.session_state.df_name = None
    st.session_state.cleaned_df = None
    st.session_state.cleaning_log = []


def kpi(label: str, value: str, delta: str = None):
    st.markdown(
        f"""<div class="insightly-kpi">
            <div class="insightly-kpi-label">{label}</div>
            <div class="insightly-kpi-value">{value}</div>
            {f'<div class="insightly-metric-change">{delta}</div>' if delta else ''}
        </div>""",
        unsafe_allow_html=True,
    )


def card(content: str):
    st.markdown(f'<div class="insightly-card">{content}</div>', unsafe_allow_html=True)


def load_sample_data(name: str):
    """Load sample data from sample_data folder."""
    path = Path(__file__).parent / "sample_data"
    try:
        if name == "Personal Finance":
            return pd.read_csv(path / "personal_finance.csv")
        elif name == "Sales Report":
            return pd.read_excel(path / "sales_report.xlsx")
        elif name == "Survey Results":
            return pd.read_json(path / "survey_results.json")
    except Exception as e:
        st.error(f"Could not load sample '{name}': {e}")
        return None
    return None


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 2.5rem;">🔍</span>
            <h1 style="margin: 0; font-size: 1.5rem;">Insightly</h1>
            <p style="color: var(--insightly-muted); font-size: 0.8rem; margin: 0;">
                Swiss Army Knife for Data
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Theme Selector ──
    theme_selector()

    st.divider()

    # ── Data Source ──
    st.markdown("### 📂 Data Source")

    # Sample data
    sample = st.selectbox(
        "Load sample data",
        ["None", "Personal Finance", "Sales Report", "Survey Results"],
        key="sample_selector",
    )

    if sample != "None" and sample != st.session_state.get("last_sample"):
        with st.spinner(f"Loading {sample}..."):
            df = load_sample_data(sample)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_name = f"📁 {sample}"
                st.session_state.cleaned_df = None
                st.session_state.last_sample = sample
                st.success(f"Loaded {sample} ({len(df):,} rows × {len(df.columns)} cols)")
                st.rerun()

    if sample == "None" and "last_sample" in st.session_state:
        del st.session_state.last_sample

    st.markdown("**— or —**")

    # File upload
    uploaded = st.file_uploader(
        "Upload a file",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet", "feather"],
        help=f"Supported: {', '.join(SUPPORTED_FORMATS.values())}",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        with st.spinner(f"Loading {uploaded.name}..."):
            df = load_file(uploaded)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_name = f"📁 {uploaded.name}"
                st.session_state.cleaned_df = None
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
                st.rerun()

    # Clipboard paste
    with st.expander("📋 Paste data"):
        pasted = st.text_area("Paste tab-separated data:", height=100, label_visibility="collapsed")
        if pasted:
            df = load_clipboard(pasted)
            if df is not None:
                st.session_state.df = df
                st.session_state.df_name = "📋 Clipboard"
                st.session_state.cleaned_df = None
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
                st.rerun()

    st.divider()

    # ── AI Settings ──
    with st.expander("🤖 AI Settings (Optional)"):
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Optional. Enter your OpenAI API key for smarter answers to complex questions.",
            key="openai_key_input",
        )
        if api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key
            st.rerun()
        st.caption("Without a key, Insightly uses built-in pattern matching to answer your questions.")

    # ── Quick Actions ──
    if st.session_state.df is not None:
        st.markdown("### ⚡ Actions")
        if st.button("🔄 Reset Data", use_container_width=True):
            # Clear Q&A history too
            st.session_state.qa_messages = []
            reset_data()
            st.rerun()

    # ── Footer ──
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: var(--insightly-muted); font-size: 0.7rem;">
            Built by <strong>Yash Patil</strong><br>
            v1.0
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

# ── Welcome screen ──
if st.session_state.df is None:
    st.markdown(
        """
        <div style="text-align: center; padding: 4rem 2rem;">
            <span style="font-size: 5rem;">🔍</span>
            <h1 style="font-size: 2.5rem; margin: 0.5rem 0;">Welcome to Insightly</h1>
            <p style="font-size: 1.1rem; color: var(--insightly-muted); max-width: 500px; margin: 0 auto 1.5rem;">
                Drop any structured data file — CSV, Excel, JSON, Parquet, or paste from clipboard —
                and get instant insights, visualizations, and a plain-English report.
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-top: 1rem;">
                <div class="insightly-card" style="width: 180px;">
                    <div style="font-size: 2rem;">📂</div>
                    <div style="font-weight: 600;">Load Data</div>
                    <div style="font-size: 0.8rem; color: var(--insightly-muted);">CSV, Excel, JSON, more</div>
                </div>
                <div class="insightly-card" style="width: 180px;">
                    <div style="font-size: 2rem;">📊</div>
                    <div style="font-weight: 600;">Auto-Profile</div>
                    <div style="font-size: 0.8rem; color: var(--insightly-muted);">Stats for every column</div>
                </div>
                <div class="insightly-card" style="width: 180px;">
                    <div style="font-size: 2rem;">🧹</div>
                    <div style="font-weight: 600;">Clean</div>
                    <div style="font-size: 0.8rem; color: var(--insightly-muted);">Fix missing, dupes, outliers</div>
                </div>
                <div class="insightly-card" style="width: 180px;">
                    <div style="font-size: 2rem;">💡</div>
                    <div style="font-weight: 600;">Insights</div>
                    <div style="font-size: 0.8rem; color: var(--insightly-muted);">Patterns & correlations</div>
                </div>
            </div>
            <p style="margin-top: 2rem; color: var(--insightly-muted); font-size: 0.9rem;">
                👈 Select a sample dataset or upload a file from the sidebar to get started
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Data loaded — show tabs ──
df = st.session_state.df
df_name = st.session_state.df_name or "Dataset"

# Show current dataset info
st.markdown(
    f"""<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
        <span style="font-size: 1.2rem;">{df_name}</span>
        <span class="insightly-badge insightly-badge-success">{len(df):,} rows</span>
        <span class="insightly-badge insightly-badge-warning">{len(df.columns)} cols</span>
    </div>""",
    unsafe_allow_html=True,
)

# ── Tabs ──
t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
    ["📂 Overview", "🔬 Profile", "🧹 Clean", "💡 Insights", "📈 Visualize", "📋 Report", "📥 Export", "💬 Ask AI"]
)

# ═══════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════

with t1:
    info = get_file_info(df)
    profile = profile_dataframe(df)
    overall = profile.get("overall", {})

    # ── KPI Row ──
    cols = st.columns(5)
    with cols[0]: kpi("Rows", f"{info.get('rows', 0):,}")
    with cols[1]: kpi("Columns", f"{info.get('columns', 0)}")
    with cols[2]: kpi("Missing", f"{info.get('missing_cells', 0):,}", f"{overall.get('missing_pct', 0):.1f}%")
    with cols[3]: kpi("Duplicates", f"{info.get('duplicate_rows', 0):,}", f"{overall.get('duplicate_pct', 0):.1f}%")
    with cols[4]: kpi("Size", info.get("memory_human", "N/A"))

    # ── Data Type Breakdown ──
    st.markdown('<div class="insightly-header">📊 Data Types</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        type_counts = {
            "Numeric": info.get("numeric_cols", 0),
            "Categorical": info.get("categorical_cols", 0),
            "Datetime": info.get("datetime_cols", 0),
            "Boolean": info.get("bool_cols", 0),
        }
        type_df = pd.DataFrame(
            {"Type": list(type_counts.keys()), "Count": list(type_counts.values())}
        ).query("Count > 0")
        if not type_df.empty:
            fig = px.pie(type_df, names="Type", values="Count", hole=0.4, title="Column Types")
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="insightly-subheader">Column Preview</div>', unsafe_allow_html=True)
        preview = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Missing %": [f"{df[c].isnull().mean() * 100:.1f}%" for c in df.columns],
            "Unique": [f"{df[c].nunique():,}" for c in df.columns],
        })
        st.dataframe(preview, use_container_width=True, hide_index=True)

    # ── Missing Value Map ──
    missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    if missing_cols:
        st.markdown('<div class="insightly-header">🔍 Missing Value Map</div>', unsafe_allow_html=True)
        if len(missing_cols) <= 30:
            fig = create_missing_heatmap(df[missing_cols])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Too many columns ({len(missing_cols)}) to display missing value heatmap.")

    # ── Data Preview ──
    st.markdown('<div class="insightly-header">👁️ Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2: PROFILE
# ═══════════════════════════════════════════════════════════════

with t2:
    profile = profile_dataframe(df)
    columns = profile.get("columns", {})
    overall = profile.get("overall", {})

    st.markdown(f'<div class="insightly-header">🔬 Column Profile</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="insightly-subheader">Detailed statistics for all {len(columns)} columns</div>',
        unsafe_allow_html=True,
    )

    col_names = list(columns.keys())

    # Quick stat summary
    stat_rows = []
    for col in col_names:
        info = columns[col]
        stat_rows.append({
            "Column": col,
            "Type": info.get("inferred_type", ""),
            "Non-null": f"{info.get('count', 0):,}",
            "Missing": f"{info.get('null_pct', 0):.1f}%",
            "Unique": f"{info.get('unique', 0):,}",
            "Mean": f"{info.get('mean', '-'):.2f}" if "mean" in info else "-",
            "Min": f"{info.get('min', '-'):.2f}" if "min" in info else "-",
            "Max": f"{info.get('max', '-'):.2f}" if "max" in info else "-",
            "Outliers": f"{info.get('outliers_iqr', '-')}" if "outliers_iqr" in info else "-",
        })

    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Individual column deep-dive ──
    st.markdown('<div class="insightly-subheader">Column Deep Dive</div>', unsafe_allow_html=True)
    selected_col = st.selectbox("Select a column to inspect:", col_names, key="profile_col")

    if selected_col:
        info = columns[selected_col]
        c1, c2 = st.columns([1, 1.5])

        with c1:
            card_content = f"""
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Type</td>
                        <td style="padding: 4px 8px;">{info.get('inferred_type', '-')}</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Dtype</td>
                        <td style="padding: 4px 8px;"><code>{info.get('dtype', '-')}</code></td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Count</td>
                        <td style="padding: 4px 8px;">{info.get('count', 0):,}</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Missing</td>
                        <td style="padding: 4px 8px;">{info.get('nulls', 0):,} ({info.get('null_pct', 0):.1f}%)</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Unique</td>
                        <td style="padding: 4px 8px;">{info.get('unique', 0):,} ({info.get('unique_pct', 0):.1f}%)</td></tr>
            """
            if "mean" in info:
                card_content += f"""
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Mean ± Std</td>
                        <td style="padding: 4px 8px;">{info.get('mean', '-'):.4f} ± {info.get('std', '-'):.4f}</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Median</td>
                        <td style="padding: 4px 8px;">{info.get('median', '-'):.4f}</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Range</td>
                        <td style="padding: 4px 8px;">{info.get('min', '-'):.2f} – {info.get('max', '-'):.2f}</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">IQR</td>
                        <td style="padding: 4px 8px;">{info.get('iqr', '-'):.4f}</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Skewness</td>
                        <td style="padding: 4px 8px;">{info.get('skew', '-'):.4f}</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">IQR Outliers</td>
                        <td style="padding: 4px 8px;">{info.get('outliers_iqr', 0):,}</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Zeros</td>
                        <td style="padding: 4px 8px;">{info.get('zeros', 0):,}</td></tr>
                """
            if "top_value" in info:
                card_content += f"""
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Top Value</td>
                        <td style="padding: 4px 8px;">'{info.get('top_value', '-')}' ({info.get('top_pct', 0):.1f}%)</td></tr>
                """
            if "min_date" in info:
                card_content += f"""
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Date Range</td>
                        <td style="padding: 4px 8px;">{info.get('min_date', '-')} to {info.get('max_date', '-')}</td></tr>
                    <tr><td style="padding: 4px 8px; font-weight: 600;">Span</td>
                        <td style="padding: 4px 8px;">{info.get('range_days', '-')} days</td></tr>
                """
            card_content += "</table>"
            card(card_content)

        with c2:
            fig = create_chart(df, selected_col)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: CLEAN
# ═══════════════════════════════════════════════════════════════

with t3:
    st.markdown('<div class="insightly-header">🧹 Data Cleaning</div>', unsafe_allow_html=True)

    # Detect issues
    issues = detect_issues(df)

    total_issues = sum(len(v) for v in issues.values())

    if total_issues == 0:
        st.success("✅ No issues detected — dataset looks clean!")
    else:
        st.warning(f"⚠️ {total_issues} issue(s) detected")

        if issues["missing"]:
            with st.expander(f"📋 Missing Values ({len(issues['missing'])})"):
                st.dataframe(pd.DataFrame(issues["missing"]), use_container_width=True, hide_index=True)

        if issues["duplicates"]:
            with st.expander(f"📋 Duplicate Rows ({issues['duplicates'][0]['count']})"):
                st.dataframe(pd.DataFrame(issues["duplicates"]), use_container_width=True, hide_index=True)

        if issues["types"]:
            with st.expander(f"📋 Type Issues ({len(issues['types'])})"):
                st.dataframe(pd.DataFrame(issues["types"]), use_container_width=True, hide_index=True)

        if issues["outliers"]:
            with st.expander(f"📋 Outliers ({len(issues['outliers'])})"):
                st.dataframe(pd.DataFrame(issues["outliers"]), use_container_width=True, hide_index=True)

    st.divider()

    # ── Cleaning Options ──
    st.markdown('<div class="insightly-subheader">Cleaning Options</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        do_dedup = st.checkbox("Drop duplicate rows", value=df.duplicated().sum() > 0)
    with c2:
        fill_strat = st.selectbox("Handle missing values", fill_missing_options(),
                                  index=0, help="mean/median: fill numeric only | mode: fill all | drop: remove rows")
    with c3:
        do_fix_types = st.checkbox("Auto-fix data types", value=True)

    do_outliers = st.checkbox("Remove outliers (IQR method)", value=False)

    if st.button("🧼 Clean Data", type="primary", use_container_width=True):
        with st.spinner("Cleaning..."):
            cleaned, log = clean_dataframe(
                df,
                drop_duplicates=do_dedup,
                fill_missing=fill_strat,
                fix_types=do_fix_types,
                remove_outliers=do_outliers,
            )
            st.session_state.cleaned_df = cleaned
            st.session_state.cleaning_log = log

    # ── Cleaning Results ──
    if st.session_state.cleaned_df is not None:
        st.divider()
        st.markdown(f'<div class="insightly-header">✅ Cleaned Data</div>', unsafe_allow_html=True)

        cleaned = st.session_state.cleaned_df
        orig_rows, orig_cols = df.shape
        new_rows, new_cols = cleaned.shape

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Original Rows", f"{orig_rows:,}", f"{orig_rows - new_rows} removed" if orig_rows != new_rows else "")
        with c2: kpi("Cleaned Rows", f"{new_rows:,}")
        with c3: kpi("Columns", f"{new_cols}")
        with c4: kpi("Changes", f"{len(st.session_state.cleaning_log)}")

        if st.session_state.cleaning_log:
            st.markdown('<div class="insightly-subheader">Change Log</div>', unsafe_allow_html=True)
            for entry in st.session_state.cleaning_log:
                st.markdown(f"- ✅ {entry}")

        st.markdown('<div class="insightly-subheader">Preview</div>', unsafe_allow_html=True)
        st.dataframe(cleaned.head(50), use_container_width=True, hide_index=True)

        if st.button("📥 Use Cleaned Data for Analysis", use_container_width=True):
            st.session_state.df = cleaned
            st.session_state.cleaned_df = None
            st.success("Switched to cleaned data!")
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# TAB 4: INSIGHTS
# ═══════════════════════════════════════════════════════════════

with t4:
    st.markdown('<div class="insightly-header">💡 Insights</div>', unsafe_allow_html=True)

    # Auto-generated insight summary
    with st.spinner("Generating insights..."):
        insights = generate_insights_summary(df)

    st.markdown('<div class="insightly-subheader">📝 Key Findings</div>', unsafe_allow_html=True)
    for ins in insights:
        st.markdown(f"- {ins}")

    st.divider()

    # ── Correlations ──
    st.markdown('<div class="insightly-subheader">📊 Correlation Analysis</div>', unsafe_allow_html=True)

    corr_result = analyze_correlations(df)

    if corr_result["matrix"] is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            fig = create_correlation_heatmap(corr_result["matrix"])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if corr_result["top"]:
                st.markdown("**Top Correlations**")
                top_df = pd.DataFrame(corr_result["top"])
                top_df["strength_label"] = top_df["strength"].apply(
                    lambda x: "🟢 Strong" if x > 0.7 else "🟡 Moderate" if x > 0.4 else "🔵 Weak"
                )
                top_df = top_df.rename(columns={
                    "col1": "Column 1", "col2": "Column 2",
                    "correlation": "r", "strength_label": "Strength",
                })
                st.dataframe(top_df[["Column 1", "Column 2", "r", "Strength"]], use_container_width=True, hide_index=True)
    else:
        st.info(corr_result.get("message", "Not enough numeric columns for correlation analysis."))

    st.divider()

    # ── Distribution Anomalies ──
    st.markdown('<div class="insightly-subheader">⚠️ Distribution Anomalies</div>', unsafe_allow_html=True)
    anomalies = detect_distribution_anomalies(df)

    if anomalies:
        st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No unusual distributions detected.")

    st.divider()

    # ── Outliers ──
    st.markdown('<div class="insightly-subheader">📈 Outlier Detection (Z-score)</div>', unsafe_allow_html=True)
    outliers = detect_outliers_zscore(df)

    if outliers:
        rows = []
        for col, info in outliers.items():
            rows.append({"Column": col, "Outliers": info["count"], "%": info["pct"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No significant outliers detected (|z| > 3).")


# ═══════════════════════════════════════════════════════════════
# TAB 5: VISUALIZE
# ═══════════════════════════════════════════════════════════════

with t5:
    st.markdown('<div class="insightly-header">📈 Visualizations</div>', unsafe_allow_html=True)

    # ── Auto Gallery ──
    st.markdown('<div class="insightly-subheader">🎨 Auto-Generated Gallery</div>', unsafe_allow_html=True)

    with st.spinner("Generating charts..."):
        gallery = auto_gallery(df, max_charts=8)

    if gallery:
        for i in range(0, len(gallery), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(gallery):
                    col_name, chart_type, fig = gallery[i + j]
                    with cols[j]:
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to auto-generate charts.")

    st.divider()

    # ── Custom Chart Builder ──
    st.markdown('<div class="insightly-subheader">🔧 Custom Chart</div>', unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns]
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    all_cols = df.columns.tolist()

    col1, col2, col3 = st.columns(3)

    with col1:
        x_col = st.selectbox("X-axis / Main column", all_cols, key="viz_x")
    with col2:
        y_col = st.selectbox("Y-axis / Secondary (optional)", ["None"] + all_cols, key="viz_y")
    with col3:
        color_col = st.selectbox("Color by (optional)", ["None"] + all_cols, key="viz_color")

    # Auto-detect chart type suggestion
    suggested = suggest_chart_type(
        df[x_col],
        df[y_col] if y_col != "None" else None,
    )

    # Chart type selector
    if y_col != "None":
        chart_options = ["scatter", "line", "bar", "box", "violin"]
    else:
        chart_options = ["histogram", "bar", "pie", "box", "violin", "kde"]

    chart_type = st.selectbox(
        "Chart type",
        chart_options,
        index=chart_options.index(suggested) if suggested in chart_options else 0,
        key="viz_type",
    )

    if st.button("Generate Chart", type="primary", use_container_width=False):
        with st.spinner("Rendering..."):
            fig = create_chart(
                df,
                x_col,
                secondary=y_col if y_col != "None" else None,
                chart_type=chart_type,
                color_col=color_col if color_col != "None" else None,
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 6: REPORT
# ═══════════════════════════════════════════════════════════════

with t6:
    st.markdown('<div class="insightly-header">📋 Narrative Report</div>', unsafe_allow_html=True)

    with st.spinner("Generating report..."):
        profile = profile_dataframe(df)
        report_text = generate_report(df, profile)

    st.markdown(report_text, unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.code(report_text, language="markdown")
            st.info("Select all and copy (Ctrl+A, Ctrl+C)")
    with col2:
        report_body = report_text.replace('\n', '<br>')
        report_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Insightly Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 800px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; color: #1a1a2e; }}
h1 {{ color: #2563eb; }} h2 {{ color: #1a1a2e; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.3rem; }}
code {{ background: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.9em; }}
</style></head><body>
{report_body}
</body></html>"""
        st.download_button(
            "📥 Download HTML Report",
            data=report_html,
            file_name="insightly_report.html",
            mime="text/html",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 7: EXPORT
# ═══════════════════════════════════════════════════════════════

with t7:
    st.markdown('<div class="insightly-header">📥 Export Data</div>', unsafe_allow_html=True)

    use_cleaned = st.checkbox("Export cleaned data", value=st.session_state.cleaned_df is not None)
    export_df = st.session_state.cleaned_df if (use_cleaned and st.session_state.cleaned_df is not None) else df

    st.markdown(f"Exporting: **{export_df.shape[0]:,} rows × {export_df.shape[1]} columns**")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name="insightly_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        try:
            excel_buffer = pd.ExcelWriter("insightly_export.xlsx", engine="openpyxl")
            export_df.to_excel(excel_buffer, index=False, sheet_name="Data")
            excel_buffer.close()
            with open("insightly_export.xlsx", "rb") as f:
                excel_data = f.read()
            os.remove("insightly_export.xlsx")
            st.download_button(
                "📥 Download Excel",
                data=excel_data,
                file_name="insightly_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Excel export unavailable: {e}")

    with col3:
        json_data = export_df.to_json(orient="records", date_format="iso").encode("utf-8")
        st.download_button(
            "📥 Download JSON",
            data=json_data,
            file_name="insightly_export.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    st.markdown('<div class="insightly-subheader">Data Summary</div>', unsafe_allow_html=True)
    st.dataframe(export_df.describe(include="all").T, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 8: ASK AI
# ═══════════════════════════════════════════════════════════════

with t8:
    st.markdown('<div class="insightly-header">💬 Ask AI About Your Data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insightly-subheader">Ask questions in plain English and get instant answers with visualizations</div>',
        unsafe_allow_html=True,
    )

    # ── LLM function builder ──
    def build_llm_func(api_key: str):
        """Build an LLM function if an API key is available."""
        if not api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            def llm_answer(question: str) -> str:
                # Build a context prompt with dataframe info
                context = f"""You are a data analysis assistant. The user has a dataset loaded with:
- {len(df):,} rows and {len(df.columns)} columns
- Columns: {', '.join(f'{c} ({df[c].dtype})' for c in df.columns)}

Sample rows (first 3):
{df.head(3).to_string()}

Summary statistics:
{df.describe(include='all').to_string()}

User question: {question}

Provide a concise, helpful answer in plain English. Use markdown formatting. If the question involves specific numbers or calculations, include them."""

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a data analyst assistant. Answer questions about the user's dataset concisely and accurately."},
                        {"role": "user", "content": context},
                    ],
                    max_tokens=500,
                    temperature=0.3,
                )
                return resp.choices[0].message.content

            return llm_answer
        except ImportError:
            st.warning("OpenAI package not installed. Run: pip install openai")
            return None
        except Exception as e:
            st.warning(f"Could not initialize LLM: {e}")
            return None

    # ── Display Chat History ──
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.qa_messages:
            role = msg["role"]
            content = msg["content"]
            fig = msg.get("figure")

            if role == "user":
                st.markdown(
                    f"""<div style="display: flex; justify-content: flex-end; margin-bottom: 0.5rem;">
                        <div style="background: var(--insightly-accent); color: #fff; padding: 0.6rem 1rem; border-radius: 16px 16px 4px 16px; max-width: 80%;">
                            {content}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div style="display: flex; justify-content: flex-start; margin-bottom: 0.5rem;">
                        <div style="background: var(--insightly-card); color: var(--insightly-text); padding: 0.6rem 1rem; border-radius: 16px 16px 16px 4px; max-width: 85%;">
                            {content}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # ── Input Area ──
    st.divider()

    # Question suggestions
    if len(st.session_state.qa_messages) == 0:
        st.markdown('<div class="insightly-subheader">💡 Try asking:</div>', unsafe_allow_html=True)
        suggestions = get_suggestions(df)
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, use_container_width=True, key=f"qs_{i}"):
                    # Process suggestion immediately
                    st.session_state.qa_messages.append({"role": "user", "content": suggestion})
                    llm_func = build_llm_func(st.session_state.openai_api_key)
                    result = answer_query(df, suggestion, llm_func=llm_func)
                    text = result.get("text", "I couldn't find an answer to that question.")
                    fig = result.get("figure")
                    st.session_state.qa_messages.append({
                        "role": "assistant",
                        "content": text,
                        "figure": fig,
                    })
                    st.rerun()

    # Text input
    col1, col2 = st.columns([6, 1])
    with col1:
        user_query = st.text_input(
            "Ask about this data...",
            placeholder="e.g. Show me top 10 rows, or Average of Amount...",
            label_visibility="collapsed",
            key="qa_input",
        )
    with col2:
        send = st.button("🚀 Ask", type="primary", use_container_width=True)

    if send and user_query:
        # Add user message
        st.session_state.qa_messages.append({"role": "user", "content": user_query})

        # Clear input by rerunning after we process
        with st.spinner("Thinking..."):
            llm_func = build_llm_func(st.session_state.openai_api_key)
            result = answer_query(df, user_query, llm_func=llm_func)

        text = result.get("text", "I couldn't find an answer to that question.")
        fig = result.get("figure")

        st.session_state.qa_messages.append({
            "role": "assistant",
            "content": text,
            "figure": fig,
        })

        st.rerun()

    if len(st.session_state.qa_messages) > 0:
        if st.button("🗑️ Clear conversation", use_container_width=False):
            st.session_state.qa_messages = []
            st.rerun()

    # ── Keyboard hint ──
    if len(st.session_state.qa_messages) == 0:
        st.markdown(
            """<div style="text-align: center; padding: 2rem; color: var(--insightly-muted); font-size: 0.9rem;">
                Type a question above and press "Ask" to get started.<br>
                Works without an API key! For complex questions, add your OpenAI key in the sidebar 🤖
            </div>""",
            unsafe_allow_html=True,
        )
