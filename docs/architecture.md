# 🏗️ Architecture

This document describes the internal architecture of Insightly — how the components interact, the data flow, and key design decisions.

## Overview

Insightly follows a **modular pipeline architecture** with a single entry point (the Streamlit app) orchestrating independent analysis modules:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Layer                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │   CSV    │  │  Excel   │  │   JSON   │  │  Parquet / TSV /   │   │
│  │          │  │ (.xlsx)  │  │          │  │  Feather / Clipbrd │   │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Loader Layer                                    │
│  src/loader.py                                                        │
│  ├── load_file()     ── Multi-format loader with encoding detection   │
│  ├── load_clipboard() ── Parse tab-separated clipboard text           │
│  └── get_file_info()  ── Returns rows, cols, types, missing, size     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Analysis Pipeline                                   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────┐             │
│  │  profiler.py       Per-column stats, type inference  │             │
│  │                    numeric/categorical/datetime       │             │
│  └──────────┬──────────────────────────────────────────┘             │
│             │                                                         │
│  ┌──────────▼──────────────────────────────────────────┐             │
│  │  analyzer.py       Correlations (Pearson r + p-val) │             │
│  │                    Z-score outliers, distribution    │             │
│  │                    anomalies, time trends, insights  │             │
│  └──────────┬──────────────────────────────────────────┘             │
│             │                                                         │
│  ┌──────────▼──────────────────────────────────────────┐             │
│  │  cleaner.py        Drop dupes, fill missing, fix     │             │
│  │                    types, remove outliers (IQR)      │             │
│  └──────────┬──────────────────────────────────────────┘             │
│             │                                                         │
│  ┌──────────▼──────────────────────────────────────────┐             │
│  │  visualizer.py     Smart chart picker, auto-gallery │             │
│  │                    Correlation heatmap, missing map  │             │
│  └──────────┬──────────────────────────────────────────┘             │
│             │                                                         │
│  ┌──────────▼──────────────────────────────────────────┐             │
│  │  reporter.py       Natural-language narrative report│             │
│  │                    with recommendations               │             │
│  └─────────────────────────────────────────────────────┘             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Q&A Engine                                       │
│  src/qa.py                                                            │
│  ├── Pattern-based engine (40+ regex patterns)                       │
│  │   ├── Row display, statistics, filtering, grouping                │
│  │   ├── Correlations, outliers, missing values, trends              │
│  │   ├── Charts, data quality, comparisons                           │
│  │   └── Fuzzy column name resolution                                │
│  ├── Optional LLM fallback (OpenAI) for complex questions            │
│  └── Context-aware question suggestions                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                                  │
│  Streamlit Dashboard (app.py) — 8 Tabs                               │
│                                                                       │
│  ├── 📂 Overview     ── KPI cards, data types, missing heatmap       │
│  ├── 🔬 Profile     ── Full stat table, column deep-dive            │
│  ├── 🧹 Clean       ── Issue detection, cleaning, change log        │
│  ├── 💡 Insights    ── Key findings, correlations, anomalies         │
│  ├── 📈 Visualize   ── Auto gallery, custom chart builder            │
│  ├── 📋 Report      ── Narrative report with export                  │
│  ├── 📥 Export      ── Download CSV/Excel/JSON                      │
│  └── 💬 Ask AI      ── Chat interface with pattern/LLM Q&A          │
│                                                                       │
│  ┌────────────────────────────────────────────────┐                  │
│  │  src/theme.py    Dark/Light/System theme toggle │                  │
│  │                  Custom CSS variables           │                  │
│  └────────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Details

### `src/loader.py` — Multi-Format Data Loader

Handles **7 input formats** with automatic detection:

| Format | Extension | Loader |
|--------|-----------|--------|
| CSV | `.csv` | Encoding detection (utf-8, latin1, cp1252), delimiter sniffing (comma/semicolon) |
| TSV | `.tsv` | Tab-separated, low-memory mode |
| Excel | `.xlsx`, `.xls` | Multi-sheet selector, openpyxl engine |
| JSON | `.json` | Array-of-objects or dict, automatic normalization |
| Parquet | `.parquet` | Columnar format, efficient for large data |
| Feather | `.feather` | Fast binary format, cross-language |
| Clipboard | — | Tab-separated text via `pd.read_csv` with StringIO |

### `src/profiler.py` — Auto-Profiler

Per-column profiling with automatic type inference:

- **Numeric columns**: count, mean, median, std, var, skew, kurtosis, IQR, min, max, zeros, negatives, IQR outlier count
- **Categorical columns**: count, top value, frequency, entropy, empty strings, top 5 value counts
- **Datetime columns**: min/max dates, date range in days, most frequent date
- **Type inference**: Boolean, Datetime, Numeric, Categorical (Low/High Cardinality), Text / High Cardinality

### `src/cleaner.py` — Data Cleaner

Four cleaning operations, all optional and configurable:

1. **Drop duplicates** — Removes exact duplicate rows
2. **Fill missing** — Options: none, drop, mean (numeric), median (numeric), mode (all types)
3. **Fix types** — Auto-converts object columns to numeric or datetime when ≥80% of values parse correctly
4. **Remove outliers** — IQR-based (1.5× rule), applies to all numeric columns simultaneously

Returns `(cleaned_df, change_log)` — the log is a list of human-readable strings describing each action.

### `src/analyzer.py` — Analysis Engine

- **Correlations**: Pearson correlation matrix with top-10 pairs ranked by absolute strength; p-values computed via scipy
- **Outlier detection**: Z-score method with configurable threshold (default |z| > 3)
- **Distribution anomalies**: Detects constant columns, high skew (>2), high kurtosis (>5), zero-inflation (>50%), dominant categories (>95%)
- **Time trends**: Linear regression on any date+value column pair with direction/strength/significance
- **Insights summary**: Auto-generated plain-English bullet points covering size, missing data, duplicates, cardinality, top correlation

### `src/visualizer.py` — Smart Visualizer

- **`suggest_chart_type()`** — Heuristic engine that picks the best chart based on data types:
  - Single numeric → histogram (or bar if ≤5 unique values)
  - Single categorical → pie (≤10), bar (≤30)
  - Two numeric → scatter with trendline
  - Categorical + numeric → box plot
  - Datetime + numeric → line chart
- **`create_chart()`** — Renders any chart type with consistent styling
- **`create_correlation_heatmap()`** — Color-coded Pearson matrix
- **`create_missing_heatmap()`** — Visual null value distribution
- **`auto_gallery()`** — Picks the top 8 most informative charts from the dataset

### `src/reporter.py` — Narrative Report Generator

Generates a structured Markdown report with sections:

1. **Dataset Overview** — Rows, columns, memory, missing/duplicate summary
2. **Column-by-Column Analysis** — Comprehensive details for every column
3. **Data Quality Assessment** — Flags high-missing, outlier, high-cardinality columns
4. **Recommendations** — Data-driven suggestions for improvement (imputation, winsorization, log transforms, etc.)

### `src/qa.py` — Natural Language Q&A Engine

A dual-engine question answering system:

**Pattern Engine** (always available):
- 40+ regex patterns covering common question types
- Fuzzy column name resolution (exact, case-insensitive, partial, word overlap)
- Handles: show rows, statistics, filtering, grouping, correlations, outliers, missing values, time trends, comparisons, charts, data quality
- Returns text answers (Markdown) and optionally Plotly figures

**LLM Engine** (optional, requires OpenAI API key):
- Uses GPT-4o-mini for complex questions the pattern engine can't handle
- Provides full dataset context (schema, sample rows, summary stats) in the prompt
- Configurable in the sidebar

## Key Design Decisions

### 1. Stateless Analysis

Each analysis module is a pure function of the DataFrame — no shared state, no side effects. This means:
- The **Clean** tab can produce a cleaned DataFrame that replaces the original
- Results can be cached with Streamlit's `st.session_state`
- Tests are simple: pass a DataFrame, check the output

### 2. Pattern-First Q&A with LLM Fallback

The QA engine uses regex patterns for 95% of common questions — this means it works **instantly** without any API key, network calls, or external dependencies. For complex questions, an optional OpenAI integration provides GPT-4o-mini answers.

### 3. Progressive Disclosure

The UI shows simple overviews first (KPI cards, column preview), then progressively reveals more detail (deep-dive stats, custom charts, full report). This prevents overwhelming new users while giving analysts the depth they need.

### 4. Theme as CSS Variables

Rather than complex theme logic, Insightly uses CSS custom properties (`--insightly-*`) that are toggled via a single `<style>` injection. Three modes (Dark, Light, System) with smooth transitions.

## Data Flow Example

```
1. User loads "Personal Finance" sample
2.   ├── loader.load_file() returns DataFrame (48 rows, 7 cols)
3.   ├── profiler.profile_dataframe() computes per-column stats
4.   ├── Overview tab renders KPIs, data types, preview
5.   │
6. User clicks "Clean" tab
7.   ├── cleaner.detect_issues() scans for problems
8.   ├── User selects "fill missing with median"
9.   ├── cleaner.clean_dataframe() returns cleaned DataFrame + log
10.  ├── User clicks "Use Cleaned Data"
11.  │   └── st.session_state.df = cleaned (replaces original)
12.  │
13. User clicks "Insights" tab
14.  ├── analyzer.analyze_correlations() computes matrix + p-values
15.  ├── analyzer.detect_distribution_anomalies() flags issues
16.  ├── analyzer.detect_outliers_zscore() finds outliers
17.  └── analyzer.generate_insights_summary() writes bullet points
```

## Extending Insightly

### Adding a new input format:
1. Add the format to `SUPPORTED_FORMATS` in `src/loader.py`
2. Add a `read_*` call in `load_file()`
3. Add test cases in `tests/test_loader.py`

### Adding a new chart type:
1. Add the chart type to `create_chart()` in `src/visualizer.py`
2. Add it to `suggest_chart_type()` for auto-detection
3. Add to the custom chart builder selectors in `app.py`

### Adding a new Q&A pattern:
1. Add a regex pattern + handler name to `QUESTION_PATTERNS` in `src/qa.py`
2. Create the handler function following the existing pattern
3. Register it in the `_HANDLERS` dict
