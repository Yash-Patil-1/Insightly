# 📖 Usage Guide

## Loading Data

Insightly supports **7 input methods** for loading data:

### Sample Data

Three pre-loaded datasets demonstrate different formats:

| Dataset | Format | Rows | Columns | Description |
|---------|--------|------|---------|-------------|
| **Personal Finance** | CSV | 48 | 7 | Bank transactions with date, category, amount, payment method, merchant, balance |
| **Sales Report** | Excel | 48 | 7 | Same finance data in Excel — demonstrates multi-format handling |
| **Survey Results** | JSON | 50 | 6 | Survey with rating, age group, gender, satisfaction, feedback |

Select from the sidebar dropdown to load instantly.

### File Upload

Supported formats and their extensions:

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Auto-detects encoding (utf-8, latin1, cp1252) and delimiter (comma, semicolon) |
| TSV | `.tsv` | Tab-separated values |
| Excel | `.xlsx`, `.xls` | Multi-sheet support — select which sheet to load |
| JSON | `.json` | Accepts array-of-objects or single object; auto-normalizes nested data |
| Parquet | `.parquet` | Columnar storage, efficient for large datasets |
| Feather | `.feather` | Fast binary format, great for round-trip performance |

Drag & drop files onto the upload area, or click to browse.

### Clipboard Paste

1. Copy tab-separated data from Excel, Google Sheets, or any spreadsheet
2. Click the **"📋 Paste data"** expander in the sidebar
3. Paste into the text area
4. Data loads automatically

## Feature Walkthrough

### 📂 Overview Tab

The Overview tab gives you a high-level snapshot of your dataset:

- **KPI Cards**: Rows, Columns, Missing cells (with %), Duplicates (with %), Memory size
- **Data Types Pie**: Visual breakdown of column types (numeric, categorical, datetime, boolean)
- **Column Preview Table**: All columns with type, missing %, and unique count
- **Missing Value Heatmap**: White = present, dark = missing — quickly spot sparsity patterns
- **Data Preview**: First 100 rows in an interactive table

### 🔬 Profile Tab

Detailed statistics for every column in a sortable table:

- **Numeric columns**: Mean, Median, Std, IQR, Min, Max, Skew, Kurtosis, Zeros, Negatives, IQR Outliers
- **Categorical columns**: Top value with frequency %, Entropy, Empty strings, Top 5 value counts
- **Datetime columns**: Date range (min/max), span in days

Click a column in the dropdown to see a **deep-dive** card with all stats plus an auto-generated chart.

### 🧹 Clean Tab

**Issue Detection** — Automatically scans for:
- Missing values (per column with count and %)
- Duplicate rows
- Type issues (object columns with mixed numeric/text)
- Outliers (IQR method, per numeric column with bounds)

**Cleaning Options**:
| Option | Description |
|--------|-------------|
| Drop duplicate rows | Removes exact duplicate rows |
| Handle missing values | `none` (skip), `drop` (remove rows), `mean`/`median` (fill numeric), `mode` (fill all) |
| Auto-fix types | Convert object → numeric/datetime when ≥80% parse correctly |
| Remove outliers | IQR-based, removes rows with values beyond 1.5×IQR from any numeric column |

After cleaning, you can **swap** the cleaned data in place for further analysis.

### 💡 Insights Tab

Auto-generated analysis without any configuration:

- **Key Findings**: Bullet-point summary of size, missing, duplicates, cardinality, top correlation
- **Correlation Matrix**: Interactive heatmap with Pearson r values + p-values
- **Top Correlations Table**: Sorted by absolute strength with strength labels (Weak/Moderate/Strong)
- **Distribution Anomalies**: Detects constant columns, high skew, high kurtosis, zero-inflation, dominant categories
- **Outlier Detection**: Z-score method (|z| > 3) with per-column counts and percentages

### 📈 Visualize Tab

**Auto-Generated Gallery**: Up to 8 charts automatically selected based on your data:
- Histograms for numeric columns
- Pie charts for low-cardinality categorical columns
- Scatter plots for top numeric pairs
- Box plots for categorical vs numeric
- Line charts for time series

**Custom Chart Builder**:
1. Select **X-axis** column (required)
2. Select **Y-axis** column (optional — for two-variable charts)
3. Select **Color by** column (optional — for grouping)
4. Chart type is **auto-suggested** based on column data types
5. Optionally override with: histogram, bar, pie, box, scatter, line, violin, kde

### 📋 Report Tab

A **plain-English narrative report** covering:
1. **Dataset Overview** — Rows, columns, memory, missing/duplicate summary
2. **Column-by-Column Analysis** — Every column with type, stats, quality flags
3. **Data Quality Assessment** — Flags issues across columns
4. **Recommendations** — Actionable suggestions (imputation, transformation, feature engineering)

Export options:
- **Copy to Clipboard** — Shows the raw Markdown for manual copy
- **Download HTML Report** — Styled HTML file for sharing

### 📥 Export Tab

Download your data (original or cleaned) in three formats:

| Format | File | Use Case |
|--------|------|----------|
| **CSV** | `insightly_export.csv` | Universal, opens in any spreadsheet |
| **Excel** | `insightly_export.xlsx` | Formatted with sheet name, for Excel users |
| **JSON** | `insightly_export.json` | API integration, web applications |

Check **"Export cleaned data"** to export the cleaned version (if you've cleaned it).

### 💬 Ask AI Tab

Ask questions about your data in **plain English**. Works two ways:

**Pattern Engine** (always available, no setup needed):
- Handles 40+ question types instantly
- Examples: "Show top 10 rows", "Average of Amount", "Filter where Amount > 500", "Correlation between Amount and Balance", "Missing values", "Plot Amount by Category"

**LLM Engine** (optional, requires OpenAI API key):
- For complex questions the pattern engine can't handle
- Enter your OpenAI API key in the sidebar → **AI Settings**
- Uses GPT-4o-mini with full dataset context

Click the suggested questions to get started, or type your own.

## Theme Settings

Toggle between **Dark**, **Light**, and **System** themes in the sidebar. The System theme respects your OS-level preference (`prefers-color-scheme`). The toggle updates instantly with smooth CSS transitions.

## Keyboard Shortcuts

- **Upload area**: Click or drag & drop files
- **Custom Chart**: Press Enter after selecting columns to trigger generation
- **Ask AI**: Type question, press Enter or click "Ask"
- **Tab navigation**: Click tabs to switch views (no keyboard shortcuts yet)

## Tips & Best Practices

1. **Start with sample data** if you're new — explore all 8 tabs
2. **Use the Overview tab** first to understand your data's shape and quality
3. **Clean before analyzing** — fix missing values and outliers before generating insights
4. **Swap cleaned data** into place to run the full pipeline on clean data
5. **Use the Report tab** to generate a shareable summary
6. **For large datasets** (100K+ rows), be patient with auto-gallery and correlation matrix generation
7. **Add your OpenAI API key** for smarter answers to complex questions in the Ask AI tab
8. **Export cleaned data** before closing to save your work

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Unsupported format" | Check file extension. Supported: `.csv`, `.tsv`, `.xlsx`, `.xls`, `.json`, `.parquet`, `.feather` |
| Encoding errors with CSV | Insightly tries 4 encodings automatically. If it fails, try converting to UTF-8 first |
| Large file is slow | Switch to Parquet format for faster loading. Reduce displayed rows in the preview |
| Excel multi-sheet | Select the correct sheet from the dropdown that appears |
| Chart rendering fails | Try a different chart type. Some column combinations don't produce meaningful charts |
| Q&A doesn't understand | Try rephrasing. Use simple, direct language. Check the suggested questions for examples |
| Missing OpenAI key | The Q&A engine works without it! Only advanced questions need the LLM fallback |
