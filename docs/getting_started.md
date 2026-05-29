# 🚀 Getting Started with Insightly

This guide walks you through your first Insightly session — from installation to exploring your first dataset.

## Prerequisites

- **Python 3.9+** installed on your system
- **pip** package manager
- A structured data file (CSV, Excel, JSON, Parquet, TSV, Feather) or clipboard data

## Installation

```bash
# Clone the repository
git clone https://github.com/Yash-Patil-1/Insightly.git
cd Insightly

# Install dependencies
pip install -r requirements.txt

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Your First Run

### 1. Launch Insightly

```bash
streamlit run app.py
```

Your browser will open at **http://localhost:8501** with the Insightly welcome screen.

### 2. Load Sample Data

Insightly ships with 3 sample datasets so you can explore immediately:

1. In the sidebar, click the **"Load sample data"** dropdown
2. Select one of:
   - **Personal Finance** (CSV) — 50 bank transactions with categories, amounts, and payment methods
   - **Sales Report** (Excel) — Same finance data in Excel format
   - **Survey Results** (JSON) — 50 survey responses with ratings, demographics, and feedback

The dataset loads instantly and you'll see KPI cards, a data preview, and missing value analysis.

### 3. Explore the Tabs

| Tab | What You'll See |
|-----|-----------------|
| **📂 Overview** | 5 KPI cards, data type pie chart, column preview table, missing value heatmap, data preview |
| **🔬 Profile** | Full statistics for every column, column deep-dive selector with detailed stats + auto-chart |
| **🧹 Clean** | Issue detection (missing, duplicates, types, outliers), cleaning options, change log, cleaned data preview |
| **💡 Insights** | Auto-generated key findings, correlation matrix with p-values, distribution anomalies, z-score outlier table |
| **📈 Visualize** | Auto-generated chart gallery (up to 8 charts), custom chart builder with auto-suggested chart type |
| **📋 Report** | Plain-English narrative report with copy-to-clipboard and HTML download |
| **📥 Export** | Download cleaned data as CSV, Excel, or JSON |
| **💬 Ask AI** | Ask questions in plain English — works with or without an OpenAI API key |

### 4. Load Your Own Data

Instead of sample data, you can:

- **Drag & drop** a file onto the upload area in the sidebar
- **Browse** your computer for CSV, Excel, JSON, Parquet, TSV, or Feather files
- **Paste tab-separated data** from clipboard using the text area in the sidebar

### 5. Generate a Report

Go to the **📋 Report** tab and click **"Copy to Clipboard"** or **"Download HTML Report"** to save your analysis.

---

## Next Steps

- [Usage Guide](usage.md) — Detailed feature walkthrough and tips
- [Architecture](architecture.md) — Deep dive into the codebase design
- [Development](development.md) — Guide for contributors and extending Insightly
- [README.md](../README.md) — Project overview and feature summary
