# 🔍 Insightly — Swiss Army Knife for Data Analysis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/github/actions/workflow/status/Yash-Patil-1/Insightly/insightly-ci.yml?branch=main&label=CI&logo=github" alt="CI">
</p>

**Drop any structured data file and get instant insights, visualizations, and a plain-English report.**

Insightly is a universal data analysis tool that handles **CSV, Excel, JSON, Parquet, Feather, TSV, and clipboard paste** — no coding, no setup, just drag and watch.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📂 Multi-Format** | CSV, Excel (.xlsx/.xls), JSON, Parquet, Feather, TSV, clipboard paste |
| **📊 Auto-Profile** | Per-column stats: mean, median, std, skew, IQR, outliers, entropy, cardinality |
| **🧹 One-Click Clean** | Drop duplicates, fill missing (mean/median/mode), auto-fix types, remove outliers |
| **💡 Smart Insights** | Auto-generated key findings, top correlations, distribution anomalies, z-score outliers |
| **📈 Auto Visualizer** | Smart chart picker chooses the best viz for each column — histogram, bar, pie, box, scatter, heatmap |
| **📋 Narrative Report** | Natural-language summary of the entire dataset with recommendations |
| **🎨 Theme Toggle** | Dark / Light / System theme — instant switch |
| **📥 Export** | Download cleaned data as CSV, Excel, or JSON |

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🖥️ How to Use

1. **Load data** — Pick a sample dataset from the sidebar, upload a file, or paste TSV from clipboard
2. **Explore** — Browse the Overview, Profile, and Insights tabs
3. **Clean** — Fix missing values, duplicates, and outliers with one click
4. **Visualize** — Auto-generated gallery or custom chart builder
5. **Report** — Generate a plain-English narrative report
6. **Export** — Download cleaned data in your preferred format

---

## 📁 Sample Data

Insightly ships with 3 sample datasets to demonstrate multi-format handling:

| Dataset | Format | Description |
|---------|--------|-------------|
| **Personal Finance** | CSV | 50 bank transactions with categories, payment methods, and balances |
| **Sales Report** | Excel | Same finance data as Excel to demonstrate multi-sheet handling |
| **Survey Results** | JSON | 50 survey responses with ratings, demographics, and feedback text |

---

## 🛠️ Tech Stack

- **Python** (pandas, numpy, scipy)
- **Streamlit** — Interactive UI
- **Plotly** — Charts and visualizations
- **openpyxl** — Excel support

---

## 📬 Contact

**Yash Patil**

- 📧 [yashpatil7714@gmail.com](mailto:yashpatil7714@gmail.com)
- 🔗 [LinkedIn](https://www.linkedin.com/in/yash-patil-997357330)
- 🐙 [GitHub](https://github.com/Yash-Patil-1)
