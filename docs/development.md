# 💻 Development Guide

## Setup for Development

```bash
# Clone the repository
git clone https://github.com/Yash-Patil-1/Insightly.git
cd Insightly

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov ruff  # Testing & linting tools
```

## Project Structure

```
Insightly/
├── src/                   # Source code
│   ├── loader.py          # Multi-format data loader
│   ├── profiler.py        # Auto-profiler & statistics
│   ├── cleaner.py         # Data cleaning operations
│   ├── analyzer.py        # Correlations, outliers, anomalies, insights
│   ├── visualizer.py      # Smart visualizer & chart gallery
│   ├── reporter.py        # Narrative report generator
│   ├── qa.py              # Natural language Q&A engine
│   └── theme.py           # Dark/Light/System theme manager
├── tests/                 # Unit tests
│   ├── test_loader.py
│   ├── test_profiler.py
│   ├── test_cleaner.py
│   ├── test_analyzer.py
│   ├── test_visualizer.py
│   ├── test_reporter.py
│   └── test_qa.py
├── sample_data/           # Sample datasets (3 formats)
│   ├── personal_finance.csv
│   ├── sales_report.xlsx
│   └── survey_results.json
├── docs/                  # Documentation
│   ├── getting_started.md
│   ├── usage.md
│   ├── architecture.md
│   └── development.md
├── app.py                 # Streamlit dashboard entry point
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```

## Coding Standards

### Style
- Follow **PEP 8** for Python code
- Use **type hints** for all function signatures
- Write **docstrings** for all classes and public methods (Google or reStructuredText style)
- Keep lines under **100 characters**

### Naming Conventions
- **Functions/Methods**: `snake_case` (e.g., `profile_dataframe()`, `detect_issues()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `SUPPORTED_FORMATS`, `QUESTION_PATTERNS`)
- **Private methods**: `_leading_underscore` (e.g., `_resolve_col()`, `_infer_type()`)
- **Module-level handlers**: `_handle_*` prefix for QA handler functions

### Module Pattern

Every analysis module should follow this pattern:

```python
\"\"\"
Insightly — Module Name
Brief description of what this module does.
\"\"\"

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def main_function(df: pd.DataFrame, param1: str, param2: int = 10) -> Dict[str, Any]:
    \"\"\"Do something useful with the DataFrame and return structured results.\"\"\"
    # ... implementation ...
    return {\"key\": value}


def _helper_function(value: Any) -> bool:
    \"\"\"Internal helper with descriptive name.\"\"\"
    return True
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# By module
pytest tests/test_loader.py -v
pytest tests/test_profiler.py -v
pytest tests/test_cleaner.py -v
pytest tests/test_analyzer.py -v
pytest tests/test_visualizer.py -v
pytest tests/test_reporter.py -v
pytest tests/test_qa.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Single test class
pytest tests/test_cleaner.py::TestDetectIssues -v

# Single test function
pytest tests/test_cleaner.py::TestDetectIssues::test_detects_missing -v
```

### Writing Tests

```python
# tests/test_my_module.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.my_module import my_function


class TestMyFunction:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "A": [1.0, 2.0, 3.0],
            "B": ["x", "y", "z"],
        })

    def test_normal_case(self, sample_df):
        result = my_function(sample_df)
        assert isinstance(result, dict)
        assert "expected_key" in result

    def test_empty_dataframe(self):
        result = my_function(pd.DataFrame())
        assert result == {}  # Should handle gracefully
```

### Test Fixtures

Common patterns used in existing tests:

- **`sample_df`** — Small clean DataFrame for basic testing
- **`dirty_df`** — DataFrame with known quality issues (nulls, outliers)
- **`finance_df`** — Load from sample_data/personal_finance.csv (QA tests)
- **`survey_df`** — Load from sample_data/survey_results.json (QA tests)
- **`missing_df`** — DataFrame with known missing value counts for validation

## Extending Insightly

### Adding a New Feature

1. **Implement the core logic** in the appropriate `src/` module
2. **Add a tab or UI component** in `app.py`
   - Use `st.markdown()` with `insightly-*` CSS classes for consistent styling
   - Store state in `st.session_state` for persistence across reruns
3. **Write tests** in the corresponding `tests/` file
4. **Run tests** to verify: `pytest tests/ -v`

### Adding a New Chart Type

1. Add the rendering logic in `src/visualizer.py`:
   ```python
   elif chart_type == "my_new_type":
       fig = px.my_new_chart(df, ...)
   ```
2. Add auto-detection in `suggest_chart_type()`
3. Add to the chart type selector options in `app.py` (Visualize tab)
4. Add test coverage in `tests/test_visualizer.py`

### Adding a New Q&A Pattern

1. Add a regex pattern and handler name to `QUESTION_PATTERNS` in `src/qa.py`:
   ```python
   r"^(?:what|which)\\s+(.+?)\\s+(?:has|with)\\s+(?:the\\s+)?(?:highest|largest|max)\\s+(.+?)$": "top_by_group",
   ```
2. Create the handler function:
   ```python
   def _handle_top_by_group(df: pd.DataFrame, group_col: str, val_col: str) -> Dict[str, Any]:
       # ... implementation ...
       return {"text": text, "figure": fig}
   ```
3. Register it in the `_HANDLERS` dict
4. Add pattern + handler tests in `tests/test_qa.py`

### Adding a New Theme

1. Add color variables in `src/theme.py`:
   - Add to `_build_fixed_theme_css()` — light and dark branches
   - Add to `_build_system_theme_css()`
2. Use the CSS variables (`--insightly-*`) in all styled elements

## Git Workflow

```bash
# Create a branch
git checkout -b feature/your-feature

# Make changes and commit
git add .
git commit -m "feat: add my new analysis feature"

# Run tests before pushing
pytest tests/ -v

# Lint
ruff check src/ --ignore E501

# Push and create PR
git push origin feature/your-feature
```

## Release Process

1. Update version in `app.py` sidebar footer
2. Update version in `pyproject.toml`
3. Run full test suite: `pytest tests/ -v`
4. Tag release: `git tag v1.0.0 && git push --tags`
