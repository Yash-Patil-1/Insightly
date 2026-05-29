"""
Insightly — Theme Manager
Handles dark/light/system theme switching via custom CSS injection.
"""

import streamlit as st

THEMES = {"Dark": "dark", "Light": "light", "System": "system"}


def init_theme():
    """Initialize theme in session state."""
    if "theme" not in st.session_state:
        st.session_state.theme = "System"


def get_theme_css(theme: str) -> str:
    """
    Return CSS overrides for the selected theme.
    Uses CSS custom properties so we can toggle cleanly.
    """
    is_dark = theme == "dark"
    if theme == "system":
        # Use prefers-color-scheme media query
        return _build_system_theme_css()
    return _build_fixed_theme_css(is_dark)


def _build_fixed_theme_css(is_dark: bool) -> str:
    """Build CSS for a fixed dark or light theme."""
    if is_dark:
        bg = "#0e1117"
        bg2 = "#1a1d24"
        card = "#262730"
        text = "#fafafa"
        muted = "#9e9e9e"
        border = "#3a3f4b"
        accent = "#4ea8de"
        success = "#4caf50"
        warning = "#ff9800"
        danger = "#f44336"
    else:
        bg = "#ffffff"
        bg2 = "#f5f7fa"
        card = "#eef0f4"
        text = "#1a1a2e"
        muted = "#6b7280"
        border = "#d1d5db"
        accent = "#2563eb"
        success = "#16a34a"
        warning = "#d97706"
        danger = "#dc2626"

    return f"""
    <style>
        :root {{
            --insightly-bg: {bg};
            --insightly-bg2: {bg2};
            --insightly-card: {card};
            --insightly-text: {text};
            --insightly-muted: {muted};
            --insightly-border: {border};
            --insightly-accent: {accent};
            --insightly-success: {success};
            --insightly-warning: {warning};
            --insightly-danger: {danger};
        }}
        .stApp {{
            background-color: var(--insightly-bg);
            color: var(--insightly-text);
        }}
        .insightly-card {{
            background: var(--insightly-card);
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            border: 1px solid var(--insightly-border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .insightly-badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .insightly-badge-success {{ background: var(--insightly-success); color: #fff; }}
        .insightly-badge-warning {{ background: var(--insightly-warning); color: #fff; }}
        .insightly-badge-danger {{ background: var(--insightly-danger); color: #fff; }}
        .insightly-kpi {{
            text-align: center;
            padding: 1rem;
            background: var(--insightly-card);
            border-radius: 12px;
            border: 1px solid var(--insightly-border);
        }}
        .insightly-kpi-label {{
            font-size: 0.8rem;
            color: var(--insightly-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .insightly-kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.2;
        }}
        .insightly-header {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .insightly-subheader {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--insightly-muted);
            margin-bottom: 0.75rem;
        }}
        .insightly-metric-change {{
            font-size: 0.85rem;
            font-weight: 500;
        }}
        div[data-testid="stFileUploader"] {{
            background: var(--insightly-card);
            border: 2px dashed var(--insightly-accent);
            border-radius: 16px;
            padding: 2rem;
        }}
        div[data-testid="stFileUploader"]:hover {{
            border-color: var(--insightly-success);
            transition: border-color 0.2s;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
        }}
        footer {{ display: none; }}
        #MainMenu {{ visibility: hidden; }}
    </style>
    """


def _build_system_theme_css() -> str:
    """Build CSS that respects the OS-level theme preference."""
    return """
    <style>
        :root {
            --insightly-bg: #ffffff;
            --insightly-bg2: #f5f7fa;
            --insightly-card: #eef0f4;
            --insightly-text: #1a1a2e;
            --insightly-muted: #6b7280;
            --insightly-border: #d1d5db;
            --insightly-accent: #2563eb;
            --insightly-success: #16a34a;
            --insightly-warning: #d97706;
            --insightly-danger: #dc2626;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --insightly-bg: #0e1117;
                --insightly-bg2: #1a1d24;
                --insightly-card: #262730;
                --insightly-text: #fafafa;
                --insightly-muted: #9e9e9e;
                --insightly-border: #3a3f4b;
                --insightly-accent: #4ea8de;
                --insightly-success: #4caf50;
                --insightly-warning: #ff9800;
                --insightly-danger: #f44336;
            }
        }
        .stApp { background-color: var(--insightly-bg); color: var(--insightly-text); }
        .insightly-card { background: var(--insightly-card); border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; border: 1px solid var(--insightly-border); box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .insightly-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
        .insightly-badge-success { background: var(--insightly-success); color: #fff; }
        .insightly-badge-warning { background: var(--insightly-warning); color: #fff; }
        .insightly-badge-danger { background: var(--insightly-danger); color: #fff; }
        .insightly-kpi { text-align: center; padding: 1rem; background: var(--insightly-card); border-radius: 12px; border: 1px solid var(--insightly-border); }
        .insightly-kpi-label { font-size: 0.8rem; color: var(--insightly-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        .insightly-kpi-value { font-size: 1.8rem; font-weight: 700; line-height: 1.2; }
        .insightly-header { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; }
        .insightly-subheader { font-size: 1.1rem; font-weight: 600; color: var(--insightly-muted); margin-bottom: 0.75rem; }
        div[data-testid="stFileUploader"] { background: var(--insightly-card); border: 2px dashed var(--insightly-accent); border-radius: 16px; padding: 2rem; }
        div[data-testid="stFileUploader"]:hover { border-color: var(--insightly-success); transition: border-color 0.2s; }
        footer { display: none; }
        #MainMenu { visibility: hidden; }
    </style>
    """


def apply_theme():
    """Read theme from session state, inject matching CSS."""
    init_theme()
    choice = st.session_state.theme
    mapped = THEMES.get(choice, "system")
    css = get_theme_css(mapped)
    st.markdown(css, unsafe_allow_html=True)


def theme_selector():
    """Render theme toggle in sidebar."""
    init_theme()
    st.markdown("### 🎨 Theme")
    current = st.session_state.theme
    idx = list(THEMES.keys()).index(current)
    selected = st.selectbox(
        "Appearance",
        options=list(THEMES.keys()),
        index=idx,
        label_visibility="collapsed",
        key="theme_selector",
    )
    if selected != current:
        st.session_state.theme = selected
        st.rerun()
