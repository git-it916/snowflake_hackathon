"""Global CSS -- Pure black + cyan/blue glass-morphism dark theme.

Injects Pretendard font, scrollbar, sidebar, glass-panel, and hidden
default Streamlit elements. Every page calls inject_global_css().
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Color palette (exported for chart builders)
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#4D9AFF",
    "accent_gradient": "linear-gradient(244deg, #9CEDFF 0%, #4D9AFF 100%)",
    "danger": "#FF4D4D",
    "warning": "#FFB84D",
    "success": "#4DFF91",
    "bg": "#000000",
    "card": "rgba(99,99,99,0.15)",
    "card_hover": "rgba(99,99,99,0.25)",
    "border": "rgba(161,161,161,0.3)",
    "text": "#FFFFFF",
    "text_sub": "#A1A1A1",
    "text_muted": "#636363",
}

CHART_PALETTE = [
    "#4D9AFF", "#9CEDFF", "#FF6B6B", "#FFB84D", "#A78BFA",
    "#4DFF91", "#FF6BC2", "#06D6A0", "#FF9F43", "#7C5CFC",
]
FUNNEL_COLORS = ["#4D9AFF", "#06D6A0", "#9CEDFF", "#FFB84D", "#4DFF91"]
GROWTH_COLOR = "#4DFF91"
DECLINE_COLOR = "#FF4D4D"
STABLE_COLOR = "#4D9AFF"

# Plotly dark layout (import this dict from chart builders)
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Pretendard, sans-serif", color="#e0e0e0"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
)


# ---------------------------------------------------------------------------
# Global CSS block
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

/* === Font (exclude Material Icons) === */
html, body, .stMarkdown, .stMetric, .stButton,
p, div, label, h1, h2, h3, input, button, textarea {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
    letter-spacing: -0.3px;
}

/* === Background === */
.stApp, .main, section[data-testid="stMain"] {
    background: #000000 !important;
}
header[data-testid="stHeader"] {
    background: rgba(0,0,0,0.8) !important;
    backdrop-filter: blur(20px);
}

/* === Glass panel utility class === */
.glass-panel {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
}

/* === Metric cards glass style === */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    overflow: hidden !important;
}
div[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    line-height: 1.3 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    line-height: 1.3 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* === Hide deploy button & hamburger === */
.stDeployButton { display: none !important; }
#MainMenu { display: none !important; }

/* === Sidebar (dark theme) === */
section[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.7);
}

/* === Page links (pill-shaped, transparent, cyan hover) === */
a[data-testid="stPageLink-NavLink"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 100px !important;
    padding: 8px 20px !important;
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
a[data-testid="stPageLink-NavLink"]:hover {
    border-color: #00E5FF !important;
    color: #00E5FF !important;
    background: rgba(0,229,255,0.05) !important;
}

/* === Scrollbar === */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* === Block container === */
.block-container {
    padding-top: 4.5rem !important;
    max-width: 1200px !important;
}

/* === Buttons === */
.stButton > button {
    border-radius: 100px !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00E5FF 0%, #3B82F6 100%) !important;
    border: none !important;
    color: #000 !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 24px rgba(0,229,255,0.4) !important;
    transform: translateY(-1px) !important;
}

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    color: rgba(255,255,255,0.4);
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.1) !important;
    color: #00E5FF !important;
}

/* === Chat === */
.stChatMessage {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    background: rgba(255,255,255,0.02) !important;
}

/* === Divider === */
hr {
    border-color: rgba(255,255,255,0.15) !important;
    margin: 1.5rem 0 !important;
}

/* === Caption === */
.stCaption, small { color: #636363 !important; }

/* === Spinner === */
.stSpinner > div { border-top-color: #00E5FF !important; }

/* === Prevent text overlap in narrow columns === */
div[data-testid="stColumn"] {
    overflow: hidden !important;
    min-width: 0 !important;
}
div[data-testid="stColumn"] p,
div[data-testid="stColumn"] span,
div[data-testid="stColumn"] label {
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
}

/* === Subheader spacing fix (prevent overlap with caption below) === */
h3 {
    margin-bottom: 0.3rem !important;
    line-height: 1.3 !important;
}

/* === Caption readability === */
.stCaption, small {
    line-height: 1.6 !important;
    word-wrap: break-word !important;
}

/* === Alert/Info boxes: transparent bg + colored border === */
div[data-testid="stAlert"] {
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(0,229,255,0.3) !important;
    border-radius: 12px !important;
}
div[data-testid="stAlert"] p {
    line-height: 1.5 !important;
    color: rgba(255,255,255,0.8) !important;
}

/* === Expander text overflow fix === */
details[data-testid="stExpander"] summary span {
    white-space: normal !important;
    word-wrap: break-word !important;
}

/* === Table text overflow fix === */
div[data-testid="stDataFrame"] {
    overflow-x: auto !important;
}

/* === Chat message spacing === */
.stChatMessage p {
    line-height: 1.6 !important;
}

/* === Keyframes === */
@keyframes ping {
    75%, 100% { transform: scale(2); opacity: 0; }
}
</style>
"""


def _is_sis() -> bool:
    """Streamlit in Snowflake 환경인지 감지."""
    try:
        import _snowflake  # noqa: F401 — SiS 전용 모듈
        return True
    except ImportError:
        return False


_SIS_OVERRIDE_CSS = """
<style>
/* SiS 전용: 흰 배경 + 검은 글씨 */
.stApp, .main, section[data-testid="stMain"] {
    background: #FFFFFF !important;
}
header[data-testid="stHeader"] {
    background: rgba(255,255,255,0.95) !important;
}
section[data-testid="stSidebar"] {
    background-color: #F8F9FA !important;
}
p, div, label, h1, h2, h3, span {
    color: #1a1a1a !important;
}
.stCaption, small { color: #666666 !important; }
div[data-testid="stMetric"] {
    background: rgba(0,0,0,0.03) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1a1a1a !important;
}
div[data-testid="stMetric"] label {
    color: rgba(0,0,0,0.5) !important;
}
</style>
"""


def inject_global_css() -> None:
    """Inject global CSS into the current page.

    SiS 환경이면 흰 배경 + 검은 글씨,
    로컬이면 다크 테마를 적용한다.
    """
    if _is_sis():
        st.markdown(_SIS_OVERRIDE_CSS, unsafe_allow_html=True)
    else:
        st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
