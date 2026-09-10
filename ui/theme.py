"""
RoleRadar's design tokens and global CSS.

Color/typography constants here are read by ui.components (score-tier
colors, status-tag colors, chart colors) so there is exactly one place
that defines "gold," "muted red," etc. The dark theme's base colors
(background/text/border/font) are set natively in .streamlit/config.toml
- Streamlit re-themes its own widgets (buttons, inputs, sliders,
dataframes, plotly/altair charts) from those values with no CSS needed.
inject_global_css() below only covers what config.toml's theme options
cannot reach: Streamlit chrome (header/footer/decoration), the sidebar
navigation list, and a handful of component-level refinements (tags,
expanders, alerts) to match the rest of the system.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------
# Color tokens - mirrors [theme] in .streamlit/config.toml. Python-side
# constants exist so ui.components can pick a color (e.g. a score-tier
# badge) without duplicating hex values or re-parsing the CSS.
# ---------------------------------------------------------------------

BG = "#07090D"
SIDEBAR_BG = "#080A0F"
SURFACE = "#0D1117"
SURFACE_2 = "#11161E"
BORDER = "#222936"
BORDER_STRONG = "#2C3444"

TEXT = "#E7EAF0"
TEXT_SECONDARY = "#8D96A8"
TEXT_MUTED = "#5E687A"

GOLD = "#D79A16"
GOLD_BRIGHT = "#F0B429"
POSITIVE = "#3E8F6F"
WARNING = "#D79A16"
NEGATIVE = "#B85450"
LINK = "#5B85AD"

FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_SERIF = "'Fraunces', Georgia, serif"
FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, monospace"

_GOOGLE_FONTS_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700"
    "&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700"
    "&family=JetBrains+Mono:wght@400;500;600"
    "&display=swap');"
)


def score_tier_color(score: float | None) -> str:
    """90+ bright gold, 80-89 gold, 70-79 neutral text, below that muted - matches the Discover card spec's score emphasis tiers."""
    if score is None:
        return TEXT_MUTED
    if score >= 90:
        return GOLD_BRIGHT
    if score >= 80:
        return GOLD
    if score >= 70:
        return TEXT
    return TEXT_SECONDARY


def status_color(kind: str) -> str:
    return {"positive": POSITIVE, "warning": WARNING, "negative": NEGATIVE}.get(kind, TEXT_SECONDARY)


def style_plotly_fig(fig: Any, *, accent: str = GOLD) -> Any:
    """
    Dark-theme a Plotly Express figure to match the rest of the system:
    near-black paper/plot background, muted gridlines, our font stack,
    and a single accent color for single-series bar/histogram/line/funnel
    traces (Plotly Express doesn't defer that color to Streamlit's theme
    at figure-build time, so it's set explicitly here rather than relying
    on theme inheritance). Never touches the underlying data/values.
    """
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_SANS, color=TEXT_SECONDARY, size=12),
        title_font=dict(family=FONT_SERIF, color=TEXT, size=16),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, color=TEXT_SECONDARY)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, color=TEXT_SECONDARY)
    fig.update_traces(marker_color=accent, selector=dict(type="bar"))
    fig.update_traces(marker_color=accent, selector=dict(type="histogram"))
    fig.update_traces(line_color=accent, marker_color=accent, selector=dict(type="scatter"))
    fig.update_traces(marker=dict(color=accent), connector=dict(line=dict(color=BORDER_STRONG)), selector=dict(type="funnel"))
    return fig


_CSS = f"""
<style>
{_GOOGLE_FONTS_IMPORT}

:root {{
    --rr-bg: {BG};
    --rr-sidebar-bg: {SIDEBAR_BG};
    --rr-surface: {SURFACE};
    --rr-surface-2: {SURFACE_2};
    --rr-border: {BORDER};
    --rr-border-strong: {BORDER_STRONG};
    --rr-text: {TEXT};
    --rr-text-secondary: {TEXT_SECONDARY};
    --rr-text-muted: {TEXT_MUTED};
    --rr-gold: {GOLD};
    --rr-gold-bright: {GOLD_BRIGHT};
    --rr-positive: {POSITIVE};
    --rr-warning: {WARNING};
    --rr-negative: {NEGATIVE};
    --rr-link: {LINK};
    --rr-font-sans: {FONT_SANS};
    --rr-font-serif: {FONT_SERIF};
    --rr-font-mono: {FONT_MONO};
}}

/* ---- Streamlit chrome: tighten padding, remove default branding ---- */
header[data-testid="stHeader"] {{
    background: var(--rr-bg);
    height: 2.25rem;
}}
[data-testid="stDecoration"] {{ display: none; }}
footer, #MainMenu {{ visibility: hidden; height: 0; }}
[data-testid="stToolbar"] {{ right: 0.5rem; }}

[data-testid="stAppViewContainer"] {{ background: var(--rr-bg); }}
[data-testid="stMainBlockContainer"], .block-container {{
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{
    background: var(--rr-sidebar-bg);
}}
[data-testid="stSidebar"] [data-testid="stMainBlockContainer"] {{
    padding-top: 0.5rem;
}}
[data-testid="stSidebarNav"] {{
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--rr-border);
}}
[data-testid="stSidebarNav"] ul {{ padding: 0 0.25rem; }}
[data-testid="stSidebarNav"] a {{
    border-radius: 0;
    border-left: 2px solid transparent;
    padding: 0.42rem 0.75rem;
    margin: 0.05rem 0;
    font-family: var(--rr-font-sans);
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--rr-text-secondary) !important;
    transition: border-color 120ms ease, color 120ms ease, background 120ms ease;
}}
[data-testid="stSidebarNav"] a:hover {{
    background: var(--rr-surface);
    color: var(--rr-text) !important;
}}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    border-left-color: var(--rr-gold);
    background: var(--rr-surface);
    color: var(--rr-gold-bright) !important;
    font-weight: 600;
}}
[data-testid="stSidebarNav"] span {{ font-family: inherit; }}
[data-testid="stSidebarNavSeparator"] {{ display: none; }}

/* ---- Headings ---- */
h1, h2, h3 {{
    font-family: var(--rr-font-serif);
    letter-spacing: -0.01em;
}}
h4, h5, h6 {{ font-family: var(--rr-font-sans); }}

/* ---- Buttons: squared, gold-outline primary, gray-outline secondary ---- */
.stButton button, .stFormSubmitButton button, [data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {{
    font-family: var(--rr-font-sans);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    transition: all 120ms ease;
}}
button[kind="primary"], [data-testid="stBaseButton-primary"] {{
    background: transparent !important;
    color: var(--rr-gold) !important;
    border: 1px solid var(--rr-gold) !important;
}}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
    background: var(--rr-gold) !important;
    color: #07090D !important;
}}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    color: var(--rr-text-secondary) !important;
    border: 1px solid var(--rr-border-strong) !important;
}}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {{
    border-color: var(--rr-text-secondary) !important;
    color: var(--rr-text) !important;
}}
button:disabled {{ opacity: 0.45 !important; }}

/* ---- Inputs / dropdowns / tags ---- */
[data-baseweb="tag"] {{
    background: var(--rr-surface-2) !important;
    border: 1px solid var(--rr-border-strong) !important;
    border-radius: 2px !important;
}}
[data-baseweb="tag"] span {{ color: var(--rr-text) !important; font-family: var(--rr-font-sans); }}

/* ---- Expanders ---- */
[data-testid="stExpander"] {{
    border: 1px solid var(--rr-border) !important;
    border-radius: 2px !important;
    background: var(--rr-surface) !important;
}}
[data-testid="stExpander"] summary {{
    font-family: var(--rr-font-sans);
    font-size: 0.85rem;
}}

/* ---- Alerts (info/success/warning/error) ---- */
[data-testid="stAlert"] {{
    font-family: var(--rr-font-sans);
    font-size: 0.85rem;
}}
[data-testid="stAlertContainer"] {{
    background: var(--rr-surface) !important;
    border: 1px solid var(--rr-border);
    border-left: 3px solid var(--rr-border-strong);
    border-radius: 2px;
    color: var(--rr-text) !important;
}}
[data-testid="stAlertContentInfo"] {{ border-left: none; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{ border-left-color: var(--rr-link); }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ border-left-color: var(--rr-positive); }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ border-left-color: var(--rr-warning); }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{ border-left-color: var(--rr-negative); }}
[data-testid="stAlertContainer"] p {{ color: var(--rr-text-secondary) !important; }}

/* ---- Dataframes (native st.dataframe, used sparingly) ---- */
[data-testid="stDataFrame"] {{ border-radius: 2px; }}

/* ---- Tabs ---- */
[data-baseweb="tab-list"] {{ gap: 0.25rem; border-bottom: 1px solid var(--rr-border); }}
[data-baseweb="tab"] {{
    font-family: var(--rr-font-sans);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--rr-text-secondary);
}}
[aria-selected="true"][data-baseweb="tab"] {{ color: var(--rr-gold-bright) !important; }}
[data-baseweb="tab-highlight"] {{ background-color: var(--rr-gold) !important; }}

/* ---- Misc text ---- */
[data-testid="stCaptionContainer"], .stCaption {{
    color: var(--rr-text-muted);
}}
a {{ color: var(--rr-link); }}
hr {{ border-color: var(--rr-border); }}

/* =====================================================================
   ui.components building blocks
   ===================================================================== */

.rr-eyebrow {{
    font-family: var(--rr-font-sans);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--rr-gold);
    margin: 0 0 0.2rem 0;
}}
.rr-page-header {{
    border-bottom: 1px solid var(--rr-border);
    padding-bottom: 0.9rem;
    margin-bottom: 1.1rem;
}}
.rr-page-header h1 {{
    font-family: var(--rr-font-serif);
    font-size: 2rem;
    font-weight: 600;
    margin: 0;
    line-height: 1.15;
    color: var(--rr-text);
}}
.rr-page-header p {{
    font-family: var(--rr-font-sans);
    font-size: 0.88rem;
    color: var(--rr-text-secondary);
    margin: 0.35rem 0 0 0;
    max-width: 62ch;
}}

.rr-section-header {{ margin: 0.25rem 0 0.6rem 0; }}
.rr-section-header h2 {{
    font-family: var(--rr-font-serif);
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
    color: var(--rr-text);
}}
.rr-section-header p {{
    font-family: var(--rr-font-sans);
    font-size: 0.82rem;
    color: var(--rr-text-secondary);
    margin: 0.15rem 0 0 0;
}}

/* ---- Metric strip ---- */
.rr-metric-strip {{
    display: flex;
    flex-wrap: wrap;
    border-top: 1px solid var(--rr-border);
    border-bottom: 1px solid var(--rr-border);
    margin: 0.75rem 0 1.25rem 0;
}}
.rr-metric {{
    flex: 1 1 108px;
    min-width: 90px;
    padding: 0.7rem 0.9rem;
    border-right: 1px solid var(--rr-border);
}}
.rr-metric:last-child {{ border-right: none; }}
.rr-metric-label {{
    font-family: var(--rr-font-sans);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--rr-text-muted);
    margin: 0 0 0.3rem 0;
    overflow-wrap: break-word;
    line-height: 1.3;
}}
.rr-metric-value {{
    font-family: var(--rr-font-serif);
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--rr-text);
    line-height: 1;
}}
.rr-metric-value.tone-gold {{ color: var(--rr-gold-bright); }}
.rr-metric-value.tone-positive {{ color: var(--rr-positive); }}
.rr-metric-value.tone-negative {{ color: var(--rr-negative); }}
.rr-metric-sub {{
    font-family: var(--rr-font-mono);
    font-size: 0.72rem;
    color: var(--rr-text-muted);
    margin-top: 0.2rem;
}}

/* ---- Score badge ---- */
.rr-score {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: var(--rr-font-mono);
    font-weight: 600;
    border: 1px solid var(--rr-border-strong);
    background: var(--rr-surface-2);
    color: var(--rr-text);
    line-height: 1;
}}
.rr-score.size-md {{ min-width: 2.4rem; height: 2.4rem; font-size: 1.05rem; }}
.rr-score.size-sm {{ min-width: 1.9rem; height: 1.9rem; font-size: 0.85rem; }}
.rr-score.size-lg {{ min-width: 3.4rem; height: 3.4rem; font-size: 1.5rem; }}
.rr-score.tier-elite {{ border-color: var(--rr-gold-bright); color: var(--rr-gold-bright); }}
.rr-score.tier-strong {{ border-color: var(--rr-gold); color: var(--rr-gold); }}
.rr-score.tier-mid {{ border-color: var(--rr-border-strong); color: var(--rr-text); }}
.rr-score.tier-low {{ border-color: var(--rr-border); color: var(--rr-text-secondary); }}

/* ---- Tags / signal chips ---- */
.rr-tag-row {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.15rem 0; }}
.rr-tag {{
    display: inline-block;
    font-family: var(--rr-font-mono);
    font-size: 0.7rem;
    padding: 0.14rem 0.5rem;
    border: 1px solid var(--rr-border-strong);
    background: var(--rr-surface-2);
    color: var(--rr-text-secondary);
    border-radius: 2px;
    white-space: nowrap;
}}
.rr-tag.kind-strength {{ border-color: rgba(62,143,111,0.45); color: var(--rr-positive); }}
.rr-tag.kind-gap {{ border-color: rgba(184,84,80,0.45); color: var(--rr-negative); }}
.rr-tag.kind-gold {{ border-color: var(--rr-gold); color: var(--rr-gold); }}

/* ---- Status / priority badges ---- */
.rr-badge {{
    display: inline-block;
    font-family: var(--rr-font-sans);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.16rem 0.55rem;
    border: 1px solid var(--rr-border-strong);
    border-radius: 2px;
    color: var(--rr-text-secondary);
}}
.rr-badge.tone-gold {{ border-color: var(--rr-gold); color: var(--rr-gold-bright); }}
.rr-badge.tone-positive {{ border-color: rgba(62,143,111,0.5); color: var(--rr-positive); }}
.rr-badge.tone-warning {{ border-color: rgba(215,154,22,0.5); color: var(--rr-warning); }}
.rr-badge.tone-negative {{ border-color: rgba(184,84,80,0.5); color: var(--rr-negative); }}

/* ---- Progress / score bars ---- */
.rr-bar-wrap {{ margin: 0.3rem 0; }}
.rr-bar-label {{
    display: flex;
    justify-content: space-between;
    font-family: var(--rr-font-sans);
    font-size: 0.72rem;
    color: var(--rr-text-secondary);
    margin-bottom: 0.22rem;
}}
.rr-bar-label .val {{ font-family: var(--rr-font-mono); color: var(--rr-text); }}
.rr-bar-track {{
    height: 6px;
    background: var(--rr-surface-2);
    border: 1px solid var(--rr-border);
    border-radius: 1px;
    overflow: hidden;
}}
.rr-bar-fill {{ height: 100%; background: var(--rr-gold); }}
.rr-bar-fill.tone-positive {{ background: var(--rr-positive); }}
.rr-bar-fill.tone-negative {{ background: var(--rr-negative); }}

/* ---- Meta / caption line (monospace metadata) ---- */
.rr-meta {{
    font-family: var(--rr-font-mono);
    font-size: 0.76rem;
    color: var(--rr-text-muted);
}}
.rr-meta .sep {{ margin: 0 0.4rem; color: var(--rr-border-strong); }}

/* ---- Custom HTML table ---- */
.rr-table-wrap {{ overflow-x: auto; border: 1px solid var(--rr-border); }}
table.rr-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: var(--rr-font-sans);
    font-size: 0.82rem;
}}
table.rr-table thead th {{
    text-align: left;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--rr-text-muted);
    background: var(--rr-surface-2);
    border-bottom: 1px solid var(--rr-border);
    padding: 0.5rem 0.7rem;
    white-space: nowrap;
}}
table.rr-table tbody td {{
    padding: 0.5rem 0.7rem;
    border-bottom: 1px solid var(--rr-border);
    color: var(--rr-text);
    vertical-align: top;
}}
table.rr-table tbody tr:hover td {{ background: var(--rr-surface); }}
table.rr-table tbody tr:last-child td {{ border-bottom: none; }}

/* ---- Activity / log feed ---- */
.rr-feed-row {{
    display: flex;
    gap: 0.75rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--rr-border);
    font-size: 0.8rem;
}}
.rr-feed-row:last-child {{ border-bottom: none; }}
.rr-feed-time {{
    font-family: var(--rr-font-mono);
    color: var(--rr-text-muted);
    flex-shrink: 0;
    width: 4.2rem;
}}
.rr-feed-text {{ color: var(--rr-text-secondary); }}

/* ---- Divider ---- */
.rr-divider {{ height: 1px; background: var(--rr-border); margin: 1.5rem 0; border: none; }}

/* ---- Card containers (via st.container(key=...), see ui.components) ---- */
div[class*="st-key-rrcard-"] {{
    border: 1px solid var(--rr-border);
    background: var(--rr-surface);
    padding: 1rem 1.15rem;
    margin-bottom: 0.85rem;
    border-radius: 2px;
    transition: border-color 120ms ease;
}}
div[class*="st-key-rrcard-"]:hover {{ border-color: var(--rr-border-strong); }}
div[class*="st-key-rrcard-hi-"] {{
    border-top: 2px solid var(--rr-gold);
}}
div[class*="st-key-rrpanel-"] {{
    border: 1px solid var(--rr-border);
    background: var(--rr-surface);
    padding: 0.9rem 1rem;
    border-radius: 2px;
    margin-bottom: 0.75rem;
}}

/* ---- Sidebar chrome: quick metrics + status footer ---- */
.rr-sidebar-label {{
    font-family: var(--rr-font-sans);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--rr-text-muted);
    margin: 0.9rem 0 0.4rem 0;
}}
.rr-sidebar-metrics {{ display: flex; flex-direction: column; gap: 0.35rem; }}
.rr-sidebar-metric-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: var(--rr-font-sans);
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--rr-text-secondary);
    padding: 0.2rem 0;
    border-bottom: 1px solid var(--rr-border);
}}
.rr-sidebar-metric-row:last-child {{ border-bottom: none; }}
.rr-sidebar-metric-row .val {{
    font-family: var(--rr-font-mono);
    font-size: 0.95rem;
    color: var(--rr-gold-bright);
}}
.rr-sidebar-footer {{
    font-family: var(--rr-font-mono);
    font-size: 0.66rem;
    color: var(--rr-text-muted);
    line-height: 1.7;
    border-top: 1px solid var(--rr-border);
    padding-top: 0.6rem;
    margin-top: 0.6rem;
}}
.rr-sidebar-footer .dot {{ color: var(--rr-positive); }}
.rr-sidebar-footer .dot.off {{ color: var(--rr-negative); }}

/* Sidebar demo-mode toggle: shrink to a small utility control */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    font-size: 0.72rem !important;
    color: var(--rr-text-secondary) !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
</style>
"""


def inject_global_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
