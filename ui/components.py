"""
Reusable rendering components for every RoleRadar page.

Every function here only renders - none compute a score, fetch data, or
own business logic. They take already-computed values (a fit score, a
list of tags, a metric dict) and emit consistent HTML/Streamlit calls
styled by ui.theme. Interactive elements (buttons, inputs) still go
through real Streamlit widgets - only presentation is templated here -
so callbacks, session_state, and reruns work exactly as they did
before this redesign.

Card-shaped sections use `st.container(key=...)` (not raw HTML) so a
real Streamlit button/expander can live inside a styled card border -
Streamlit exposes a stable `st-key-<key>` CSS class for a keyed
container (see ui.theme's `div[class*="st-key-rrcard-"]` rule), which
is what actually draws the border/padding/hover/top-accent.
"""

from __future__ import annotations

import html
from contextlib import contextmanager
from typing import Any, Iterable, Literal

import streamlit as st

from ui.theme import score_tier_color, status_color

_PRIORITY_LABEL = {"dream": "Dream", "high": "High", "interested": "Interested", "backup": "Backup"}
_PRIORITY_TONE = {"dream": "tone-gold", "high": "tone-warning", "interested": "", "backup": ""}


def _esc(value: Any) -> str:
    return html.escape(str(value))


# ---------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------


def render_page_header(eyebrow: str, title: str, subtitle: str | None = None) -> None:
    """Top-of-page identity block: small gold eyebrow label, serif title, muted subtitle."""
    parts = [
        '<div class="rr-page-header">',
        f'<p class="rr-eyebrow">{_esc(eyebrow)}</p>',
        f"<h1>{_esc(title)}</h1>",
    ]
    if subtitle:
        parts.append(f"<p>{_esc(subtitle)}</p>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str | None = None) -> None:
    parts = ['<div class="rr-section-header">', f"<h2>{_esc(title)}</h2>"]
    if subtitle:
        parts.append(f"<p>{_esc(subtitle)}</p>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="rr-divider"/>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Metric strip
# ---------------------------------------------------------------------


def render_metric_strip(metrics: list[dict[str, Any]]) -> None:
    """
    A dense horizontal row of metrics separated by thin vertical borders
    (not isolated white cards). Each item: {"label": str, "value": str,
    "sublabel": str | None, "tone": "default" | "gold" | "positive" | "negative"}.
    """
    cells = []
    for metric in metrics:
        tone = metric.get("tone", "default")
        tone_class = f" tone-{tone}" if tone != "default" else ""
        sub = f'<div class="rr-metric-sub">{_esc(metric["sublabel"])}</div>' if metric.get("sublabel") else ""
        cells.append(
            '<div class="rr-metric">'
            f'<div class="rr-metric-label">{_esc(metric["label"])}</div>'
            f'<div class="rr-metric-value{tone_class}">{_esc(metric["value"])}</div>'
            f"{sub}"
            "</div>"
        )
    st.markdown(f'<div class="rr-metric-strip">{"".join(cells)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Score badge
# ---------------------------------------------------------------------


def _score_tier_class(score: float | None) -> str:
    if score is None:
        return "tier-low"
    if score >= 90:
        return "tier-elite"
    if score >= 80:
        return "tier-strong"
    if score >= 70:
        return "tier-mid"
    return "tier-low"


def score_badge_html(score: float | None, size: Literal["sm", "md", "lg"] = "md") -> str:
    text = f"{score:.0f}" if score is not None else "—"
    return f'<span class="rr-score size-{size} {_score_tier_class(score)}">{text}</span>'


def render_score_badge(score: float | None, size: Literal["sm", "md", "lg"] = "md") -> None:
    st.markdown(score_badge_html(score, size), unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Tags / signal chips
# ---------------------------------------------------------------------


def tag_html(text: str, kind: Literal["neutral", "strength", "gap", "gold"] = "neutral") -> str:
    kind_class = f" kind-{kind}" if kind != "neutral" else ""
    return f'<span class="rr-tag{kind_class}">{_esc(text)}</span>'


def render_tags(tags: Iterable[str], kind: Literal["neutral", "strength", "gap", "gold"] = "neutral") -> None:
    tags = list(tags)
    if not tags:
        return
    row = "".join(tag_html(t, kind) for t in tags)
    st.markdown(f'<div class="rr-tag-row">{row}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Status / priority badges
# ---------------------------------------------------------------------


def status_badge_html(text: str, tone: Literal["default", "gold", "positive", "warning", "negative"] = "default") -> str:
    tone_class = f" tone-{tone}" if tone != "default" else ""
    return f'<span class="rr-badge{tone_class}">{_esc(text)}</span>'


def render_status_badge(text: str, tone: Literal["default", "gold", "positive", "warning", "negative"] = "default") -> None:
    st.markdown(status_badge_html(text, tone), unsafe_allow_html=True)


def priority_badge_html(priority: str | None) -> str:
    if not priority:
        return status_badge_html("Unflagged")
    return status_badge_html(_PRIORITY_LABEL.get(priority, priority).upper(), tone="gold" if priority == "dream" else "warning" if priority == "high" else "default")


def render_priority_badge(priority: str | None) -> None:
    st.markdown(priority_badge_html(priority), unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------


def render_bar(
    label: str,
    value: float,
    max_value: float = 100.0,
    *,
    value_text: str | None = None,
    tone: Literal["gold", "positive", "negative"] = "gold",
) -> None:
    pct = max(0.0, min(100.0, (value / max_value) * 100.0 if max_value else 0.0))
    display_value = value_text if value_text is not None else f"{value:.0f}"
    st.markdown(
        '<div class="rr-bar-wrap">'
        f'<div class="rr-bar-label"><span>{_esc(label)}</span><span class="val">{_esc(display_value)}</span></div>'
        f'<div class="rr-bar-track"><div class="rr-bar-fill tone-{tone}" style="width:{pct:.1f}%"></div></div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Meta line (monospace metadata: location / date / source / id)
# ---------------------------------------------------------------------


def render_meta_line(parts: Iterable[str]) -> None:
    parts = [p for p in parts if p]
    if not parts:
        return
    joined = '<span class="sep">·</span>'.join(_esc(p) for p in parts)
    st.markdown(f'<div class="rr-meta">{joined}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------


@contextmanager
def card(key: str, *, high_signal: bool = False):
    """A bordered card panel that can contain real Streamlit widgets (buttons, expanders)."""
    prefix = "rrcard-hi" if high_signal else "rrcard"
    with st.container(key=f"{prefix}-{key}"):
        yield


@contextmanager
def panel(key: str):
    """A lighter-weight bordered panel for dashboard side-sections (replaces st.container(border=True))."""
    with st.container(key=f"rrpanel-{key}"):
        yield


# ---------------------------------------------------------------------
# Tables (custom HTML - st.dataframe's canvas grid can't be restyled to
# match this system's dark, uppercase-header, hover-row look)
# ---------------------------------------------------------------------


def render_table(headers: list[str], rows: list[list[Any]]) -> None:
    if not rows:
        return
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{cell if isinstance(cell, str) and cell.startswith('<') else _esc(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        '<div class="rr-table-wrap"><table class="rr-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Activity / log feed
# ---------------------------------------------------------------------


def render_activity_feed(items: list[dict[str, str]]) -> None:
    """items: [{"time": "09/09", "text": "Stripe role score increased 74 -> 82"}]"""
    if not items:
        return
    rows = "".join(
        f'<div class="rr-feed-row"><span class="rr-feed-time">{_esc(item["time"])}</span>'
        f'<span class="rr-feed-text">{_esc(item["text"])}</span></div>'
        for item in items
    )
    st.markdown(rows, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Ranked list (e.g. highest-leverage skills)
# ---------------------------------------------------------------------


def render_ranked_list(items: list[dict[str, str]]) -> None:
    """items: [{"rank": "01", "label": "Kubernetes", "value": "+8.4 avg fit"}]"""
    rows = []
    for item in items:
        rows.append(
            '<div class="rr-feed-row">'
            f'<span class="rr-feed-time">{_esc(item["rank"])}</span>'
            f'<span class="rr-feed-text" style="flex:1;color:var(--rr-text);">{_esc(item["label"])}</span>'
            f'<span class="rr-meta">{_esc(item["value"])}</span>'
            "</div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------


def render_empty_state(message: str, detail: str | None = None) -> None:
    with panel("empty"):
        st.markdown(f'<p style="color:var(--rr-text-secondary);font-size:0.9rem;margin:0;">{_esc(message)}</p>', unsafe_allow_html=True)
        if detail:
            st.caption(detail)
