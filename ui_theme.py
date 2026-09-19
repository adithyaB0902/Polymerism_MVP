"""Visual theme and small reusable UI components for the POLYMEMSIM app.

Design tokens (see .streamlit/config.toml for the base Streamlit theme
these extend): a "lab instrument" palette grounded in the subject —
water/membrane flux teal as the primary accent, warm amber for
caution/calibration states, on a clinical near-white surface. Type is
IBM Plex Sans (an engineering-tool typeface family) for text and IBM
Plex Mono for data readouts (flux values, scores, counts), reinforcing
the instrumentation feel without inventing a generic dashboard look.
"""

import streamlit as st

INK = "#132A3A"
SURFACE = "#F3F7F6"
BORDER = "#DCE7E4"
TEAL = "#0E7C7B"
TEAL_DARK = "#0A5F5E"
AMBER = "#D9822B"
GREEN = "#2F8F4E"
RED = "#C1443C"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

/* ---------- Hero header ---------- */
.pms-hero {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.15rem 1.25rem 1.0rem 1.25rem;
    border: 1px solid rgba(14, 124, 123, 0.18);
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(14, 124, 123, 0.10), rgba(12, 60, 77, 0.02));
    box-shadow: 0 8px 24px rgba(19, 42, 58, 0.08);
    margin-bottom: 0.9rem;
}}
.pms-hero-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 3.2rem;
    height: 3.2rem;
    border-radius: 14px;
    background: linear-gradient(135deg, {TEAL}, {TEAL_DARK});
    color: white;
    font-size: 1.8rem;
    line-height: 1;
    box-shadow: 0 10px 24px rgba(14, 124, 123, 0.25);
}}
.pms-hero-title {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    font-size: 2.0rem;
    letter-spacing: -0.01em;
    color: {INK};
    margin: 0;
    padding: 0;
}}
.pms-hero-subtitle {{
    font-size: 0.95rem;
    color: #4B6B72;
    margin: 0.1rem 0 0 0;
}}

.pms-card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.9rem;
    margin: 1rem 0 1.2rem 0;
}}
.pms-card {{
    background: linear-gradient(180deg, #ffffff 0%, {SURFACE} 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1rem 1.05rem;
    box-shadow: 0 6px 18px rgba(19, 42, 58, 0.04);
}}
.pms-card h4 {{
    margin: 0 0 0.35rem 0;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5B7A80;
}}
.pms-card p {{
    margin: 0;
    color: {INK};
    line-height: 1.5;
}}
.pms-badge {{
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 999px;
    background: rgba(14, 124, 123, 0.10);
    color: {TEAL_DARK};
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
.pms-callout {{
    margin: 1rem 0 0.8rem 0;
    padding: 0.8rem 1rem;
    border-left: 4px solid {TEAL};
    border-radius: 10px;
    background: rgba(14, 124, 123, 0.05);
    color: {INK};
}}

/* ---------- Instrument-style stat strip ---------- */
.pms-stat-strip {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin: 0.6rem 0 1.1rem 0;
}}
.pms-stat {{
    flex: 1 1 140px;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-top: 3px solid {TEAL};
    border-radius: 6px;
    padding: 0.55rem 0.8rem 0.6rem 0.8rem;
}}
.pms-stat-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 1.5rem;
    color: {INK};
    line-height: 1.1;
}}
.pms-stat-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #5B7A80;
    margin-top: 0.15rem;
}}

/* ---------- Sidebar section labels ---------- */
.pms-sidebar-section {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {TEAL_DARK};
    border-bottom: 2px solid {BORDER};
    padding-bottom: 0.25rem;
    margin: 1.1rem 0 0.5rem 0;
}}
.pms-sidebar-section:first-of-type {{ margin-top: 0.2rem; }}

/* ---------- Tabs ---------- */
[data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
}}
[data-baseweb="tab"] {{
    font-weight: 600;
    font-size: 0.95rem;
    border-radius: 6px 6px 0 0;
    padding: 0.5rem 1rem;
}}
[data-baseweb="tab"][aria-selected="true"] {{
    background-color: {SURFACE};
    color: {TEAL_DARK};
}}
[data-baseweb="tab-highlight"] {{
    background-color: {TEAL} !important;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid {TEAL};
}}
.stButton > button[kind="primary"], .stButton > button:hover {{
    border-color: {TEAL_DARK};
    color: {TEAL_DARK};
}}

/* ---------- Metrics ---------- */
[data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace;
    color: {INK};
}}

/* ---------- Expanders & bordered containers ---------- */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 8px;
}}

/* ---------- Feasibility verdict badge ---------- */
.pms-verdict {{
    display: inline-block;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    margin: 0.3rem 0 0.6rem 0;
}}
.pms-verdict-strong {{ background: #E4F3EA; color: {GREEN}; }}
.pms-verdict-promising {{ background: #FCEFDD; color: {AMBER}; }}
.pms-verdict-low {{ background: #FBE9E7; color: {RED}; }}
.pms-verdict-neutral {{ background: {SURFACE}; color: {TEAL_DARK}; }}

.pms-score {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 3rem;
    line-height: 1;
}}
.pms-score-label {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #5B7A80;
    margin-bottom: 0.2rem;
}}

/* ---------- Pass/fail checklist ---------- */
.pms-check-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0;
    font-size: 0.92rem;
}}
.pms-check-pass {{ color: {GREEN}; font-weight: 700; }}
.pms-check-fail {{ color: {RED}; font-weight: 700; }}
</style>
"""


def inject_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(icon, title, subtitle):
    st.markdown(
        f"""<div class="pms-hero">
              <div class="pms-hero-icon">{icon}</div>
              <div>
                <p class="pms-hero-title">{title}</p>
                <p class="pms-hero-subtitle">{subtitle}</p>
              </div>
            </div>""",
        unsafe_allow_html=True)


def stat_strip(stats):
    """`stats` is a dict of {label: value}."""
    cards = "".join(
        f'<div class="pms-stat"><div class="pms-stat-value">{value}</div>'
        f'<div class="pms-stat-label">{label}</div></div>'
        for label, value in stats.items())
    st.markdown(f'<div class="pms-stat-strip">{cards}</div>', unsafe_allow_html=True)


def sidebar_section(container, icon, label):
    container.markdown(f'<div class="pms-sidebar-section">{icon} {label}</div>', unsafe_allow_html=True)


def verdict_badge(text, tone="neutral"):
    st.markdown(f'<span class="pms-verdict pms-verdict-{tone}">{text}</span>', unsafe_allow_html=True)


def score_display(value, max_value=100, label="Overall Screening Score"):
    if value >= 80:
        color = GREEN
    elif value >= 60:
        color = AMBER
    else:
        color = RED
    st.markdown(
        f"""<div class="pms-score-label">{label}</div>
            <div class="pms-score" style="color:{color};">{value:.1f} <span style="font-size:1.3rem;color:#8CA3A8;">/ {max_value}</span></div>""",
        unsafe_allow_html=True)


def check_row(label, passed):
    icon = "✓" if passed else "✗"
    cls = "pms-check-pass" if passed else "pms-check-fail"
    st.markdown(f'<div class="pms-check-row"><span class="{cls}">{icon}</span> {label}</div>',
               unsafe_allow_html=True)
