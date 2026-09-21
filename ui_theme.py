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

/* ---------- App shell ---------- */
body {{
    background: #EEF5F3;
}}
[data-testid="stAppViewContainer"] {{
    background:
        linear-gradient(135deg, rgba(255,255,255,0.9), rgba(238,245,243,0.96)),
        repeating-linear-gradient(0deg, rgba(14,124,123,0.025) 0, rgba(14,124,123,0.025) 1px, transparent 1px, transparent 28px);
}}
[data-testid="stMainBlockContainer"] {{
    max-width: 1440px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}}
[data-testid="stSidebar"] {{
    display: none;
}}
[data-testid="stAppViewContainer"] > .main {{
    margin-left: 0;
}}
[data-testid="stHeader"] {{
    background: rgba(255,255,255,0.7);
}}

/* ---------- Hero header ---------- */
.pms-hero {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.15rem 1.25rem 1.0rem 1.25rem;
    border: 1px solid rgba(14, 124, 123, 0.18);
    border-radius: 16px;
    background: linear-gradient(135deg, #E4F2EF 0%, #F8FBFA 58%, #FFFDF8 100%);
    box-shadow: 0 12px 30px rgba(19, 42, 58, 0.09);
    margin-bottom: 0.9rem;
    animation: pms-reveal 420ms ease-out both;
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

@keyframes pms-reveal {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}
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

/* ---------- Setup section labels ---------- */
.pms-setup-section {{
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
.pms-setup-section:first-of-type {{ margin-top: 0.2rem; }}

/* ---------- Tabs ---------- */
[data-baseweb="tab-list"] {{
    gap: 6px;
    border-bottom: 1px solid {BORDER};
    padding: 0.2rem 0.25rem 0;
    background: rgba(255,255,255,0.56);
    border-radius: 10px 10px 0 0;
}}
[data-baseweb="tab"] {{
    font-weight: 600;
    font-size: 0.88rem;
    color: #557177;
    border-radius: 8px 8px 0 0;
    padding: 0.65rem 0.95rem;
}}
[data-baseweb="tab"][aria-selected="true"] {{
    background-color: #E3F1EE;
    color: {TEAL_DARK};
}}
[data-baseweb="tab-highlight"] {{
    background-color: {TEAL} !important;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    min-height: 2.45rem;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid {TEAL};
    box-shadow: 0 2px 5px rgba(19,42,58,0.05);
    transition: transform 120ms ease, box-shadow 120ms ease;
}}
.stButton > button[kind="primary"], .stButton > button:hover {{
    border-color: {TEAL_DARK};
    color: {TEAL_DARK};
    box-shadow: 0 5px 12px rgba(14,124,123,0.14);
    transform: translateY(-1px);
}}
[data-testid="stDownloadButton"] > button {{
    min-height: 2.35rem;
    border-radius: 8px;
    background: #FFFFFF;
    border: 1px solid {BORDER};
    color: {TEAL_DARK};
}}

/* ---------- Form controls and data surfaces ---------- */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within,
div[data-testid="stNumberInput"]:focus-within,
div[data-testid="stTextInput"]:focus-within {{
    border-color: {TEAL} !important;
    box-shadow: 0 0 0 2px rgba(14,124,123,0.12);
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
    background: rgba(255,255,255,0.62);
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 8px;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}
hr {{
    border-color: {BORDER};
    margin: 1.35rem 0;
}}

@media (max-width: 800px) {{
    [data-testid="stMainBlockContainer"] {{ padding: 1rem 0.75rem 2rem 0.75rem; }}
    .pms-hero {{ padding: 0.9rem; gap: 0.7rem; }}
    .pms-hero-title {{ font-size: 1.55rem; }}
    [data-baseweb="tab"] {{ font-size: 0.78rem; padding: 0.5rem 0.55rem; }}
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


def inject_theme(mode="light"):
    """Inject the shared responsive visual system."""
    dark = mode == "dark"
    shell = f"""
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ display: none !important; }}
    [data-testid="stHeader"], [data-testid="stFooter"], [data-testid="stMainMenu"], [data-testid="stToolbar"] {{ display: none !important; }}
    [data-testid="stAppViewContainer"] {{
        background: {'radial-gradient(circle at 12% 10%, rgba(67, 211, 190, .34), transparent 34%), radial-gradient(circle at 88% 12%, rgba(137, 92, 226, .28), transparent 30%), linear-gradient(135deg, #081A31 0%, #0C4051 52%, #251A4C 100%)' if dark else 'radial-gradient(circle at 10% 0%, rgba(66, 190, 174, .22), transparent 34%), radial-gradient(circle at 90% 0%, rgba(127, 93, 201, .16), transparent 28%), linear-gradient(135deg, #F5FBFA 0%, #E7F4F1 52%, #F7F1FF 100%)'};
        min-height: 100vh;
    }}
    [data-testid="stMainBlockContainer"] {{ max-width: 1400px; padding: 1.25rem 1.25rem 4rem; }}
    .pms-glass, [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {'rgba(9, 27, 51, .70)' if dark else 'rgba(255,255,255,.66)'};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {'rgba(255,255,255,.22)' if dark else 'rgba(255,255,255,.86)'};
        border-radius: 20px;
        box-shadow: 0 16px 40px rgba(5, 20, 38, .12);
    }}
    .pms-top-nav {{ position: sticky; top: .6rem; z-index: 20; padding: .55rem .75rem; margin-bottom: 1rem; }}
    .pms-top-nav-title {{ font-size: 1.05rem; font-weight: 700; color: {'#F6FFFE' if dark else INK}; white-space: nowrap; }}
    .pms-stepper {{ display: flex; gap: .45rem; overflow-x: auto; padding: .35rem 0 .8rem; }}
    .pms-step {{ flex: 1 0 90px; min-width: 90px; text-align: center; font-size: .72rem; color: {'#C5D5E7' if dark else '#557177'}; }}
    .pms-step::before {{ content: ''; display: block; height: 5px; border-radius: 99px; margin-bottom: .35rem; background: {'#3C5572' if dark else '#CBDCD9'}; }}
    .pms-step.done::before, .pms-step.current::before {{ background: #35B9A6; }}
    .pms-step.current {{ color: {'#FFFFFF' if dark else TEAL_DARK}; font-weight: 700; }}
    .pms-hero-start {{ padding: clamp(1.4rem, 4vw, 3rem); border-radius: 24px; margin: 1rem 0 1.2rem; color: {'#F7FFFF' if dark else INK}; }}
    .pms-hero-start h1 {{ font-size: clamp(2rem, 5vw, 4rem); line-height: 1.04; margin: 0 0 1rem; max-width: 760px; }}
    .pms-hero-start p {{ max-width: 650px; font-size: 1.05rem; line-height: 1.65; }}
    .pms-metric-card {{ padding: 1rem 1.1rem; border-radius: 20px; background: {'rgba(42, 79, 113, .72)' if dark else 'rgba(255,255,255,.82)'}; box-shadow: 7px 7px 18px rgba(4,20,36,.10), -5px -5px 14px rgba(255,255,255,.30); transition: transform 180ms ease; }}
    .pms-metric-card:hover {{ transform: translateY(-3px); }}
    .pms-badge-glass {{ display: inline-flex; gap: .35rem; align-items: center; padding: .35rem .7rem; border-radius: 999px; background: rgba(53,185,166,.16); border: 1px solid rgba(53,185,166,.35); font-size: .78rem; font-weight: 700; }}
    .pms-page-footer {{ display: flex; justify-content: space-between; gap: .75rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid {'rgba(255,255,255,.22)' if dark else BORDER}; }}
    @media (max-width: 700px) {{ .pms-top-nav {{ top: 0; }} .pms-top-nav-title {{ font-size: .9rem; }} [data-testid="stMainBlockContainer"] {{ padding-left: .75rem; padding-right: .75rem; }} }}
    @media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation-duration: .01ms !important; transition-duration: .01ms !important; }} }}
    </style>
    """
    style_markup = CSS + shell
    # Streamlit's HTML element keeps style tags out of the Markdown renderer;
    # the fallback preserves compatibility with older Streamlit releases.
    if hasattr(st, "html"):
        st.html(style_markup)
    else:
        st.markdown(style_markup, unsafe_allow_html=True)


def glass_card(title, body, icon=""):
    st.markdown(f'<div class="pms-glass" style="padding:1.1rem 1.25rem; margin:.65rem 0;"><div class="pms-badge-glass">{icon} {title}</div><p>{body}</p></div>', unsafe_allow_html=True)


def metric_card(label, value, help_text=""):
    st.markdown(f'<div class="pms-metric-card"><div class="pms-stat-label">{label}</div><div class="pms-stat-value">{value}</div><div style="font-size:.78rem; color:#6B858A;">{help_text}</div></div>', unsafe_allow_html=True)


def badge(text, tone="info"):
    icons = {"simulated": "◌", "predicted": "◈", "measured": "●", "reliable": "✓", "caution": "!", "not reliable": "×"}
    st.markdown(f'<span class="pms-badge-glass">{icons.get(tone.lower(), "•")} {text}</span>', unsafe_allow_html=True)


def stepper(steps, current=0, completed=0):
    items = []
    for index, label in enumerate(steps):
        state = "done" if index < completed else "current" if index == current else "upcoming"
        items.append(f'<div class="pms-step {state}">{label}</div>')
    st.markdown(f'<div class="pms-stepper">{"".join(items)}</div>', unsafe_allow_html=True)


def top_nav(stages, mode="light"):
    left, right = st.columns([5, 1])
    with left:
        st.markdown('<div class="pms-top-nav pms-glass"><span class="pms-top-nav-title">◒ POLYMEMSIM</span></div>', unsafe_allow_html=True)
    with right:
        dark = st.toggle("Dark mode", value=mode == "dark", key="pms_dark_mode")
    selected = st.segmented_control("Stage", stages, default=st.session_state.get("pms_stage", stages[0]), key="pms_stage", label_visibility="collapsed")
    return selected or stages[0], dark


def page_footer(st, stages=None, back_label="Back", next_label="Next step"):
    stages = stages or ["Start", "Plan and Enter", "Analyze", "Optimize and Confirm", "More"]

    def move_stage(offset):
        current = st.session_state.get("pms_stage", stages[0])
        index = stages.index(current) if current in stages else 0
        st.session_state["pms_stage"] = stages[max(0, min(len(stages) - 1, index + offset))]

    left, _, right = st.columns([1, 4, 1])
    with left:
        st.button(f"← {back_label}", key=f"footer_back_{back_label}", on_click=move_stage, args=(-1,))
    with right:
        st.button(f"{next_label} →", key=f"footer_next_{next_label}", type="primary", on_click=move_stage, args=(1,))


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


def section_label(container, icon, label):
    container.markdown(f'<div class="pms-setup-section">{icon} {label}</div>', unsafe_allow_html=True)


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
