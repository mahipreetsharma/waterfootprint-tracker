"""AquaTrace — redesigned Streamlit frontend for water-footprint analysis.

This file keeps the existing project integrations:
    utils.database_handler.get_item_footprint
    utils.database_handler.load_water_data
    utils.vision_engine.load_yolo_model
    utils.vision_engine.process_frame

Run from the project root with: streamlit run app.py
"""

from __future__ import annotations

import html
import hashlib
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from utils.database_handler import get_item_footprint, load_water_data
from utils.vision_engine import load_yolo_model, process_frame


# -----------------------------------------------------------------------------
# 1. Page configuration and design system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AquaTrace | Water Footprint Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  --ink: #132c38; --ink-soft: #49616d; --muted: #6e828b;
  --paper: #f4f7f6; --white: #fff; --line: #dce7e7;
  --teal: #078b91; --teal-bright: #20c5b4; --navy: #082e3c;
  --blue: #2788c9; --blue-bg: #edf7ff; --green: #289966; --green-bg: #eefaf2;
  --grey: #71828b; --grey-bg: #f1f4f5;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { color: var(--ink); background: var(--paper); }
[data-testid="stHeader"] { background: rgba(244,247,246,.92); }
.block-container { max-width: 1440px; padding-top: 1.3rem; padding-bottom: 3.5rem; }
[data-testid="stSidebar"] { background: #0a303e; border-right: 1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] * { color: #eef9fa; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #c7dfe3; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }
[data-testid="stSidebar"] [data-testid="stProgressBar"] > div > div { background: #37c8b8; }

/* Water-lab hero: high contrast, calm background, animated ripple rings. */
.main-header { position:relative; isolation:isolate; overflow:hidden; display:flex; align-items:center; min-height:290px;
  padding: clamp(1.5rem, 4vw, 3rem); margin:.2rem 0 1.8rem; color:#fff; border-radius:26px;
  background: linear-gradient(108deg,#092c3a 0%,#0b4554 62%,#087d83 100%);
  box-shadow: 0 18px 40px rgba(7,48,62,.13); animation: rise-in .55s ease both; }
.main-header::before { content:""; position:absolute; z-index:-1; right:-68px; top:-190px; width:420px; height:420px;
  border:1px solid rgba(181,255,246,.3); border-radius:50%; box-shadow:0 0 0 28px rgba(181,255,246,.055),0 0 0 62px rgba(181,255,246,.035),0 0 0 98px rgba(181,255,246,.025);
  animation:ripple 14s ease-in-out infinite alternate; }
.main-header::after { content:""; position:absolute; z-index:-1; width:230px; height:230px; right:12%; bottom:-180px; border-radius:50%; background:rgba(32,197,180,.22); filter:blur(2px); }
.hero-layout { display:grid; grid-template-columns:minmax(0,1fr) 240px; gap:2rem; align-items:center; width:100%; }
.hero-content { max-width:820px; }
.hero-eyebrow { display:inline-flex; align-items:center; gap:.5rem; padding:.42rem .72rem; margin-bottom:.9rem; border:1px solid rgba(255,255,255,.24); border-radius:6px;
  color:#b9fff5; background:rgba(255,255,255,.06); font-size:.74rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
.hero-title { margin:0; color:#fff!important; font-family:'Manrope',sans-serif; font-size:clamp(2rem,4vw,3.5rem); font-weight:800; line-height:1.04; letter-spacing:-.055em; }
.hero-copy { max-width:730px; margin:1rem 0 1.2rem; color:#d8e9ed; font-size:1.02rem; line-height:1.75; }
.hero-pills { display:flex; flex-wrap:wrap; gap:.5rem; }
.hero-pill { padding:.45rem .65rem; border:1px solid rgba(255,255,255,.2); border-radius:8px; color:#effdfb; background:rgba(255,255,255,.07); font-size:.81rem; }
.hero-side { min-height:175px; display:flex; flex-direction:column; justify-content:center; padding:1.15rem; border:1px solid rgba(255,255,255,.2); border-radius:18px; background:rgba(255,255,255,.075); backdrop-filter:blur(8px); }
.hero-side-number { color:#9ff4e8; font-family:'Manrope',sans-serif; font-size:3.5rem; font-weight:800; line-height:1; }
.hero-side-title { margin-top:.7rem; color:#fff; font-size:1rem; font-weight:800; }
.hero-side-copy { margin-top:.3rem; color:#d3e9ec; font-size:.82rem; line-height:1.55; }

/* Clear hierarchy, white surfaces, gentle motion rather than excessive gradients. */
.section-kicker { margin:1.25rem 0 .3rem; color:var(--teal); font-size:.72rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.section-title { margin:0 0 .4rem; color:var(--ink); font-family:'Manrope',sans-serif; font-size:clamp(1.35rem,2.3vw,1.85rem); font-weight:800; letter-spacing:-.04em; }
.section-copy { color:var(--ink-soft); line-height:1.7; }
.surface-card { height:100%; padding:1.2rem 1.25rem; border:1px solid var(--line); border-radius:17px; background:#fff; box-shadow:0 5px 18px rgba(19,57,68,.035); animation:rise-in .45s ease both; }
.search-panel { padding:1rem 1.1rem .7rem; border:1px solid #cce2e1; border-radius:18px; background:#fff; box-shadow:0 8px 24px rgba(19,57,68,.045); }
.info-card { padding:1rem 1.1rem; border:1px solid var(--line); border-left:4px solid var(--teal); border-radius:12px; background:#fff; color:var(--ink); }
.blue-card,.green-card,.grey-card { height:100%; padding:1.1rem; margin-bottom:.8rem; border-radius:14px; transition:transform .18s ease,box-shadow .18s ease; }
.blue-card { border:1px solid #cbe6fa; border-left:4px solid var(--blue); background:var(--blue-bg); color:#164a6b; }
.green-card { border:1px solid #ccebd8; border-left:4px solid var(--green); background:var(--green-bg); color:#18593d; }
.grey-card { border:1px solid #dce4e8; border-left:4px solid var(--grey); background:var(--grey-bg); color:#384e59; }
.blue-card:hover,.green-card:hover,.grey-card:hover { transform:translateY(-2px); box-shadow:0 9px 20px rgba(20,63,76,.07); }
.pillar-title { display:flex; align-items:center; gap:.55rem; font-family:'Manrope',sans-serif; font-size:1.06rem; font-weight:800; }
.pillar-dot { display:inline-block; width:.7rem; height:.7rem; border-radius:3px; }
.pillar-blue { background:var(--blue); }.pillar-green { background:var(--green); }.pillar-grey { background:var(--grey); }
.fact-label { margin-top:.7rem; color:var(--muted); font-size:.7rem; font-weight:800; text-transform:uppercase; letter-spacing:.1em; }
.fact-value { color:var(--ink); font-size:.94rem; font-weight:700; line-height:1.45; }
.mode-caption { display:flex; align-items:center; gap:.55rem; margin:.3rem 0 .8rem; color:#55717d; font-size:.87rem; }
.mode-dot { width:.55rem; height:.55rem; border-radius:50%; background:#17a797; box-shadow:0 0 0 4px rgba(23,167,151,.12); }
.sidebar-brand { margin:.25rem 0 1.4rem; padding:1rem; border:1px solid rgba(255,255,255,.14); border-radius:16px; background:rgba(255,255,255,.055); }
.sidebar-brand-name { font-family:'Manrope',sans-serif; font-size:1.3rem; font-weight:800; letter-spacing:-.04em; color:#fff; }
.sidebar-brand-sub { margin-top:.25rem; color:#c0e0e5; font-size:.78rem; }
.sidebar-stat { display:flex; justify-content:space-between; align-items:center; gap:.7rem; margin:.65rem 0 .3rem; color:#e7fafb; font-size:.83rem; }
.formula-card { height:100%; padding:1.15rem; border:1px solid var(--line); border-radius:14px; background:#fff; }

[data-testid="stTextInputRootElement"] input,
[data-testid="stNumberInputField"] input,
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stTextArea"] textarea {
  border-radius:10px!important;
  border-color:#cfdede!important;
  background:#fff!important;
  color:#132c38!important;
  -webkit-text-fill-color:#132c38!important;
  opacity:1!important;
}
[data-testid="stTextInputRootElement"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
  color:#6e828b!important;
  -webkit-text-fill-color:#6e828b!important;
  opacity:1!important;
}
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] * {
  color:#49616d!important;
  -webkit-text-fill-color:#49616d!important;
}
[data-testid="stTextInputRootElement"] input:focus,
[data-testid="stNumberInputField"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color:#078b91!important;
  box-shadow:0 0 0 3px rgba(7,139,145,.12)!important;
}
button[data-testid="stBaseButton-primaryFormSubmit"],
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-secondary"],
div[data-testid="stDownloadButton"] button {
  min-height:2.65rem;
  border:1px solid #cbdcde!important;
  border-radius:10px;
  background:#e8f5f4!important;
  color:#075b62!important;
  -webkit-text-fill-color:#075b62!important;
  font-weight:800;
  transition:transform .17s ease,box-shadow .17s ease,background .17s ease;
}
button[data-testid="stBaseButton-primaryFormSubmit"],
button[data-testid="stBaseButton-primary"] {
  color:#fff!important;
  -webkit-text-fill-color:#fff!important;
  border:0!important;
  background:#087e86!important;
}
button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
  color:#fff!important;
  -webkit-text-fill-color:#fff!important;
  background:#066871!important;
  box-shadow:0 8px 19px rgba(7,99,108,.2);
}
button[data-testid="stBaseButton-secondary"]:hover,
div[data-testid="stDownloadButton"] button:hover {
  border-color:#078b91!important;
  background:#d8eeeb!important;
  color:#064b52!important;
  -webkit-text-fill-color:#064b52!important;
  box-shadow:0 6px 15px rgba(7,99,108,.12);
}
button[data-testid="stBaseButton-primary"]:disabled {
  opacity:.72;
  cursor:default;
}
[data-testid="stMetric"] { padding:1rem; border:1px solid var(--line); border-radius:14px; background:#fff; box-shadow:0 4px 14px rgba(19,57,68,.025); transition:transform .18s ease,box-shadow .18s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-2px); box-shadow:0 8px 20px rgba(19,57,68,.07); }
[data-testid="stMetricLabel"] { color:#627984!important; font-weight:700; }
[data-testid="stMetricValue"] { color:#123745!important; font-family:'Manrope',sans-serif; font-weight:800; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] button[data-baseweb="tab"] { padding:.65rem .85rem; border-radius:8px 8px 0 0; color:#607782; font-weight:800; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#087e86; background:#e7f7f5; }
[data-testid="stFileUploader"] { padding:.6rem; border:1px dashed #a8c9cb; border-radius:13px; background:#f9fdfd; }
[data-testid="stImage"] img { border-radius:13px; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stExpander"] { border-color:var(--line)!important; border-radius:12px!important; background:#fff; }
hr { border-color:var(--line); margin:1.5rem 0; }
.stCaption,[data-testid="stCaptionContainer"] { color:#6e828b; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
@keyframes rise-in { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes ripple { from { transform:translate3d(0,0,0) scale(1); } to { transform:translate3d(-16px,12px,0) scale(1.04); } }
@media(max-width:850px) { .hero-layout { grid-template-columns:1fr; } .hero-side { display:none; } .main-header { min-height:0; } }
@media(max-width:720px) { .block-container { padding-top:.8rem; } .main-header { border-radius:18px; } .hero-copy { font-size:.94rem; } [data-testid="stMetric"] { padding:.7rem; } }
@media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms!important; animation-iteration-count:1!important; transition-duration:.01ms!important; scroll-behavior:auto!important; } }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Data/model initialization and small helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading vision model…")
def get_model():
    """Load the YOLO model once per Streamlit server process."""
    return load_yolo_model()


@st.cache_data(show_spinner=False)
def get_water_database():
    """Cache water-footprint data separately from the heavyweight vision model."""
    data = load_water_data()
    return data or {}


water_db = get_water_database()

_SESSION_DEFAULTS = {
    "app_mode": "manual",
    "search_input": "",
    "lookup_quantity": 1,
    "submitted_search": "",
    "submitted_quantity": 1,
    "camera_active": False,
    "_aqua_camera_capture": None,
    "image_analysis_result": None,
}
for _key, _default in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


def release_camera() -> None:
    """Release an open camera before leaving the live camera view."""
    cap = st.session_state.get("_aqua_camera_capture")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state["_aqua_camera_capture"] = None


def set_mode(mode: str) -> None:
    """Switch views and clean up camera resources when appropriate."""
    previous_mode = st.session_state.get("app_mode", "manual")
    if previous_mode == "camera" and mode != "camera":
        release_camera()
    if previous_mode != mode:
        st.session_state["camera_active"] = False
    st.session_state["app_mode"] = mode


def start_camera() -> None:
    st.session_state["camera_active"] = True


def stop_camera() -> None:
    st.session_state["camera_active"] = False
    release_camera()


def clear_search() -> None:
    """Clear both the lookup field and the last submitted result."""
    st.session_state["search_input"] = ""
    st.session_state["lookup_quantity"] = 1
    st.session_state["submitted_search"] = ""
    st.session_state["submitted_quantity"] = 1


def fmt_liters(value: float | int) -> str:
    try:
        return f"{float(value):,.0f} L"
    except (TypeError, ValueError):
        return "0 L"


def percent(part: float, total: float) -> str:
    return f"{(part / total * 100):.1f}%" if total else "0.0%"


def safe_water_value(item_data: dict[str, Any], key: str) -> float:
    try:
        return float(item_data.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def has_component_breakdown(item_data: dict[str, Any]) -> bool:
    return all(
        isinstance(item_data.get(key), (int, float))
        for key in ("blue", "green", "grey")
    )


def footprint_unit(item_data: dict[str, Any]) -> str:
    unit = item_data.get("unit", "item")
    return unit if isinstance(unit, str) and unit else "item"


def display_footprint_analysis(query_name: str, item_data: dict[str, Any], quantity: int = 1) -> None:
    """Render overview, component breakdown and everyday comparisons."""
    total_l = safe_water_value(item_data, "total") * quantity
    components_available = has_component_breakdown(item_data)
    unit = footprint_unit(item_data)
    blue_l = safe_water_value(item_data, "blue") * quantity if components_available else 0
    green_l = safe_water_value(item_data, "green") * quantity if components_available else 0
    grey_l = safe_water_value(item_data, "grey") * quantity if components_available else 0
    safe_name = html.escape(query_name.strip().title())

    st.markdown(
        f'<div class="section-kicker">Your results</div>'
        f'<h2 class="section-title">Water footprint for {safe_name}</h2>'
        f'<p class="section-copy">Estimated footprint for <b>{quantity}</b> × '
        f'<b>{html.escape(unit)}</b>. '
        'Use the tabs below to explore the total, water types and everyday equivalents.</p>',
        unsafe_allow_html=True,
    )
    if item_data.get("estimate_type") == "illustrative_generic":
        st.warning(
            item_data.get(
                "estimate_note",
                "This is a rough illustrative estimate, not an authoritative "
                "product-specific measurement.",
            )
        )

    tab_summary, tab_breakdown, tab_equivalents = st.tabs(
        ["✦ Overview", "◉ Component breakdown", "↗ Everyday equivalents"]
    )

    with tab_summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Total footprint · {unit}", fmt_liters(total_l))
        if components_available:
            m2.metric("Blue water", fmt_liters(blue_l), percent(blue_l, total_l))
            m3.metric("Green water", fmt_liters(green_l), percent(green_l, total_l))
            m4.metric("Grey water", fmt_liters(grey_l), percent(grey_l, total_l))
        else:
            m2.metric("Blue water", "Not available")
            m3.metric("Green water", "Not available")
            m4.metric("Grey water", "Not available")
        st.markdown(
            '<div class="surface-card" style="margin-top:1rem;">'
            '<b>Reading this result</b><p style="margin:.45rem 0 0;color:#58717d;line-height:1.7;">'
            'A water footprint estimates the freshwater consumed or polluted across a product’s '
            'life cycle—not just the water used directly at home. Values depend on the underlying '
            'database and the production context.</p></div>',
            unsafe_allow_html=True,
        )

    with tab_breakdown:
        col_text, col_chart = st.columns([1.05, 1], gap="large")
        if components_available:
            with col_text:
                st.markdown("#### Where does the water go?")
                st.markdown(
                    f'<div class="blue-card"><b>Blue water · {fmt_liters(blue_l)}</b><br>'
                    '<span>Fresh surface water and groundwater from rivers, lakes and aquifers, often used for irrigation or manufacturing.</span></div>'
                    f'<div class="green-card"><b>Green water · {fmt_liters(green_l)}</b><br>'
                    '<span>Rainwater held in soil and used by plants through growth and evapotranspiration.</span></div>'
                    f'<div class="grey-card"><b>Grey water · {fmt_liters(grey_l)}</b><br>'
                    '<span>An indicator of the freshwater volume needed to assimilate pollutants to applicable water-quality standards.</span></div>',
                    unsafe_allow_html=True,
                )
            with col_chart:
                st.markdown("#### Volumetric distribution")
                chart_df = pd.DataFrame(
                    {
                        "Water component": ["Blue", "Green", "Grey"],
                        "Liters": [blue_l, green_l, grey_l],
                    }
                )
                st.bar_chart(chart_df, x="Water component", y="Liters", height=300)
                st.caption("The chart uses the component values returned by your local footprint database.")
        else:
            with col_text:
                st.info(
                    "The source provides an aggregate footprint total only; "
                    "blue, green, and grey component values are not available."
                )
            with col_chart:
                st.markdown("#### Component distribution")
                st.caption("No chart is shown because component-level values are unavailable.")

    with tab_equivalents:
        st.markdown("#### Put the footprint into perspective")
        st.caption("Illustrative comparisons based on the assumptions shown below each metric.")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("10-minute showers", f"{total_l / 65:,.1f}")
        e1.caption("Assumes 65 L per shower.")
        e2.metric("Drinking-water days", f"{total_l / 3:,.0f}")
        e2.caption("Assumes 3 L per person per day.")
        e3.metric("Full bathtubs", f"{total_l / 150:,.1f}")
        e3.caption("Assumes 150 L per tub.")
        e4.metric("Toilet flushes", f"{total_l / 6:,.0f}")
        e4.caption("Assumes 6 L per flush.")


def render_detection_telemetry(detected_items: list[str], database: dict[str, Any]) -> None:
    """Render recognized objects and their stored footprint estimates."""
    matched_items = [item for item in detected_items if item in database]
    if not matched_items:
        st.warning("No database-matched objects were detected. Try an image with clearer, well-lit objects.")
        return

    total_image_water = sum(safe_water_value(database[item], "total") for item in matched_items)
    st.metric("Estimated detected footprint", fmt_liters(total_image_water))
    if any(
        database[item].get("estimate_type") == "illustrative_generic"
        for item in matched_items
    ):
        st.warning(
            "Some detected-item values are rough illustrative estimates, not "
            "authoritative product-specific measurements."
        )
    st.markdown(
        '<div class="section-kicker">Recognized objects</div>',
        unsafe_allow_html=True,
    )
    for item in matched_items:
        data = database[item]
        with st.container(border=True):
            item_col, total_col = st.columns([2, 1])
            item_col.markdown(f"**{html.escape(item.title())}**")
            if has_component_breakdown(data):
                item_col.caption(
                    f"Blue {fmt_liters(safe_water_value(data, 'blue'))}  ·  "
                    f"Green {fmt_liters(safe_water_value(data, 'green'))}  ·  "
                    f"Grey {fmt_liters(safe_water_value(data, 'grey'))}"
                )
            else:
                item_col.caption("Blue/green/grey split unavailable from the source.")
            total_col.metric(
                f"Per detected item · {footprint_unit(data)}",
                fmt_liters(safe_water_value(data, "total")),
            )


# -----------------------------------------------------------------------------
# 3. Sidebar: concise context and conservation guidance
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div style="font-size:1.65rem;margin-bottom:.2rem;">💧</div>'
        '<div class="sidebar-brand-name">AquaTrace</div>'
        '<div class="sidebar-brand-sub">Water footprint intelligence</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### The invisible water story")
    st.markdown(
        "Much of a product’s water footprint is **embedded in its supply chain**—"
        "from growing raw materials to manufacturing and transport."
    )
    st.markdown("---")
    st.markdown("### Global footprint mix")
    st.markdown('<div class="sidebar-stat"><span>Green water</span><b>74%</b></div>', unsafe_allow_html=True)
    st.progress(0.74)
    st.markdown('<div class="sidebar-stat"><span>Grey water</span><b>15%</b></div>', unsafe_allow_html=True)
    st.progress(0.15)
    st.markdown('<div class="sidebar-stat"><span>Blue water</span><b>11%</b></div>', unsafe_allow_html=True)
    st.progress(0.11)
    st.caption("Illustrative global shares from the source content; shares vary by methodology and dataset.")
    st.markdown("---")
    st.markdown("### Small choices, meaningful impact")
    st.markdown(
        "- **Keep products longer.** Extending product life can reduce demand for new manufacturing.\n"
        "- **Reuse and repair.** Second-hand and refurbished goods can avoid some new-material demand.\n"
        "- **Look beyond the tap.** Food, textiles and electronics can carry substantial embedded water."
    )
    st.markdown("---")
    st.caption("Assessment context: ISO 14046 and Water Footprint Network concepts. Product values come from the app’s local database.")


# -----------------------------------------------------------------------------
# 4. Hero header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
      <div class="hero-layout">
        <div class="hero-content">
          <div class="hero-eyebrow">AQUATRACE · WATER FOOTPRINT INTELLIGENCE</div>
          <h1 class="hero-title">Every product has<br>a water story.</h1>
          <p class="hero-copy">Explore the freshwater used, the rainwater consumed by plants, and the water-quality pressure associated with everyday goods. Search a product or use computer vision to make hidden water more visible.</p>
          <div class="hero-pills">
            <span class="hero-pill">Blue · Surface & groundwater</span>
            <span class="hero-pill">Green · Rainwater in soil</span>
            <span class="hero-pill">Grey · Pollution indicator</span>
          </div>
        </div>
        <div class="hero-side">
          <div class="hero-side-number">3</div>
          <div class="hero-side-title">connected dimensions</div>
          <div class="hero-side-copy">One footprint can describe water quantity, rainfall dependence and water-quality pressure.</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Water footprint, explained</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">More than the water you see</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">A <b>water footprint</b> estimates the freshwater consumed and the water pollution associated with producing the goods and services we use. It follows a product beyond the tap—through farming, raw materials, processing and manufacturing.</p>',
    unsafe_allow_html=True,
)
intro_a, intro_b, intro_c = st.columns(3, gap="medium")
with intro_a:
    st.markdown(
        '<div class="surface-card"><div class="fact-label">01 · Direct water</div>'
        '<h3 style="margin:.45rem 0 .6rem;font-family:Manrope,sans-serif;">What you use</h3>'
        '<p style="color:#49616d;line-height:1.7;margin:0;">Water used at home or work—for drinking, bathing, cleaning and cooking. This is visible, but it is only one part of the story.</p></div>',
        unsafe_allow_html=True,
    )
with intro_b:
    st.markdown(
        '<div class="surface-card"><div class="fact-label">02 · Virtual water</div>'
        '<h3 style="margin:.45rem 0 .6rem;font-family:Manrope,sans-serif;">What is embedded</h3>'
        '<p style="color:#49616d;line-height:1.7;margin:0;">Water associated with growing food and fibres, extracting materials, making components and preparing products for sale.</p></div>',
        unsafe_allow_html=True,
    )
with intro_c:
    st.markdown(
        '<div class="surface-card"><div class="fact-label">03 · Why it matters</div>'
        '<h3 style="margin:.45rem 0 .6rem;font-family:Manrope,sans-serif;">Where & how</h3>'
        '<p style="color:#49616d;line-height:1.7;margin:0;">A litre used in a water-stressed basin can have a different impact from a litre used where water is abundant. Production method and pollution also matter.</p></div>',
        unsafe_allow_html=True,
    )


# 5. Instant lookup — analyze explicitly, retain results until cleared
# -----------------------------------------------------------------------------
st.markdown('<div class="section-kicker">Start with an item</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Calculate a product footprint</h2>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy" style="margin-bottom:.8rem;">Search the local footprint database, choose a quantity, then press <b>Analyze footprint</b>. Results stay visible until you clear or submit another search.</div>',
    unsafe_allow_html=True,
)
with st.container(border=True):
    form_col, clear_col = st.columns([5.2, 1], vertical_alignment="bottom")
    with form_col:
        with st.form("footprint_lookup_form", clear_on_submit=False):
            search_col, quantity_col, submit_col = st.columns([3.4, 1, 1.25], vertical_alignment="bottom")
            with search_col:
                search_query = st.text_input(
                    "Product or everyday item",
                    key="search_input",
                    placeholder="Try coffee, cotton shirt, rice…",
                    help="Names must be supported by your local water-footprint database.",
                )
            with quantity_col:
                item_qty = st.number_input(
                    "Quantity",
                    min_value=1,
                    max_value=100,
                    step=1,
                    key="lookup_quantity",
                    help="Multiplies the estimate by the recorded unit, such as one item or one 100 g portion.",
                )
            with submit_col:
                submit_lookup = st.form_submit_button("Analyze footprint", type="primary", use_container_width=True)
    with clear_col:
        st.button("Clear search", key="clear_lookup_button", on_click=clear_search, use_container_width=True)

if submit_lookup:
    st.session_state["submitted_search"] = search_query.strip().lower()
    st.session_state["submitted_quantity"] = int(item_qty)

active_search = st.session_state.get("submitted_search", "")
active_quantity = int(st.session_state.get("submitted_quantity", 1))
query_is_pending = bool(search_query.strip()) and search_query.strip().lower() != active_search
if query_is_pending and active_search:
    st.info("The search field has changed. The results below still show the last submitted item; press **Analyze footprint** to update them.")
if active_search:
    item_data = get_item_footprint(active_search)
    if item_data:
        display_footprint_analysis(active_search, item_data, active_quantity)
    else:
        st.warning(
            f"**{html.escape(active_search.title())}** was not found in the local database. "
            "Check the spelling or try another item supported by your dataset."
        )
        with st.expander("How to get a better match"):
            st.markdown(
                "- Try a shorter item name, such as `rice` instead of a full sentence.\n"
                "- Use the product label/category expected by your dataset.\n"
                "- Remember that this lookup only returns items present in the local database."
            )
elif query_is_pending:
    st.info("Your text has not been analyzed yet. Press **Analyze footprint** to submit it.")
else:
    st.markdown(
        '<div class="mode-caption"><span class="mode-dot"></span>Ready to explore — submit an item above or choose a detection mode below.</div>',
        unsafe_allow_html=True,
    )

st.divider()


# -----------------------------------------------------------------------------
# 6. Detection mode picker
# -----------------------------------------------------------------------------
st.markdown('<div class="section-kicker">Choose your workflow</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Explore with three detection modes</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Use a database lookup for a known item, computer vision for a photo, or your webcam for live scanning.</p>',
    unsafe_allow_html=True,
)

mode_cols = st.columns(3, gap="medium")
mode_options = [
    ("manual", "⌕  Manual lookup", "Search and compare database estimates"),
    ("image", "▧  Image scanner", "Analyze objects in a saved photo"),
    ("camera", "◎  Live camera", "Scan a frame from the local webcam"),
]
for col, (mode_id, label, caption) in zip(mode_cols, mode_options):
    with col:
        active = st.session_state.app_mode == mode_id
        with st.container(border=True):
            st.markdown(
                f'<div style="font-size:.76rem;letter-spacing:.1em;text-transform:uppercase;color:{"#087e8b" if active else "#7a929c"};font-weight:800;margin-bottom:.4rem;">'
                f'{"● SELECTED MODE" if active else "DETECTION MODE"}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{label}**")
            st.caption(caption)
            st.button(
                "✓ Current mode" if active else "Open this mode →",
                key=f"mode_{mode_id}",
                type="primary" if active else "secondary",
                use_container_width=True,
                disabled=active,
                on_click=set_mode,
                args=(mode_id,),
            )

st.markdown(" ")


# -----------------------------------------------------------------------------
# 7. Mode: Manual search
# -----------------------------------------------------------------------------
if st.session_state.app_mode == "manual":
    st.markdown('<div class="section-kicker">Mode 01</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Manual object lookup</h3>', unsafe_allow_html=True)
    guide_a, guide_b = st.columns([1.15, 1], gap="large")
    with guide_a:
        st.markdown(
            '<div class="surface-card">'
            '<h4 style="margin-top:0;">How to get a useful estimate</h4>'
            '<p style="color:#58717d;line-height:1.75;">① Search for an item in the lookup above.<br>② Adjust the quantity to represent multiple units.<br>③ Review the blue, green and grey components to understand the type of water use.</p>'
            '<p style="color:#58717d;line-height:1.75;margin-bottom:0;">The result is a database estimate, not a live measurement of water flowing from a tap.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with guide_b:
        st.markdown(
            '<div class="surface-card" style="height:100%;background:linear-gradient(140deg,#e8fbf7,#f8fdfd);">'
            '<div style="font-size:1.8rem;">🌱</div>'
            '<h4 style="margin:.45rem 0;">A useful question to ask</h4>'
            '<p style="color:#58717d;line-height:1.75;">“Which part of this product’s footprint is largest—and what choices might reduce it?”</p>'
            '<span style="display:inline-block;background:#d3f5eb;color:#087e75;padding:.35rem .65rem;border-radius:999px;font-weight:700;font-size:.78rem;">Understand · Compare · Improve</span>'
            '</div>',
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 8. Mode: Image upload detection
# -----------------------------------------------------------------------------
elif st.session_state.app_mode == "image":
    st.markdown('<div class="section-kicker">Mode 02</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Analyze an image</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Choose a clear photo. The vision model detects only its supported object classes that map to entries in your local water-footprint database; products without a matching model class cannot be recognized. People are excluded. Uploading alone does not run the analysis; use the button below.</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drop an image here or browse files",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploaded_image_file",
        help="Supported formats: JPG, JPEG, PNG and WEBP.",
    )
    uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else b""
    upload_digest = hashlib.sha256(uploaded_bytes).hexdigest() if uploaded_bytes else ""
    analyze_col, tip_col = st.columns([1, 2], vertical_alignment="center")
    with analyze_col:
        analyze_image = st.button(
            "Analyze image →",
            key="analyze_image_button",
            type="primary",
            use_container_width=True,
            disabled=not bool(uploaded_bytes),
        )
    with tip_col:
        st.caption("Tip: good lighting, visible objects and minimal clutter generally improve detection quality.")

    if analyze_image and uploaded_bytes:
        bgr_image = cv2.imdecode(np.frombuffer(uploaded_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr_image is None:
            st.error("The image could not be decoded. Please try another JPG, PNG or WEBP file.")
            st.session_state["image_analysis_result"] = None
        else:
            try:
                with st.spinner("Analyzing image and matching detected objects…"):
                    vision_model = get_model()
                    annotated_frame, detected_items = process_frame(bgr_image, vision_model, water_db)
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.session_state["image_analysis_result"] = {
                    "digest": upload_digest,
                    "annotated_rgb": annotated_rgb,
                    "detected_items": list(detected_items or []),
                }
            except Exception as exc:
                st.session_state["image_analysis_result"] = None
                st.error("Image analysis did not complete. Verify that the YOLO model and database are configured correctly.")
                with st.expander("Technical details"):
                    st.code(str(exc))

    image_result = st.session_state.get("image_analysis_result")
    if image_result and image_result.get("digest") == upload_digest and upload_digest:
        img_col, impact_col = st.columns([1.3, 1], gap="large")
        with img_col:
            st.markdown("#### Detection preview")
            st.image(image_result["annotated_rgb"], caption="Objects recognized by AquaTrace", use_container_width=True)
        with impact_col:
            st.markdown("#### Footprint telemetry")
            render_detection_telemetry(image_result["detected_items"], water_db)
    elif uploaded_file is not None:
        st.info("Image selected. Press **Analyze image** to run object detection.")
    else:
        st.markdown(
            '<div class="surface-card" style="text-align:center;padding:1.6rem;">'
            '<div style="font-size:2rem;">▧</div><h4 style="margin:.5rem 0;">Your image workspace is ready</h4>'
            '<p style="margin:0;color:#607782;">Upload a photo to begin. Analysis starts only when you choose to run it.</p></div>',
            unsafe_allow_html=True,
        )


# 9. Mode: Live camera detection
# -----------------------------------------------------------------------------
elif st.session_state.app_mode == "camera":
    st.markdown('<div class="section-kicker">Mode 03</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">Live camera scanner</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Start the scanner to analyze frames from the webcam attached to the machine running Streamlit. Detection is limited to supported model classes that map to the water-footprint database; people and unsupported products are excluded. This OpenCV webcam mode is intended for a local app; a cloud-hosted app generally needs a browser-streaming component such as streamlit-webrtc.</p>',
        unsafe_allow_html=True,
    )

    button_col, status_col = st.columns([1, 2], vertical_alignment="center")
    with button_col:
        if st.session_state.get("camera_active", False):
            st.button("■ Stop live scanner", key="stop_camera_button", on_click=stop_camera, use_container_width=True)
        else:
            st.button("▶ Start live scanner", key="start_camera_button", type="primary", on_click=start_camera, use_container_width=True)
    with status_col:
        if st.session_state.get("camera_active", False):
            st.markdown('<div class="mode-caption"><span class="mode-dot"></span>Scanner active · frames are processed by the local application.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="mode-caption">◎ Scanner paused · press Start when you are ready.</div>', unsafe_allow_html=True)

    if st.session_state.get("camera_active", False) and hasattr(st, "fragment"):
        # Periodic fragments allow the Stop button outside the fragment to remain usable.
        @st.fragment(run_every=0.35)
        def live_camera_fragment():
            if not st.session_state.get("camera_active", False):
                release_camera()
                return
            frame_col, telemetry_col = st.columns([1.3, 1], gap="large")
            cap = st.session_state.get("_aqua_camera_capture")
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                st.session_state["_aqua_camera_capture"] = cap
            if not cap.isOpened():
                st.error("The webcam could not be opened. Check camera availability and OS permissions, then stop and start the scanner again.")
                release_camera()
                return
            ok, frame = cap.read()
            if not ok:
                with frame_col:
                    st.warning("Waiting for a camera frame…")
                return
            try:
                annotated, items = process_frame(frame, get_model(), water_db)
                with frame_col:
                    st.markdown("#### Live preview")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                with telemetry_col:
                    st.markdown("#### Frame impact")
                    render_detection_telemetry(items, water_db)
            except Exception as exc:
                st.error("Live frame analysis failed. Check the model and camera configuration.")
                with st.expander("Technical details"):
                    st.code(str(exc))
        live_camera_fragment()
    elif st.session_state.get("camera_active", False):
        # Safe fallback for Streamlit versions without periodic fragments.
        st.info("Your Streamlit version does not support periodic live fragments. Capture a browser-camera frame below to analyze a still image.")
        captured = st.camera_input("Capture a frame to analyze", key="fallback_camera_capture")
        captured_bytes = captured.getvalue() if captured is not None else b""
        captured_digest = hashlib.sha256(captured_bytes).hexdigest() if captured_bytes else ""
        if captured is not None:
            analyze_captured_frame = st.button("Analyze captured frame", key="analyze_fallback_frame_button", type="primary")
            # Streamlit buttons rerun the app; retain one-shot analysis based on the button result.
            if analyze_captured_frame and captured_digest != st.session_state.get("fallback_camera_result_digest"):
                frame = cv2.imdecode(np.frombuffer(captured_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    try:
                        with st.spinner("Analyzing captured frame…"):
                            annotated, items = process_frame(frame, get_model(), water_db)
                        st.session_state["fallback_camera_result"] = {
                            "digest": captured_digest,
                            "annotated": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            "items": list(items or []),
                        }
                        st.session_state["fallback_camera_result_digest"] = captured_digest
                    except Exception as exc:
                        st.error("Camera frame analysis failed.")
                        with st.expander("Technical details"):
                            st.code(str(exc))
            fallback_result = st.session_state.get("fallback_camera_result")
            if fallback_result and fallback_result.get("digest") == captured_digest:
                frame_col, telemetry_col = st.columns([1.3, 1], gap="large")
                with frame_col:
                    st.image(fallback_result["annotated"], caption="Captured frame analysis", use_container_width=True)
                with telemetry_col:
                    render_detection_telemetry(fallback_result["items"], water_db)
    else:
        release_camera()
        st.markdown(
            '<div class="surface-card" style="text-align:center;padding:2rem;">'
            '<div style="font-size:2.2rem;">◎</div>'
            '<h4 style="margin:.6rem 0;">The scanner is ready</h4>'
            '<p style="color:#617985;margin:0;">Start the live scanner when your webcam is available. You can stop it at any time using the same control.</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# 10. Educational hub
# -----------------------------------------------------------------------------
st.divider()
st.markdown('<div class="section-kicker">Learn the science</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Understand the water behind products</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">A water footprint is an indicator of freshwater use and pollution associated with a product, person, community or business, including supply-chain stages.</p>',
    unsafe_allow_html=True,
)

edu_tab1, edu_tab2, edu_tab3, edu_tab4 = st.tabs(
    ["🌊 The three water pillars", "∑ Calculation framework", "↗ Product life cycle", "✓ Interpret & reduce"]
)

with edu_tab1:
    st.markdown(
        '<p class="section-copy">The components describe different kinds of pressure. They should be read separately as well as together: blue is linked to surface and groundwater, green to rainfall stored in soil, and grey to pollution assimilation.</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            '<div class="blue-card"><div class="pillar-title"><span class="pillar-dot pillar-blue"></span>Blue water</div>'
            '<div class="fact-label">Source</div><div class="fact-value">Rivers, lakes, reservoirs and groundwater aquifers</div>'
            '<div class="fact-label">What it measures</div><p>Fresh surface or groundwater consumed or incorporated during a process. Depending on the assessment, withdrawals and return flows also matter.</p>'
            '<div class="fact-label">Common examples</div><p>Irrigation pumped from a river or aquifer; water consumed during industrial processing.</p>'
            '<div class="fact-label">Why it matters</div><p style="margin-bottom:0;">A large blue footprint can be especially concerning where water is scarce or ecosystems depend on the same source.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="green-card"><div class="pillar-title"><span class="pillar-dot pillar-green"></span>Green water</div>'
            '<div class="fact-label">Source</div><div class="fact-value">Rainfall stored in the soil as moisture</div>'
            '<div class="fact-label">What it measures</div><p>Rainwater used by plants and returned to the atmosphere through evaporation and transpiration, or incorporated into biomass.</p>'
            '<div class="fact-label">Common examples</div><p>Rain-fed crops, pasture, timber and natural fibres grown using soil moisture.</p>'
            '<div class="fact-label">Why it matters</div><p style="margin-bottom:0;">It connects production to rainfall, soil health, land use and climate variability—not just irrigation infrastructure.</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="grey-card"><div class="pillar-title"><span class="pillar-dot pillar-grey"></span>Grey water</div>'
            '<div class="fact-label">Source</div><div class="fact-value">A calculated indicator of water-quality pressure</div>'
            '<div class="fact-label">What it measures</div><p>The theoretical volume of freshwater needed to assimilate a pollutant load while meeting the relevant water-quality standard.</p>'
            '<div class="fact-label">Common examples</div><p>Pollution associated with fertilizer runoff, pesticide residues, textile dyes or industrial effluents.</p>'
            '<div class="fact-label">Why it matters</div><p style="margin-bottom:0;">It is not simply the amount of wastewater discharged, nor a literal tank of water consumed; it expresses a dilution/quality requirement.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Compare the three components")
    comparison = pd.DataFrame(
        [
            {"Component": "Blue", "Primary connection": "Surface water / groundwater", "Main lens": "Freshwater consumption and scarcity", "Illustrative example": "Irrigation from an aquifer"},
            {"Component": "Green", "Primary connection": "Rainwater in soil", "Main lens": "Plant water use and rainfall", "Illustrative example": "Rain-fed cotton crop"},
            {"Component": "Grey", "Primary connection": "Water quality", "Main lens": "Pollutant load and quality standards", "Illustrative example": "Fertilizer runoff"},
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

with edu_tab2:
    formula_left, formula_right = st.columns(2, gap="large")
    with formula_left:
        with st.container(border=True):
            st.markdown("#### 1. Total footprint")
            st.latex(r"WF_{\text{total}} = WF_{\text{blue}} + WF_{\text{green}} + WF_{\text{grey}}")
            st.markdown("The total indicator combines the blue, green and grey components reported by the assessment.")
        st.markdown('<div style="height:.8rem;"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("#### 2. Blue and green water")
            st.latex(r"WF_{\text{blue/green}} = \frac{CWU_{\text{blue/green}} + W_{\text{incorporated}}}{Y}")
            st.caption("CWU = crop or process water use; Y = yield. The exact formulation depends on the assessment boundary and units.")
    with formula_right:
        with st.container(border=True):
            st.markdown("#### 3. Grey water")
            st.latex(r"WF_{\text{grey}} = \frac{L}{c_{\max} - c_{\text{nat}}}")
            st.markdown("The pollutant load is divided by the allowable concentration difference between the standard and natural background level.")
            st.latex(r"WF_{\text{grey}} = \frac{\alpha \times AR}{c_{\max} - c_{\text{nat}}}")
            st.caption("α = leaching/runoff fraction; AR = chemical application or emission rate.")

with edu_tab3:
    st.markdown("#### Where hidden water can accumulate")
    stages = [
        ("01", "Raw materials & agriculture", "Crop cultivation, evapotranspiration, irrigation, extraction and agricultural runoff."),
        ("02", "Processing & refining", "Washing, chemical treatment, smelting, polymerization and component fabrication."),
        ("03", "Manufacturing & assembly", "Cooling, steam generation, cleaning, dyeing, finishing and wastewater management."),
        ("04", "Packaging & distribution", "Packaging materials and the energy and logistics required to move products to consumers."),
    ]
    for number, title, description in stages:
        with st.container(border=True):
            number_col, text_col = st.columns([0.55, 5], vertical_alignment="center")
            number_col.markdown(
                f'<div style="display:grid;place-items:center;width:2.5rem;height:2.5rem;border-radius:13px;background:#d9f7f0;color:#087e8b;font-weight:800;">{number}</div>',
                unsafe_allow_html=True,
            )
            text_col.markdown(f"**{title}**")
            text_col.caption(description)

with edu_tab4:
    action_left, action_right = st.columns(2, gap="large")
    with action_left:
        st.markdown("#### How to interpret a result")
        with st.container(border=True):
            st.markdown("**Compare like with like.**")
            st.write("Check whether values are reported per item, per kilogram, per serving or across a full life cycle before comparing products.")
            st.markdown("**Look at location and timing.**")
            st.write("A footprint does not communicate local water scarcity by itself. The same volume can have different consequences depending on basin conditions and season.")
            st.markdown("**Separate quantity from pollution.**")
            st.write("A higher grey-water estimate signals a larger calculated pollution-assimilation requirement; it is not a direct measurement of the volume of wastewater discharged.")
    with action_right:
        st.markdown("#### Practical ways to reduce pressure")
        with st.container(border=True):
            st.markdown("**Avoid avoidable waste.**")
            st.write("Food that is grown, processed and transported but never eaten carries a footprint without delivering its intended value.")
            st.markdown("**Use products for longer.**")
            st.write("Repairing, reusing, buying durable goods and considering refurbished options can reduce demand for new raw materials and manufacturing.")
            st.markdown("**Ask how it was produced.**")
            st.write("Efficient irrigation, soil stewardship, cleaner processing and wastewater treatment can influence different parts of the footprint.")
    with st.expander("What can change a footprint estimate?"):
        st.markdown(
            "- **Geography and season:** rainfall, water stress and source-water conditions differ by place and time.\n"
            "- **Production system:** rain-fed versus irrigated agriculture, crop yield, factory processes and energy systems affect results.\n"
            "- **Assessment boundary:** studies may include different stages of the supply chain or report different functional units.\n"
            "- **Database coverage:** AquaTrace can only return estimates supported by its local database; absent entries should not be treated as zero footprint."
        )

st.markdown(
    '<div style="margin-top:1.8rem;padding:1rem 1.2rem;border-radius:16px;border:1px solid #dcebef;background:rgba(255,255,255,.7);color:#6b818b;font-size:.82rem;line-height:1.65;">'
    '<b style="color:#34515f;">A note on interpretation</b><br>'
    'Water-footprint numbers are estimates that depend on geography, production methods, database coverage and assessment boundaries. A grey-water estimate is a water-quality indicator, not a literal wastewater volume. Use results to compare and learn, not as a substitute for a documented life-cycle assessment.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="text-align:center;margin-top:1.6rem;color:#78909a;font-size:.78rem;">'
    'AquaTrace · Make invisible water visible · Built for awareness and better decisions'
    '</div>',
    unsafe_allow_html=True,
)
