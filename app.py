import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Earthquake Predictor — QuakeSense",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model & features ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("xgb_tune.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_columns = load_artifacts()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #0a0f1e 0%, #0d1529 50%, #080d1a 100%);
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg,
        rgba(234,88,12,0.12) 0%,
        rgba(239,68,68,0.08) 50%,
        rgba(234,88,12,0.05) 100%);
    border: 1px solid rgba(234,88,12,0.25);
    border-radius: 22px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(234,88,12,0.07);
    filter: blur(50px);
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(239,68,68,0.05);
    filter: blur(40px);
}
.hero-badge {
    display: inline-block;
    background: rgba(234,88,12,0.15);
    border: 1px solid rgba(234,88,12,0.35);
    color: #fb923c;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #fb923c, #ef4444, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.97rem;
    color: #94a3b8;
    max-width: 600px;
    line-height: 1.7;
    margin: 0;
}

/* ── Section cards ── */
.sec-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-badge {
    background: rgba(234,88,12,0.12);
    border-radius: 6px;
    padding: 3px 9px;
    font-size: 0.72rem;
    color: #fb923c;
    font-family: 'DM Mono', monospace;
}

/* ── Inputs ── */
div[data-testid="stNumberInput"] > div > div > input,
div[data-testid="stSelectbox"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Sliders ── */
.stSlider > div > div > div { background: rgba(234,88,12,0.25) !important; }
.stSlider > div > div > div > div { background: #f97316 !important; }

/* ── Radio ── */
.stRadio > label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* ── Date/time inputs ── */
div[data-testid="stDateInput"] input,
div[data-testid="stTimeInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ea580c, #ef4444) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 2rem !important;
    border-radius: 12px !important;
    border: none !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(234,88,12,0.35) !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(234,88,12,0.5) !important;
}

/* ── Result cards ── */
.result-minor {
    background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(22,163,74,0.06));
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: popIn 0.5s cubic-bezier(0.175,0.885,0.32,1.275);
}
.result-light {
    background: linear-gradient(135deg, rgba(234,179,8,0.12), rgba(202,138,4,0.06));
    border: 1px solid rgba(234,179,8,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: popIn 0.5s cubic-bezier(0.175,0.885,0.32,1.275);
}
.result-moderate {
    background: linear-gradient(135deg, rgba(249,115,22,0.12), rgba(234,88,12,0.06));
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: popIn 0.5s cubic-bezier(0.175,0.885,0.32,1.275);
}
.result-strong {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: popIn 0.5s cubic-bezier(0.175,0.885,0.32,1.275);
}
.result-major {
    background: linear-gradient(135deg, rgba(139,0,0,0.2), rgba(185,28,28,0.1));
    border: 2px solid rgba(220,38,38,0.5);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: shake 0.5s ease-out, popIn 0.5s cubic-bezier(0.175,0.885,0.32,1.275);
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes shake {
    0%,100% { transform: translateX(0); }
    20%     { transform: translateX(-6px); }
    40%     { transform: translateX(6px); }
    60%     { transform: translateX(-4px); }
    80%     { transform: translateX(4px); }
}
.result-emoji { font-size: 4rem; margin-bottom: 0.5rem; display: block; }
.result-mag {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.3rem 0;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0.3rem 0 0.8rem;
}
.result-desc {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.7;
    max-width: 500px;
    margin: 0 auto;
}

/* ── Richter scale bar ── */
.scale-wrap { margin: 1.5rem 0 0.5rem; }
.scale-label { font-size: 0.78rem; color: #64748b; margin-bottom: 8px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.scale-track {
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(90deg,
        #22c55e 0%,
        #84cc16 20%,
        #eab308 35%,
        #f97316 55%,
        #ef4444 75%,
        #991b1b 100%);
    position: relative;
    margin-bottom: 6px;
}
.scale-marker {
    position: absolute;
    top: -4px;
    width: 20px; height: 20px;
    border-radius: 50%;
    background: white;
    border: 3px solid #1e293b;
    transform: translateX(-50%);
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    transition: left 1s ease;
}
.scale-ticks {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #475569;
    font-family: 'DM Mono', monospace;
    padding: 0 2px;
}

/* ── Info cards ── */
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 1rem; }
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
}
.info-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #fb923c;
}
.info-lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(234,179,8,0.07);
    border: 1px solid rgba(234,179,8,0.2);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    font-size: 0.8rem;
    color: #fbbf24;
    line-height: 1.6;
    margin-top: 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,15,30,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #64748b; font-size: 0.85rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(234,88,12,0.3); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Magnitude classification ──────────────────────────────────────────────────
def classify_magnitude(mag):
    if mag < 2.5:
        return {
            "cls": "minor", "label": "Minor",
            "emoji": "🟢", "color": "#4ade80",
            "desc": "Micro/minor earthquake. Generally not felt by people. Detected only by seismographs. No damage expected.",
            "pct": max(5, mag / 10 * 100)
        }
    elif mag < 4.0:
        return {
            "cls": "light", "label": "Light",
            "emoji": "🟡", "color": "#facc15",
            "desc": "Often felt but rarely causes damage. Some rattling of dishes, windows, and doors. No structural damage.",
            "pct": mag / 10 * 100
        }
    elif mag < 5.0:
        return {
            "cls": "moderate", "label": "Moderate",
            "emoji": "🟠", "color": "#fb923c",
            "desc": "Felt by all. Slight damage to poorly constructed buildings. Minor cracking of plaster.",
            "pct": mag / 10 * 100
        }
    elif mag < 6.0:
        return {
            "cls": "strong", "label": "Strong",
            "emoji": "🔴", "color": "#f87171",
            "desc": "Damage to many buildings. Poorly constructed buildings destroyed. Heavy furniture overturned.",
            "pct": mag / 10 * 100
        }
    else:
        return {
            "cls": "major", "label": "Major / Great",
            "emoji": "🚨", "color": "#ef4444",
            "desc": "MAJOR EARTHQUAKE. Serious damage over large areas. Many buildings destroyed. Casualties expected.",
            "pct": min(98, mag / 10 * 100)
        }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 1.5rem;'>
      <div style='font-size:2.8rem'>🌍</div>
      <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                  background:linear-gradient(90deg,#fb923c,#ef4444);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        QuakeSense
      </div>
      <div style='font-size:0.72rem;color:#475569;margin-top:4px;letter-spacing:1.5px;text-transform:uppercase;'>
        Earthquake Predictor
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    <p>An <strong style='color:#fb923c'>XGBoost Regressor</strong> trained on Nepal earthquake 
    data (1990–2026) to predict earthquake magnitude from seismic and geographic parameters.</p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Model Info")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Algorithm", "XGBoost")
        st.metric("Features", "19")
    with c2:
        st.metric("Task", "Regression")
        st.metric("Tuning", "RandomCV")

    st.markdown("---")
    st.markdown("### 🗺️ Richter Scale Guide")
    scale_items = [
        ("< 2.5", "🟢", "Micro"),
        ("2.5–4.0", "🟡", "Light"),
        ("4.0–5.0", "🟠", "Moderate"),
        ("5.0–6.0", "🔴", "Strong"),
        ("6.0 +", "🚨", "Major"),
    ]
    for mag_r, emoji, label in scale_items:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:4px 0;"
            f"font-size:0.82rem;color:#94a3b8'>"
            f"<span>{emoji}</span><span style='flex:1'>{label}</span>"
            f"<span style='font-family:DM Mono,monospace;color:#475569'>{mag_r}</span></div>",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='font-size:0.72rem;color:#334155;'>
    ⚠️ For research & educational purposes only. Not a real-time early warning system.
    </p>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🌏 Nepal Region · 1990–2026 · XGBoost</div>
  <div class="hero-title">QuakeSense</div>
  <p class="hero-sub">
    Predict earthquake magnitude from seismic parameters and geographic data.
    Enter the coordinates, depth, and station information below to get an
    instant magnitude estimate from our trained XGBoost model.
  </p>
</div>
""", unsafe_allow_html=True)


# ── SECTION 1: Date & Time ────────────────────────────────────────────────────
st.markdown("""
<div class="sec-card">
  <div class="sec-title">📅 Date & Time <span class="sec-badge">Temporal Features</span></div>
</div>
""", unsafe_allow_html=True)

now = datetime.now()
tc1, tc2, tc3, tc4 = st.columns(4)
with tc1:
    year  = st.number_input("Year",  min_value=1990, max_value=2030, value=now.year)
with tc2:
    month = st.number_input("Month", min_value=1,    max_value=12,   value=now.month)
with tc3:
    day   = st.number_input("Day",   min_value=1,    max_value=31,   value=now.day)
with tc4:
    hour  = st.number_input("Hour",  min_value=0,    max_value=23,   value=now.hour)


# ── SECTION 2: Geographic Location ───────────────────────────────────────────
st.markdown("""
<div class="sec-card">
  <div class="sec-title">🗺️ Geographic Location <span class="sec-badge">Coordinates</span></div>
</div>
""", unsafe_allow_html=True)

gc1, gc2, gc3 = st.columns(3)
with gc1:
    latitude  = st.number_input("Latitude",  min_value=20.0, max_value=40.0, value=27.7,  step=0.01, format="%.4f")
with gc2:
    longitude = st.number_input("Longitude", min_value=75.0, max_value=100.0, value=85.3, step=0.01, format="%.4f")
with gc3:
    depth     = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=10.0, step=0.1)

# Country & Location source
cc1, cc2 = st.columns(2)
with cc1:
    country = st.selectbox("Country / Region", [
        "Nepal", "China", "India",
        "Southern Tibetan Plateau", "western Xizang"
    ])
with cc2:
    location_source_us = st.radio(
        "Location Source",
        ["USGS (us)", "Other"],
        horizontal=True
    )


# ── SECTION 3: Seismic Parameters ────────────────────────────────────────────
st.markdown("""
<div class="sec-card">
  <div class="sec-title">📡 Seismic Station Parameters <span class="sec-badge">Sensor Data</span></div>
</div>
""", unsafe_allow_html=True)

sc1, sc2 = st.columns(2)
with sc1:
    st.markdown("<p style='color:#64748b;font-size:0.82rem;'>Number of stations that reported this event</p>", unsafe_allow_html=True)
    nst = st.slider("NST — Station Count", min_value=0, max_value=200, value=25)

    st.markdown("<p style='color:#64748b;font-size:0.82rem;margin-top:1rem;'>Largest azimuthal gap (degrees)</p>", unsafe_allow_html=True)
    gap = st.slider("GAP — Azimuthal Gap (°)", min_value=0.0, max_value=360.0, value=100.0, step=0.5)

with sc2:
    st.markdown("<p style='color:#64748b;font-size:0.82rem;'>Root mean square of travel-time residuals</p>", unsafe_allow_html=True)
    rms = st.slider("RMS — Travel Time Residual", min_value=0.0, max_value=5.0, value=0.8, step=0.01)

    depth_category = st.selectbox("Depth Category", [
        "Shallow (0–70 km)",
        "Intermediate (70–300 km)",
        "Deep (300–700 km)"
    ])


# ── SECTION 4: Error Measurements ────────────────────────────────────────────
st.markdown("""
<div class="sec-card">
  <div class="sec-title">📏 Error Measurements <span class="sec-badge">Uncertainty</span></div>
</div>
""", unsafe_allow_html=True)

ec1, ec2 = st.columns(2)
with ec1:
    horizontalError = st.number_input(
        "Horizontal Error (km)",
        min_value=0.0, max_value=100.0, value=5.0, step=0.1,
        help="Uncertainty of reported location of the event in horizontal direction"
    )
with ec2:
    depthError = st.number_input(
        "Depth Error (km)",
        min_value=0.0, max_value=100.0, value=3.0, step=0.1,
        help="Uncertainty of reported depth of the event"
    )

st.markdown("<div style='margin:1.5rem 0 1rem'></div>", unsafe_allow_html=True)

# ── Predict Button ─────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Earthquake Magnitude", use_container_width=True)

if predict_clicked:
    with st.spinner("Analyzing seismic parameters..."):
        time.sleep(1.0)

    # ── Build feature vector ──────────────────────────────────────────────────
    # Match exactly: feature_columns.pkl order
    # ['latitude','longitude','depth','nst','gap','rms','horizontalError',
    #  'depthError','year','month','day','hour',
    #  'locationSource_us',
    #  'country_China','country_India','country_Nepal',
    #  'country_Southern Tibetan Plateau','country_western Xizang',
    #  'depth_category_Intermeiate']

    loc_src_us = 1 if location_source_us == "USGS (us)" else 0

    country_china    = 1 if country == "China" else 0
    country_india    = 1 if country == "India" else 0
    country_nepal    = 1 if country == "Nepal" else 0
    country_stp      = 1 if country == "Southern Tibetan Plateau" else 0
    country_wxizang  = 1 if country == "western Xizang" else 0

    # depth_category_Intermeiate (note: original has typo 'Intermeiate')
    depth_cat_intermediate = 1 if "Intermediate" in depth_category else 0

    features = [
        latitude, longitude, depth,
        nst, gap, rms,
        horizontalError, depthError,
        year, month, day, hour,
        loc_src_us,
        country_china, country_india, country_nepal,
        country_stp, country_wxizang,
        depth_cat_intermediate
    ]

    X_input = np.array(features).reshape(1, -1)
    magnitude = float(model.predict(X_input)[0])
    magnitude = round(magnitude, 2)

    info = classify_magnitude(magnitude)

    st.markdown("<div style='margin:1.5rem 0 0.5rem'></div>", unsafe_allow_html=True)

    # ── Result card ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-{info['cls']}">
      <span class="result-emoji">{info['emoji']}</span>
      <div class="result-mag" style="color:{info['color']}">{magnitude}</div>
      <div class="result-label" style="color:{info['color']}">{info['label']} Earthquake</div>
      <p class="result-desc">{info['desc']}</p>

      <div class="scale-wrap" style="max-width:480px;margin:1.5rem auto 0;">
        <div class="scale-label">Richter Scale Position</div>
        <div class="scale-track">
          <div class="scale-marker" style="left:{min(info['pct'],96)}%"></div>
        </div>
        <div class="scale-ticks">
          <span>0</span><span>2</span><span>4</span><span>5</span><span>6</span><span>8</span><span>10</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input summary ─────────────────────────────────────────────────────────
    st.markdown("<div style='margin:1.5rem 0 0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#475569;font-size:0.78rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;'>Input Summary</p>", unsafe_allow_html=True)

    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    summary = [
        (sc1, f"{latitude}°N", "Latitude"),
        (sc2, f"{longitude}°E", "Longitude"),
        (sc3, f"{depth} km", "Depth"),
        (sc4, f"{nst}", "Stations"),
        (sc5, f"{gap}°", "Gap"),
        (sc6, country, "Country"),
    ]
    for col, val, lbl in summary:
        with col:
            st.markdown(f"""
            <div class="info-card">
              <div class="info-val">{val}</div>
              <div class="info-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <strong>Research Tool Only:</strong> This model was trained on historical earthquake data
      from Nepal region (1990–2026). Predictions are for educational purposes and should
      NOT be used as a real-time early warning or emergency system.
    </div>
    """, unsafe_allow_html=True)