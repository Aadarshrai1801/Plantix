"""
Plantix — AI Plant Disease Detector
A Streamlit front-end for the CNN model trained in notebook/01_training.ipynb,
styled as a dark, glass-panelled diagnostic console ("Spectral Lab"):
bioluminescent green for healthy reads, UV violet for the scanning/technical
layer, and a warning coral for flagged specimens.

Layout: a top status/stat strip gives at-a-glance context, a segmented tab
bar replaces the old sidebar so the canvas runs full-width, and each page
leads with the thing the visitor actually came to do.
"""

import hashlib
import io
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.disease_info import CLASS_NAMES, DISEASE_INFO

# --------------------------------------------------------------------------------------
# App config
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).parent
MODEL_PATH_H5 = ROOT / "models" / "trained_model.h5"
MODEL_PATH_KERAS = ROOT / "models" / "trained_model.keras"
HISTORY_PATH = ROOT / "history" / "training_history.json"
IMAGE_SIZE = (128, 128)

CHART_FONT = dict(family="Inter, sans-serif", color="#F2F8F3")
CHART_GRID = "rgba(255,255,255,0.10)"
COLOR_GREEN = "#6FFFB0"
COLOR_GREEN_DEEP = "#35C98A"
COLOR_VIOLET = "#B586FF"
COLOR_RED = "#FF6B81"
COLOR_AMBER = "#FFC65C"

st.set_page_config(
    page_title="Plantix | Spectral Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------------------
# Design tokens & global styling
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg: #07090A;
            --bg-elevated: #12171A;
            --glass: rgba(255,255,255,0.055);
            --glass-hover: rgba(255,255,255,0.09);
            --border: rgba(255,255,255,0.12);
            --border-hover: rgba(111,255,176,0.5);
            --ink: #F2F8F3;
            --ink-soft: #C5D6C8;
            --ink-faint: #8AA091;
            --green: #6FFFB0;
            --green-deep: #35C98A;
            --violet: #B586FF;
            --red: #FF6B81;
            --amber: #FFC65C;
            --shadow: 0 20px 60px -24px rgba(0,0,0,0.8), 0 1px 0 rgba(255,255,255,0.06) inset;
            --radius: 18px;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink) !important; }
        #MainMenu, footer, header {visibility: hidden;}

        .stApp {
            background:
                radial-gradient(circle at 12% -10%, rgba(111,255,176,0.15), transparent 42%),
                radial-gradient(circle at 92% 4%, rgba(181,134,255,0.13), transparent 38%),
                radial-gradient(circle at 50% 115%, rgba(111,255,176,0.06), transparent 45%),
                var(--bg);
            background-attachment: fixed;
        }

        h1, h2, h3, h4, .display { font-family: 'Space Grotesk', sans-serif; color: var(--ink); letter-spacing: -0.01em; }
        .mono { font-family: 'JetBrains Mono', monospace; }

        /* ---------- Force legibility across every native Streamlit text element ---------- */
        p, span, li, label, .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong,
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
        [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"],
        .stTextInput label, .stSelectbox label, .stSlider label, .stFileUploader label,
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        div[data-testid="stExpander"] summary, div[data-testid="stExpander"] p,
        .stAlert p, .stAlert div { color: var(--ink) !important; }
        [data-testid="stCaptionContainer"], .stCaption, small { color: var(--ink-faint) !important; }

        /* ---------- Glass base ---------- */
        .glass {
            background: var(--glass); border: 1px solid var(--border); border-radius: var(--radius);
            backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); box-shadow: var(--shadow);
        }

        /* ---------- Top bar ---------- */
        .topbar {
            display: flex; align-items: center; justify-content: space-between;
            padding: 1rem 1.6rem; margin: -1rem -1rem 1rem -1rem;
            background: rgba(255,255,255,0.045);
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
        }
        .topbar-left { display: flex; align-items: center; gap: 0.9rem; }
        .logo-mark {
            width: 42px; height: 42px; border-radius: 12px;
            background: linear-gradient(140deg, var(--green) 0%, var(--violet) 100%);
            color: #05140D;
            display: flex; align-items: center; justify-content: center;
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.2rem;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.2) inset, 0 0 24px -4px rgba(111,255,176,0.7);
        }
        .wordmark { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.35rem; line-height: 1; letter-spacing: -0.01em; color: var(--ink); }
        .tagline { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.1em; color: var(--ink-faint); text-transform: uppercase; margin-top: 3px; }
        .status-chip { display: flex; align-items: center; gap: 0.55rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.06em; color: var(--ink-soft); text-transform: uppercase; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .status-dot.on { background: var(--green); box-shadow: 0 0 0 3px rgba(111,255,176,0.2), 0 0 10px 1px var(--green); animation: pulse 2.2s ease-in-out infinite; }
        .status-dot.off { background: var(--red); box-shadow: 0 0 0 3px rgba(255,107,129,0.2); }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.45; } }

        /* ---------- Stat strip ---------- */
        .stat-strip { display: flex; gap: 0.8rem; margin-bottom: 1.6rem; flex-wrap: wrap; }
        .stat-chip {
            flex: 1; min-width: 150px;
            background: var(--glass); border: 1px solid var(--border); border-radius: 14px;
            padding: 0.75rem 1rem; backdrop-filter: blur(14px);
        }
        .stat-chip-label { font-family: 'JetBrains Mono', monospace; font-size: 0.64rem; letter-spacing: 0.08em; color: var(--ink-faint); text-transform: uppercase; }
        .stat-chip-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3rem; color: var(--ink); margin-top: 0.15rem; }
        .stat-chip-value.accent { color: var(--green); }

        /* ---------- Segmented tab nav ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem; background: var(--glass); border: 1px solid var(--border);
            border-radius: 14px; padding: 0.35rem; backdrop-filter: blur(18px);
            margin-bottom: 1.8rem; width: fit-content;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent; border: none; border-radius: 10px;
            padding: 0.55rem 1.3rem; color: var(--ink-soft) !important; font-weight: 500;
            transition: all 0.15s ease;
        }
        .stTabs [data-baseweb="tab"] p { color: inherit !important; font-weight: inherit; }
        .stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,0.06); }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(111,255,176,0.16), rgba(181,134,255,0.12)) !important;
            color: var(--green) !important; font-weight: 600;
            box-shadow: 0 0 0 1px rgba(111,255,176,0.3) inset;
        }
        .stTabs [aria-selected="true"] p { color: var(--green) !important; font-weight: 600; }
        .stTabs [data-baseweb="tab-highlight"] { display: none; }
        .stTabs [data-baseweb="tab-border"] { display: none; }

        /* ---------- Section header ---------- */
        .section-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.14em; color: var(--green); text-transform: uppercase; margin-bottom: 0.35rem; }
        .section-title { font-size: 1.9rem; font-weight: 700; margin: 0 0 0.3rem 0; font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
        .section-sub { color: var(--ink-soft); font-size: 1rem; margin-bottom: 1.5rem; max-width: 640px; line-height: 1.55; }

        /* ---------- Generic card ---------- */
        .card {
            background: var(--glass); border: 1px solid var(--border); border-radius: var(--radius);
            padding: 1.4rem 1.6rem; margin-bottom: 1rem; box-shadow: var(--shadow);
            backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); color: var(--ink);
            transition: border-color 0.18s ease, background 0.18s ease;
        }
        .card:hover { border-color: var(--border-hover); background: var(--glass-hover); }
        .card h4 { margin-top: 0; }
        .card ul { color: var(--ink-soft); }

        /* ---------- Specimen / diagnostic readout card ---------- */
        .specimen-card {
            position: relative; overflow: hidden;
            background: var(--glass); border: 1px solid var(--border); border-radius: 20px;
            padding: 1.7rem 1.9rem 1.5rem 1.9rem; box-shadow: var(--shadow); margin-bottom: 1.1rem;
            backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
        }
        .specimen-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
        .specimen-card.ok::before { background: linear-gradient(90deg, var(--green), var(--green-deep)); box-shadow: 0 0 16px 0 var(--green); }
        .specimen-card.flag::before { background: linear-gradient(90deg, var(--red), var(--amber)); box-shadow: 0 0 16px 0 var(--red); }
        .specimen-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
        .specimen-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.12em; color: var(--ink-faint); text-transform: uppercase; }
        .specimen-id { font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 600; color: var(--ink); letter-spacing: 0.02em; }
        .specimen-title { font-size: 1.5rem; font-weight: 700; margin: 0.75rem 0 0.4rem 0; font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
        .specimen-desc { color: var(--ink-soft); font-size: 0.96rem; line-height: 1.55; margin: 0; }

        .status-badge {
            flex-shrink: 0; padding: 0.45rem 0.95rem; border-radius: 999px;
            font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.68rem;
            letter-spacing: 0.08em; text-transform: uppercase; white-space: nowrap; border: 1px solid currentColor;
        }
        .status-badge.ok { color: var(--green); box-shadow: 0 0 18px -4px var(--green); }
        .status-badge.flag { color: var(--red); box-shadow: 0 0 18px -4px var(--red); }

        /* ---------- Confidence bar ---------- */
        .confidence-row { display: flex; align-items: center; gap: 0.9rem; margin-top: 1.1rem; }
        .confidence-track { flex: 1; height: 8px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; }
        .confidence-fill { height: 100%; border-radius: 999px; }
        .confidence-fill.ok { background: linear-gradient(90deg, var(--green-deep), var(--green)); box-shadow: 0 0 10px 0 var(--green); }
        .confidence-fill.flag { background: linear-gradient(90deg, var(--amber), var(--red)); box-shadow: 0 0 10px 0 var(--red); }
        .confidence-label { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.95rem; color: var(--ink); min-width: 52px; text-align: right; }

        .scan-divider { position: relative; height: 1px; margin: 1.3rem 0; background: linear-gradient(90deg, transparent, var(--border) 12%, var(--border) 88%, transparent); }
        .scan-divider::after {
            content: ""; position: absolute; top: -1px; left: 0; height: 2px; width: 40%;
            background: linear-gradient(90deg, var(--green), transparent);
            animation: sweep 3.2s ease-in-out infinite; filter: drop-shadow(0 0 6px var(--green));
        }
        @keyframes sweep { 0% { left: -40%; } 100% { left: 100%; } }

        /* ---------- Ledger (metrics as rows) ---------- */
        .ledger { display: flex; flex-direction: column; gap: 0; }
        .ledger-row { display: flex; justify-content: space-between; align-items: baseline; padding: 0.6rem 0; border-bottom: 1px solid var(--border); }
        .ledger-row:last-child { border-bottom: none; }
        .ledger-label { font-size: 0.82rem; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.05em; }
        .ledger-value { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 1.05rem; color: var(--ink); }

        /* ---------- Metric grid (Performance Log) ---------- */
        .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-bottom: 1.2rem; }
        .metric-card { background: var(--glass); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem; backdrop-filter: blur(14px); }
        .metric-card-label { font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; letter-spacing: 0.08em; color: var(--ink-faint); text-transform: uppercase; }
        .metric-card-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.6rem; color: var(--ink); margin-top: 0.2rem; }
        @media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }

        /* ---------- Tag chips (disease library) ---------- */
        .tag { display: inline-block; padding: 0.24rem 0.8rem; border-radius: 999px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; border: 1px solid; }
        .tag-healthy { background: rgba(111,255,176,0.12); color: var(--green); border-color: rgba(111,255,176,0.4); }
        .tag-disease { background: rgba(255,107,129,0.12); color: var(--red); border-color: rgba(255,107,129,0.4); }
        .severity-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-faint); }

        .index-card { background: var(--glass); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.7rem; box-shadow: var(--shadow); backdrop-filter: blur(14px); transition: border-color 0.18s ease; }
        .index-card:hover { border-color: var(--border-hover); }
        .index-card-head { display: flex; justify-content: space-between; align-items: center; gap: 0.8rem; }
        .index-crop { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem; color: var(--ink); }

        /* ---------- Buttons ---------- */
        .stButton>button {
            border-radius: 12px; border: 1px solid transparent;
            background: linear-gradient(135deg, var(--green) 0%, var(--violet) 130%);
            color: #05140D !important; font-weight: 700;
            padding: 0.6rem 1.4rem; box-shadow: 0 0 0 1px rgba(255,255,255,0.12) inset, 0 12px 30px -12px rgba(111,255,176,0.55);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton>button p { color: #05140D !important; font-weight: 700; }
        .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 0 0 1px rgba(255,255,255,0.18) inset, 0 16px 34px -10px rgba(111,255,176,0.7); }
        .stButton>button:disabled { background: rgba(255,255,255,0.07); color: var(--ink-faint) !important; box-shadow: none; }
        .stButton>button:disabled p { color: var(--ink-faint) !important; }

        .stDownloadButton>button {
            border-radius: 12px; border: 1px solid var(--border); background: var(--glass);
            color: var(--ink) !important; font-weight: 600; backdrop-filter: blur(14px);
        }
        .stDownloadButton>button p { color: var(--ink) !important; }
        .stDownloadButton>button:hover { border-color: var(--border-hover); color: var(--green) !important; }
        .stDownloadButton>button:hover p { color: var(--green) !important; }

        /* ---------- Inputs / misc widget overrides ---------- */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
            background: var(--glass) !important; border: 1px solid var(--border) !important;
            border-radius: 10px !important; color: var(--ink) !important;
        }
        .stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--green) !important; box-shadow: 0 0 10px 1px var(--green); }
        [data-testid="stFileUploaderDropzone"], .stCameraInput {
            border-radius: 14px; background: var(--glass) !important; border: 1px dashed var(--border) !important;
        }
        div[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 14px; background: var(--glass); backdrop-filter: blur(14px); }

        .footnote { color: var(--ink-faint) !important; font-size: 0.8rem; }
        hr { border-color: var(--border) !important; }

        [data-testid="stMetricValue"] { color: var(--ink); font-family: 'JetBrains Mono', monospace; }
        [data-testid="stMetricLabel"] { color: var(--ink-soft); }

        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 8px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(111,255,176,0.4); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained CNN model…")
def load_model():
    """Try both saved model files; some environments save `.keras` files that
    are actually legacy HDF5 under the hood, which the loader rejects by
    extension, so we fall back to the `.h5` copy on failure."""
    errors = []
    for path in (MODEL_PATH_H5, MODEL_PATH_KERAS):
        if not path.exists():
            continue
        try:
            return tf.keras.models.load_model(str(path))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{path.name}: {e}")
    if errors:
        st.session_state["_model_load_errors"] = errors
    return None


@st.cache_data(show_spinner=False)
def load_history():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return None


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Match the exact preprocessing used in 02_testing.ipynb (no rescaling)."""
    img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def predict(model, pil_img: Image.Image):
    input_arr = preprocess_image(pil_img)
    preds = model.predict(input_arr, verbose=0)[0]
    order = np.argsort(preds)[::-1]
    return preds, order


def specimen_id(class_name: str, salt: str = "") -> str:
    """Deterministic-looking specimen code for the report card."""
    digest = hashlib.md5(f"{class_name}{salt}".encode()).hexdigest()[:6].upper()
    crop_code = "".join(c for c in class_name if c.isupper())[:3] or "PLX"
    return f"PLX-{crop_code}-{digest}"


def ledger_html(rows: list) -> str:
    body = "".join(
        f'<div class="ledger-row"><span class="ledger-label">{label}</span>'
        f'<span class="ledger-value">{value}</span></div>'
        for label, value in rows
    )
    return f'<div class="ledger">{body}</div>'


# --------------------------------------------------------------------------------------
# Top bar
# --------------------------------------------------------------------------------------
model = load_model()
history = load_history()

status_class = "on" if model is not None else "off"
status_text = "MODEL ONLINE" if model is not None else "MODEL OFFLINE"

st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar-left">
            <div class="logo-mark">P</div>
            <div>
                <div class="wordmark">Plantix</div>
                <div class="tagline">Spectral Plant Diagnostics</div>
            </div>
        </div>
        <div class="status-chip"><span class="status-dot {status_class}"></span>{status_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Stat strip — at-a-glance context, replaces the old sidebar footnote
# --------------------------------------------------------------------------------------
crops_covered = len({v["crop"] for v in DISEASE_INFO.values()})
val_acc_display = f"{history['val_accuracy'][-1]*100:.1f}%" if history else "—"

st.markdown(
    f"""
    <div class="stat-strip">
        <div class="stat-chip">
            <div class="stat-chip-label">Trained classes</div>
            <div class="stat-chip-value">{len(CLASS_NAMES)}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">Crop species</div>
            <div class="stat-chip-value">{crops_covered}</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">Input resolution</div>
            <div class="stat-chip-value">128×128</div>
        </div>
        <div class="stat-chip">
            <div class="stat-chip-label">Validation accuracy</div>
            <div class="stat-chip-value accent">{val_acc_display}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "history_log" not in st.session_state:
    st.session_state.history_log = []

# --------------------------------------------------------------------------------------
# Top navigation — segmented tab bar (replaces the old sidebar radio)
# --------------------------------------------------------------------------------------
tab_inspect, tab_perf, tab_index, tab_about = st.tabs(
    ["🔬  Inspection Bay", "📈  Performance Log", "🗂️  Specimen Index", "🧾  Field Notes"]
)

# --------------------------------------------------------------------------------------
# PAGE: Inspection Bay
# --------------------------------------------------------------------------------------
with tab_inspect:
    st.markdown(
        """
        <div class="section-eyebrow">LIVE DIAGNOSIS</div>
        <div class="section-title">Inspection Bay</div>
        <div class="section-sub">Submit a leaf photo for automated screening. The model returns a
        diagnostic readout with condition, confidence, and field guidance.</div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "No trained model could be loaded. Place `trained_model.h5` (or "
            "`trained_model.keras`) inside the `models/` folder."
        )
        if st.session_state.get("_model_load_errors"):
            with st.expander("Show load errors"):
                for err in st.session_state["_model_load_errors"]:
                    st.code(err)
        st.stop()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown("<div class='section-eyebrow'>STEP 1 · SUBMIT SAMPLE</div>", unsafe_allow_html=True)
        tab_upload, tab_camera = st.tabs(["📁  Upload", "📷  Camera"])
        image_source = None
        with tab_upload:
            uploaded = st.file_uploader(
                "Upload a leaf photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
            )
            if uploaded is not None:
                image_source = Image.open(uploaded)
        with tab_camera:
            snapped = st.camera_input("Take a photo", label_visibility="collapsed")
            if snapped is not None:
                image_source = Image.open(snapped)

        if image_source is not None:
            st.image(image_source, caption="Sample under review", use_column_width=True)

        with st.expander("⚙️  Advanced options"):
            top_k = st.slider("Show top-K predictions", 3, 10, 5)
            confidence_threshold = st.slider("Low-confidence warning threshold (%)", 10, 90, 50)

        run = st.button("Run inspection →", use_container_width=True, disabled=image_source is None)

    with right:
        st.markdown("<div class='section-eyebrow'>STEP 2 · READOUT</div>", unsafe_allow_html=True)
        if not run and image_source is None:
            st.markdown(
                "<div class='card footnote'>Awaiting sample. Upload or capture a leaf photo, "
                "then click <b>Run inspection</b> to generate a readout.</div>",
                unsafe_allow_html=True,
            )
        elif run and image_source is not None:
            with st.spinner("Running inference through the CNN…"):
                start = time.time()
                preds, order = predict(model, image_source)
                elapsed = time.time() - start

            top_idx = order[0]
            top_class = CLASS_NAMES[top_idx]
            top_conf = float(preds[top_idx]) * 100
            info = DISEASE_INFO[top_class]

            state_class = "ok" if info["healthy"] else "flag"
            badge_text = "PASSED" if info["healthy"] else "FLAGGED"
            sid = specimen_id(top_class, datetime.now().strftime("%Y%m%d"))

            st.markdown(
                f"""
                <div class="specimen-card {state_class}">
                    <div class="specimen-head">
                        <div>
                            <div class="specimen-eyebrow">DIAGNOSTIC READOUT</div>
                            <div class="specimen-id mono">{sid}</div>
                        </div>
                        <div class="status-badge {state_class}">{badge_text}</div>
                    </div>
                    <div class="specimen-title">{info['crop']} — {info['condition']}</div>
                    <p class="specimen-desc">{info['description']}</p>
                    <div class="confidence-row">
                        <div class="confidence-track"><div class="confidence-fill {state_class}" style="width:{top_conf:.1f}%;"></div></div>
                        <div class="confidence-label mono">{top_conf:.1f}%</div>
                    </div>
                    <div class="scan-divider"></div>
                    {ledger_html([
                        ("Severity", info['severity']),
                        ("Inference time", f"{elapsed*1000:.0f} ms"),
                    ])}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if top_conf < confidence_threshold:
                st.warning(
                    f"Confidence is below your {confidence_threshold}% threshold. "
                    "Try a clearer, well-lit, close-up photo of a single leaf."
                )

            st.markdown("<div class='section-eyebrow' style='margin-top:1.2rem;'>TOP CANDIDATES</div>", unsafe_allow_html=True)
            top_df = pd.DataFrame(
                {
                    "Class": [CLASS_NAMES[i] for i in order[:top_k]],
                    "Confidence (%)": [float(preds[i]) * 100 for i in order[:top_k]],
                }
            ).sort_values("Confidence (%)")
            fig = go.Figure(
                go.Bar(
                    x=top_df["Confidence (%)"], y=top_df["Class"], orientation="h",
                    marker=dict(
                        color=top_df["Confidence (%)"],
                        colorscale=[[0, "rgba(111,255,176,0.2)"], [1, COLOR_GREEN]],
                        line=dict(color=COLOR_GREEN_DEEP, width=0.5),
                    ),
                    text=top_df["Confidence (%)"].map(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                    textfont=dict(color="#F2F8F3"),
                )
            )
            fig.update_layout(
                height=320, margin=dict(l=0, r=10, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=CHART_FONT, yaxis_title=None, xaxis_title=None,
                xaxis=dict(gridcolor=CHART_GRID, color="#C5D6C8"),
                yaxis=dict(gridcolor=CHART_GRID, color="#C5D6C8"),
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🩺 Symptoms" if not info["healthy"] else "##### 🌱 Observations")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")
            with c2:
                st.markdown("##### 🌱 Field guidance")
                for t in info["treatment"]:
                    st.markdown(f"- {t}")

            report = (
                f"Plantix Diagnostic Readout — {sid}\n"
                f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n"
                f"Predicted class : {top_class}\n"
                f"Crop            : {info['crop']}\n"
                f"Condition       : {info['condition']}\n"
                f"Confidence      : {top_conf:.2f}%\n"
                f"Severity        : {info['severity']}\n\n"
                f"Description: {info['description']}\n\n"
                f"Top-{top_k} predictions:\n"
                + "\n".join(f"  {r.Class}: {r._2:.2f}%" for r in top_df.sort_values("Confidence (%)", ascending=False).itertuples())
            )
            st.download_button(
                "⬇️  Download report (.txt)", data=report,
                file_name=f"{sid}.txt",
                use_container_width=True,
            )

            st.session_state.history_log.insert(
                0,
                {"id": sid, "time": datetime.now().strftime("%H:%M:%S"), "class": top_class, "confidence": f"{top_conf:.1f}%"},
            )

    if st.session_state.history_log:
        st.divider()
        st.markdown("<div class='section-eyebrow'>SESSION LOG</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history_log), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
# PAGE: Performance Log
# --------------------------------------------------------------------------------------
with tab_perf:
    st.markdown(
        """
        <div class="section-eyebrow">TRAINING RECORD</div>
        <div class="section-title">Performance Log</div>
        <div class="section-sub">How the CNN behind Plantix learned — 5 convolutional blocks,
        trained for 10 epochs on ~88K labeled leaf images.</div>
        """,
        unsafe_allow_html=True,
    )

    if history is None:
        st.warning("`history/training_history.json` not found.")
    else:
        final_acc = history["accuracy"][-1] * 100
        final_val_acc = history["val_accuracy"][-1] * 100
        final_loss = history["loss"][-1]
        final_val_loss = history["val_loss"][-1]

        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-card"><div class="metric-card-label">Train accuracy</div><div class="metric-card-value">{final_acc:.2f}%</div></div>
                <div class="metric-card"><div class="metric-card-label">Validation accuracy</div><div class="metric-card-value">{final_val_acc:.2f}%</div></div>
                <div class="metric-card"><div class="metric-card-label">Train loss</div><div class="metric-card-value">{final_loss:.3f}</div></div>
                <div class="metric-card"><div class="metric-card-label">Validation loss</div><div class="metric-card-value">{final_val_loss:.3f}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        epochs = list(range(1, len(history["accuracy"]) + 1))
        colA, colB = st.columns(2)
        with colA:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=epochs, y=[a * 100 for a in history["accuracy"]], name="Train", line=dict(color=COLOR_GREEN, width=3)))
            fig_acc.add_trace(go.Scatter(x=epochs, y=[a * 100 for a in history["val_accuracy"]], name="Validation", line=dict(color=COLOR_RED, width=3, dash="dot")))
            fig_acc.update_layout(
                title="Accuracy over epochs", height=360,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=CHART_FONT,
                xaxis_title="Epoch", yaxis_title="Accuracy (%)", legend=dict(orientation="h", y=1.12),
                xaxis=dict(gridcolor=CHART_GRID, color="#C5D6C8"), yaxis=dict(gridcolor=CHART_GRID, color="#C5D6C8"),
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        with colB:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs, y=history["loss"], name="Train", line=dict(color=COLOR_AMBER, width=3)))
            fig_loss.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Validation", line=dict(color=COLOR_RED, width=3, dash="dot")))
            fig_loss.update_layout(
                title="Loss over epochs", height=360,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=CHART_FONT,
                xaxis_title="Epoch", yaxis_title="Loss", legend=dict(orientation="h", y=1.12),
                xaxis=dict(gridcolor=CHART_GRID, color="#C5D6C8"), yaxis=dict(gridcolor=CHART_GRID, color="#C5D6C8"),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown("<div class='section-eyebrow' style='margin-top:0.8rem;'>ARCHITECTURE</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
        5 stacked <b>Conv2D → Conv2D → MaxPool</b> blocks (32 → 64 → 128 → 256 → 512 filters),
        followed by <b>Dropout(0.25)</b>, a <b>Flatten</b>, a <b>Dense(1500, relu)</b> layer,
        <b>Dropout(0.4)</b>, and a final <b>Dense({len(CLASS_NAMES)}, softmax)</b> output layer.
        Trained with the Adam optimizer (lr=0.0001) and categorical cross-entropy loss
        on 128×128 RGB leaf images across {len(CLASS_NAMES)} crop/disease classes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is not None:
        with st.expander("View full Keras model summary"):
            buf = io.StringIO()
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            st.code(buf.getvalue(), language="text")

# --------------------------------------------------------------------------------------
# PAGE: Specimen Index
# --------------------------------------------------------------------------------------
with tab_index:
    st.markdown(
        """
        <div class="section-eyebrow">REFERENCE CATALOG</div>
        <div class="section-title">Specimen Index</div>
        <div class="section-sub">All 38 crop/condition classes the model was trained to recognize.</div>
        """,
        unsafe_allow_html=True,
    )

    crops = sorted({v["crop"] for v in DISEASE_INFO.values()})
    colf1, colf2 = st.columns([1, 2])
    with colf1:
        crop_filter = st.selectbox("Filter by crop", ["All crops"] + crops)
    with colf2:
        search = st.text_input("Search condition", placeholder="e.g. blight, rust, mildew…")

    items = list(DISEASE_INFO.items())
    if crop_filter != "All crops":
        items = [(k, v) for k, v in items if v["crop"] == crop_filter]
    if search:
        items = [(k, v) for k, v in items if search.lower() in v["condition"].lower() or search.lower() in v["crop"].lower()]

    st.markdown(f"<span class='footnote'>Showing {len(items)} of {len(DISEASE_INFO)} classes</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (name, info) in enumerate(items):
        tag_class = "tag-healthy" if info["healthy"] else "tag-disease"
        tag_text = "HEALTHY" if info["healthy"] else "DISEASE"
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="index-card">
                    <div class="index-card-head">
                        <span class="index-crop">{info['crop']} — {info['condition']}</span>
                        <span class="tag {tag_class}">{tag_text}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Details"):
                st.write(info["description"])
                st.markdown(f"<span class='severity-badge'>SEVERITY · {info['severity']}</span>", unsafe_allow_html=True)
                st.markdown("**Symptoms:**")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")
                st.markdown("**Care / Treatment:**")
                for t in info["treatment"]:
                    st.markdown(f"- {t}")

# --------------------------------------------------------------------------------------
# PAGE: Field Notes (About)
# --------------------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        <div class="section-eyebrow">ABOUT THIS TOOL</div>
        <div class="section-title">Field Notes</div>
        <div class="section-sub">Why Plantix exists, and what it can and can't do for you.</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="card">
        <b>Plantix</b> uses a convolutional neural network trained on ~88K labeled leaf images
        spanning {len(CLASS_NAMES)} crop/disease classes (14 crop species) to help farmers,
        agronomists, and gardeners identify plant health issues from a single photo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="card">
            <h4>🧠 Tech stack</h4>
            <ul style="margin:0; padding-left:1.1rem;">
                <li>TensorFlow / Keras CNN</li>
                <li>Streamlit web app</li>
                <li>Plotly for interactive charts</li>
                <li>Pillow / OpenCV for image handling</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
            <h4>✨ Features</h4>
            <ul style="margin:0; padding-left:1.1rem;">
                <li>Upload or camera-based leaf capture</li>
                <li>Diagnostic readout with confidence breakdown</li>
                <li>Reference catalog with symptoms &amp; care tips</li>
                <li>Downloadable diagnosis report</li>
                <li>Training performance log</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "⚠️ Plantix is a decision-support tool, not a substitute for professional "
        "agronomic diagnosis. Confirm severe or spreading cases with a local expert "
        "before applying treatments."
    )