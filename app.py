"""
╔══════════════════════════════════════════════════════╗
║   CUSTOMER CHURN INTELLIGENCE PLATFORM               ║
║   Logistic Regression · Predictive Analytics         ║
╚══════════════════════════════════════════════════════╝

Run with:  streamlit run churn_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import base64
import re
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Customer Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Inter:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #09090f;
    --surface:   #11121d;
    --card:      #181929;
    --border:    #252840;
    --accent:    #7c6dfa;
    --accent2:   #fa6d9a;
    --accent3:   #6dfabd;
    --text:      #e8e8f5;
    --muted:     #6e6e9a;
    --danger:    #fa4d6d;
    --safe:      #4dfaad;
    --warn:      #fac84d;
}

/* ── App background ── */
.stApp { background: var(--bg); }
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }

/* ── Typography ── */
html, body, .stApp, .stMarkdown, p, li, span {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}
h1,h2,h3,h4 { font-family: 'Syne', sans-serif; }
code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #11121d 0%, #1a1233 50%, #0d1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(124,109,250,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(250,109,154,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(120deg, #c3b8ff, #7c6dfa, #fa6d9a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    font-size: 0.82rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.hero-pill {
    display: inline-block;
    background: rgba(124,109,250,0.15);
    border: 1px solid rgba(124,109,250,0.4);
    color: #c3b8ff;
    border-radius: 100px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.9rem;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.purple::before { background: linear-gradient(90deg, var(--accent), transparent); }
.metric-card.pink::before   { background: linear-gradient(90deg, var(--accent2), transparent); }
.metric-card.green::before  { background: linear-gradient(90deg, var(--accent3), transparent); }
.metric-card.yellow::before { background: linear-gradient(90deg, var(--warn), transparent); }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin-top: 0.3rem;
}
.metric-delta.up   { color: var(--safe); }
.metric-delta.down { color: var(--danger); }

/* ── Section headings ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
    border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
    margin: 1.6rem 0 1rem 0;
}

/* ── Table styling ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    border-radius: 12px;
    overflow: hidden;
}
.styled-table th {
    background: rgba(124,109,250,0.15);
    color: #c3b8ff;
    padding: 0.7rem 1rem;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.05em;
    font-size: 0.72rem;
    text-transform: uppercase;
}
.styled-table td {
    padding: 0.6rem 1rem;
    border-top: 1px solid var(--border);
    color: var(--text);
}
.styled-table tr:hover td { background: rgba(255,255,255,0.02); }

/* ── Risk badge ── */
.badge {
    display: inline-block;
    border-radius: 100px;
    padding: 0.18rem 0.7rem;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}
.badge-high   { background: rgba(250,77,109,0.18); color: #fa4d6d; border: 1px solid rgba(250,77,109,0.4); }
.badge-medium { background: rgba(250,200,77,0.15); color: #fac84d; border: 1px solid rgba(250,200,77,0.4); }
.badge-low    { background: rgba(77,250,173,0.12); color: #4dfaad; border: 1px solid rgba(77,250,173,0.4); }

/* ── Predict result card ── */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.2rem;
    text-align: center;
}
.result-card.churn {
    background: linear-gradient(135deg, rgba(250,77,109,0.12), rgba(250,109,154,0.06));
    border: 1px solid rgba(250,77,109,0.4);
}
.result-card.stay {
    background: linear-gradient(135deg, rgba(77,250,173,0.1), rgba(109,250,189,0.04));
    border: 1px solid rgba(77,250,173,0.35);
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.result-title.churn { color: #fa4d6d; }
.result-title.stay  { color: #4dfaad; }
.result-prob {
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    font-size: 0.85rem;
}

/* ── Probability bar ── */
.prob-bar-wrap {
    background: var(--border);
    border-radius: 100px;
    height: 10px;
    margin: 0.8rem 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}
.prob-bar-fill.churn { background: linear-gradient(90deg, #fa4d6d, #fa6d9a); }
.prob-bar-fill.stay  { background: linear-gradient(90deg, #4dfaad, #6dfabd); }

/* ── Sidebar tweaks ── */
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(120deg, #c3b8ff, #fa6d9a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
}

/* ── Streamlit widget recolour ── */
.stSelectbox label, .stSlider label, .stRadio label,
.stNumberInput label, .stCheckbox label {
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-baseweb="select"] > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9b6dfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.02em;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    gap: 0.2rem;
    padding: 0 0.4rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    color: var(--muted) !important;
    font-size: 0.88rem !important;
    font-weight: 600;
    padding: 0.7rem 1.2rem !important;
    border-radius: 8px 8px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(124,109,250,0.15) !important;
    color: #c3b8ff !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.4rem 0; }

/* ── Feature importance bar ── */
.feat-row {
    display: flex; align-items: center; gap: 0.8rem;
    margin-bottom: 0.55rem;
}
.feat-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    width: 200px;
    flex-shrink: 0;
    text-align: right;
}
.feat-bar-wrap { flex: 1; height: 8px; background: var(--border); border-radius: 100px; overflow: hidden; }
.feat-bar-fill { height: 100%; border-radius: 100px; }
.feat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text);
    width: 42px;
}

/* ── Info box ── */
.info-box {
    background: rgba(124,109,250,0.08);
    border: 1px solid rgba(124,109,250,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
.info-box p { margin: 0; font-size: 0.85rem; color: var(--muted); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn.csv")
    return df

@st.cache_resource
def train_model(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    df2 = df.drop(columns=["customerID"]).copy()
    cat_cols = df2.select_dtypes(include=["object"]).columns.tolist()
    cat_cols.remove("Churn")

    df_enc = pd.get_dummies(df2, columns=cat_cols)
    le = LabelEncoder()
    df_enc["Churn"] = le.fit_transform(df2["Churn"])

    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc_test  = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred, output_dict=True)

    return model, X, X_train, X_test, y_train, y_test, y_pred, acc_test, acc_train, cm, report, df_enc

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Tiny SVG bar chart (no Plotly dependency)
# ─────────────────────────────────────────────────────────────────────────────
def bar_svg(labels, values, colors, width=380, height=180, title=""):
    """Render a minimal horizontal bar chart as inline SVG HTML."""
    if not values:
        return ""
    max_v = max(values) or 1
    bar_h = 22
    gap   = 10
    pad_l = 130
    pad_r = 60
    pad_t = 36
    pad_b = 16
    total_h = pad_t + len(values) * (bar_h + gap) + pad_b

    lines = [
        f'<svg width="{width}" height="{total_h}" xmlns="http://www.w3.org/2000/svg" style="font-family:JetBrains Mono,monospace">',
    ]
    if title:
        lines.append(f'<text x="{pad_l}" y="20" font-size="11" fill="#6e6e9a" text-anchor="start" font-weight="600">{title}</text>')

    avail = width - pad_l - pad_r
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = pad_t + i * (bar_h + gap)
        bar_w = int(avail * value / max_v)

        # background bar
        lines.append(f'<rect x="{pad_l}" y="{y}" width="{avail}" height="{bar_h}" rx="5" fill="#252840"/>')
        # fill bar
        lines.append(f'<rect x="{pad_l}" y="{y}" width="{bar_w}" height="{bar_h}" rx="5" fill="{color}" opacity="0.85"/>')
        # label
        lines.append(f'<text x="{pad_l-8}" y="{y+bar_h//2+4}" font-size="10" fill="#6e6e9a" text-anchor="end">{label[:20]}</text>')
        # value
        lines.append(f'<text x="{pad_l+avail+6}" y="{y+bar_h//2+4}" font-size="10" fill="#e8e8f5" text-anchor="start">{value:.0f}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def heatmap_svg(matrix, labels, title="Confusion Matrix", width=280, height=280):
    """2×2 confusion matrix as SVG."""
    cell = (width - 60) // 2
    colors = [["#4dfaad", "#fa4d6d"], ["#fa4d6d", "#4dfaad"]]

    lines = [
        f'<svg width="{width}" height="{height+40}" xmlns="http://www.w3.org/2000/svg" style="font-family:JetBrains Mono,monospace">',
        f'<text x="{width//2}" y="20" font-size="11" fill="#6e6e9a" text-anchor="middle" font-weight="600">{title}</text>',
    ]
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            x = 50 + j * cell
            y = 30 + i * cell
            alpha = min(0.9, 0.15 + 0.75 * val / (matrix.max() or 1))
            # parse hex color for alpha: simplify with opacity
            lines.append(f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" rx="8" fill="{colors[i][j]}" opacity="{alpha:.2f}"/>')
            lines.append(f'<text x="{x+cell//2-2}" y="{y+cell//2+5}" font-size="22" fill="#e8e8f5" text-anchor="middle" font-weight="700">{val}</text>')

    # Axis labels
    for j, lbl in enumerate(labels):
        x = 50 + j * cell + cell // 2 - 2
        lines.append(f'<text x="{x}" y="{30 + 2*cell + 18}" font-size="9" fill="#6e6e9a" text-anchor="middle">Pred {lbl}</text>')
    for i, lbl in enumerate(labels):
        y = 30 + i * cell + cell // 2 + 5
        lines.append(f'<text x="46" y="{y}" font-size="9" fill="#6e6e9a" text-anchor="end">Act {lbl}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def donut_svg(val, total, color="#7c6dfa", size=120, label=""):
    """Simple SVG donut/ring chart."""
    r = 44
    cx = cy = size // 2
    circ = 2 * 3.14159 * r
    frac = val / total if total else 0
    dash = frac * circ

    return f"""
    <svg width="{size}" height="{size+24}" xmlns="http://www.w3.org/2000/svg"
         style="font-family:JetBrains Mono,monospace">
      <circle cx="{cx}" cy="{cy}" r="{r}" stroke="#252840" stroke-width="12" fill="none"/>
      <circle cx="{cx}" cy="{cy}" r="{r}"
              stroke="{color}" stroke-width="12" fill="none"
              stroke-dasharray="{dash:.1f} {circ:.1f}"
              stroke-dashoffset="{circ/4:.1f}"
              stroke-linecap="round"/>
      <text x="{cx}" y="{cy+5}" font-size="14" fill="#e8e8f5"
            text-anchor="middle" font-weight="700">{frac*100:.0f}%</text>
      <text x="{cx}" y="{size+16}" font-size="9" fill="#6e6e9a"
            text-anchor="middle">{label}</text>
    </svg>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔮 ChurnIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Customer Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["🏠  Dashboard", "🔬  Model Analysis", "🎯  Predict Customer", "📋  Data Explorer"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#6e6e9a; line-height:1.8;">
    <b style="color:#c3b8ff;">Algorithm</b><br>Logistic Regression<br><br>
    <b style="color:#c3b8ff;">Dataset</b><br>300 customers · 20 features<br><br>
    <b style="color:#c3b8ff;">Target</b><br>Binary churn classification
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
try:
    df = load_data()
    model, X, X_train, X_test, y_train, y_test, y_pred, acc_test, acc_train, cm, report, df_enc = train_model(df)
    data_ok = True
except Exception as e:
    data_ok = False
    load_error = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# ███  PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if nav == "🏠  Dashboard":
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Customer Churn Intelligence</div>
        <div class="hero-sub">Powered by Logistic Regression · Real-time Prediction Engine</div>
        <div class="hero-pill">✦ Logistic Regression · Binary Classification · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.error(f"⚠️ Could not load dataset. Make sure `customer_churn_prediction_dataset.csv` is in the same folder.\n\nError: {load_error}")
        st.stop()

    # ── KPI Row ──
    total    = len(df)
    churned  = (df["Churn"] == "Yes").sum()
    retained = total - churned
    churn_rate = churned / total * 100
    avg_monthly = df["MonthlyCharges"].mean()

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card purple">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total}</div>
            <div class="metric-delta up">↑ Full dataset loaded</div>
        </div>
        <div class="metric-card pink">
            <div class="metric-label">Churned</div>
            <div class="metric-value">{churned}</div>
            <div class="metric-delta down">↓ Left the service</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Retained</div>
            <div class="metric-value">{retained}</div>
            <div class="metric-delta up">↑ Active customers</div>
        </div>
        <div class="metric-card yellow">
            <div class="metric-label">Churn Rate</div>
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-delta down">↓ Risk level</div>
        </div>
        <div class="metric-card purple">
            <div class="metric-label">Avg Monthly $</div>
            <div class="metric-value">${avg_monthly:.0f}</div>
            <div class="metric-delta">Per customer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row ──
    col1, col2, col3 = st.columns([2, 2, 1.4])

    with col1:
        st.markdown('<div class="section-title">Churn by Contract Type</div>', unsafe_allow_html=True)
        ct = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").sum())
        svg = bar_svg(
            labels=ct.index.tolist(),
            values=ct.values.tolist(),
            colors=["#fa4d6d", "#fac84d", "#4dfaad"],
            width=360,
            title="Churned customers per contract"
        )
        st.markdown(svg, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Churn by Internet Service</div>', unsafe_allow_html=True)
        is_ = df.groupby("InternetService")["Churn"].apply(lambda x: (x == "Yes").sum())
        svg2 = bar_svg(
            labels=is_.index.tolist(),
            values=is_.values.tolist(),
            colors=["#7c6dfa", "#fa6d9a", "#6dfabd"],
            width=360,
            title="Churned customers per service type"
        )
        st.markdown(svg2, unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-title">Status Split</div>', unsafe_allow_html=True)
        st.markdown(
            donut_svg(churned, total, color="#fa4d6d", size=130, label="Churn Rate"),
            unsafe_allow_html=True
        )
        st.markdown(
            donut_svg(retained, total, color="#4dfaad", size=130, label="Retention Rate"),
            unsafe_allow_html=True
        )

    # ── Payment method breakdown ──
    st.markdown('<div class="section-title">Churn by Payment Method</div>', unsafe_allow_html=True)
    pm = df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x == "Yes").sum()).sort_values(ascending=False)
    svg3 = bar_svg(
        labels=pm.index.tolist(),
        values=pm.values.tolist(),
        colors=["#fa4d6d", "#fac84d", "#7c6dfa", "#4dfaad"],
        width=600,
        title="Churned customers per payment method"
    )
    st.markdown(svg3, unsafe_allow_html=True)

    # ── Model accuracy pills ──
    st.markdown("---")
    st.markdown(f"""
    <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-top:0.5rem;">
        <div style="background:rgba(77,250,173,0.1); border:1px solid rgba(77,250,173,0.3);
                    border-radius:12px; padding:0.8rem 1.4rem; min-width:160px;">
            <div style="font-family:JetBrains Mono,monospace; font-size:0.68rem;
                        color:#6e6e9a; text-transform:uppercase; letter-spacing:0.08em;">Test Accuracy</div>
            <div style="font-family:Syne,sans-serif; font-size:1.8rem; font-weight:800;
                        color:#4dfaad;">{acc_test*100:.1f}%</div>
        </div>
        <div style="background:rgba(124,109,250,0.1); border:1px solid rgba(124,109,250,0.3);
                    border-radius:12px; padding:0.8rem 1.4rem; min-width:160px;">
            <div style="font-family:JetBrains Mono,monospace; font-size:0.68rem;
                        color:#6e6e9a; text-transform:uppercase; letter-spacing:0.08em;">Train Accuracy</div>
            <div style="font-family:Syne,sans-serif; font-size:1.8rem; font-weight:800;
                        color:#c3b8ff;">{acc_train*100:.1f}%</div>
        </div>
        <div style="background:rgba(250,200,77,0.1); border:1px solid rgba(250,200,77,0.3);
                    border-radius:12px; padding:0.8rem 1.4rem; min-width:160px;">
            <div style="font-family:JetBrains Mono,monospace; font-size:0.68rem;
                        color:#6e6e9a; text-transform:uppercase; letter-spacing:0.08em;">Precision (Churn)</div>
            <div style="font-family:Syne,sans-serif; font-size:1.8rem; font-weight:800;
                        color:#fac84d;">{report['1']['precision']*100:.1f}%</div>
        </div>
        <div style="background:rgba(250,109,154,0.1); border:1px solid rgba(250,109,154,0.3);
                    border-radius:12px; padding:0.8rem 1.4rem; min-width:160px;">
            <div style="font-family:JetBrains Mono,monospace; font-size:0.68rem;
                        color:#6e6e9a; text-transform:uppercase; letter-spacing:0.08em;">Recall (Churn)</div>
            <div style="font-family:Syne,sans-serif; font-size:1.8rem; font-weight:800;
                        color:#fa6d9a;">{report['1']['recall']*100:.1f}%</div>
        </div>
        <div style="background:rgba(109,250,189,0.1); border:1px solid rgba(109,250,189,0.3);
                    border-radius:12px; padding:0.8rem 1.4rem; min-width:160px;">
            <div style="font-family:JetBrains Mono,monospace; font-size:0.68rem;
                        color:#6e6e9a; text-transform:uppercase; letter-spacing:0.08em;">F1-Score (Churn)</div>
            <div style="font-family:Syne,sans-serif; font-size:1.8rem; font-weight:800;
                        color:#6dfabd;">{report['1']['f1-score']*100:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ███  PAGE: MODEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "🔬  Model Analysis":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Model Analysis</div>
        <div class="hero-sub">Logistic Regression · Coefficients · Performance Metrics</div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.error("Dataset not found."); st.stop()

    tab1, tab2, tab3 = st.tabs(["📊  Confusion Matrix", "⚖️  Feature Importance", "📈  Classification Report"])

    with tab1:
        col_a, col_b = st.columns([1.2, 2])
        with col_a:
            st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
            st.markdown(
                heatmap_svg(cm, ["No", "Yes"], width=280, height=260),
                unsafe_allow_html=True
            )
        with col_b:
            st.markdown('<div class="section-title">What These Numbers Mean</div>', unsafe_allow_html=True)
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            <table class="styled-table" style="margin-top:0.5rem">
              <thead><tr><th>Metric</th><th>Value</th><th>Meaning</th></tr></thead>
              <tbody>
                <tr><td>True Negatives (TN)</td>
                    <td><span class="badge badge-low">{tn}</span></td>
                    <td>Correctly predicted as NOT churned</td></tr>
                <tr><td>False Positives (FP)</td>
                    <td><span class="badge badge-medium">{fp}</span></td>
                    <td>Incorrectly flagged as churned</td></tr>
                <tr><td>False Negatives (FN)</td>
                    <td><span class="badge badge-high">{fn}</span></td>
                    <td>Missed churners — high cost!</td></tr>
                <tr><td>True Positives (TP)</td>
                    <td><span class="badge badge-low">{tp}</span></td>
                    <td>Correctly identified churners</td></tr>
              </tbody>
            </table>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-title">Top Feature Importances (|coefficient|)</div>', unsafe_allow_html=True)
        coefs = pd.Series(np.abs(model.coef_[0]), index=X.columns).sort_values(ascending=False).head(15)
        max_c = coefs.max()

        palette = [
            "#7c6dfa","#9b6dfa","#fa6d9a","#fa4d6d","#fac84d",
            "#6dfabd","#4dfaad","#6daffa","#fa9b6d","#c3b8ff",
            "#fa6dbd","#adfac3","#fa6d6d","#6dfaf0","#d4fa6d"
        ]
        html_bars = []
        for i, (feat, coef) in enumerate(coefs.items()):
            pct = coef / max_c * 100
            color = palette[i % len(palette)]
            short = feat[:28] + "…" if len(feat) > 28 else feat
            html_bars.append(f"""
            <div class="feat-row">
                <div class="feat-name">{short}</div>
                <div class="feat-bar-wrap">
                    <div class="feat-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
                </div>
                <div class="feat-val">{coef:.3f}</div>
            </div>""")

        st.markdown("\n".join(html_bars), unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:1rem;">
        <p>ℹ️  <b style="color:#c3b8ff;">Coefficient magnitude</b> indicates how strongly each feature influences
        the churn probability. Larger = more influential. Sign indicates direction.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-title">Full Classification Report</div>', unsafe_allow_html=True)
        rows = []
        for cls, label in [("0", "No Churn"), ("1", "Churn"), ("macro avg", "Macro Avg"), ("weighted avg", "Weighted Avg")]:
            if cls in report:
                r = report[cls]
                badge_cls = "badge-low" if cls == "0" else "badge-high" if cls == "1" else "badge-medium"
                rows.append(f"""
                <tr>
                    <td><span class="badge {badge_cls}">{label}</span></td>
                    <td>{r.get('precision', 0)*100:.1f}%</td>
                    <td>{r.get('recall', 0)*100:.1f}%</td>
                    <td>{r.get('f1-score', 0)*100:.1f}%</td>
                    <td>{int(r.get('support', 0))}</td>
                </tr>""")

        st.markdown(f"""
        <table class="styled-table">
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1.4rem; display:flex; gap:1rem; flex-wrap:wrap;">
            <div style="font-family:JetBrains Mono,monospace; font-size:0.8rem;
                        background:var(--card); border:1px solid var(--border);
                        border-radius:10px; padding:0.8rem 1.2rem;">
                <span style="color:#6e6e9a;">Overall Accuracy → </span>
                <span style="color:#4dfaad; font-weight:700;">{report['accuracy']*100:.2f}%</span>
            </div>
            <div style="font-family:JetBrains Mono,monospace; font-size:0.8rem;
                        background:var(--card); border:1px solid var(--border);
                        border-radius:10px; padding:0.8rem 1.2rem;">
                <span style="color:#6e6e9a;">Test samples → </span>
                <span style="color:#c3b8ff; font-weight:700;">{len(y_test)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ███  PAGE: PREDICT CUSTOMER
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "🎯  Predict Customer":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Live Prediction</div>
        <div class="hero-sub">Enter customer profile → get instant churn probability</div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.error("Dataset not found."); st.stop()

    col_form, col_result = st.columns([1.4, 1])

    with col_form:
        st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            gender       = st.selectbox("Gender", ["Male", "Female"])
            senior       = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner      = st.selectbox("Partner", ["Yes", "No"])
            dependents   = st.selectbox("Dependents", ["No", "Yes"])
            phone_svc    = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_svc = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with c2:
            online_sec   = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_bkp   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_prot  = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            stream_tv    = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            stream_mov   = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract     = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        c3, c4 = st.columns(2)
        with c3:
            paperless    = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment      = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                            "Bank transfer", "Credit card"])
        with c4:
            tenure       = st.slider("Tenure (months)", 0, 72, 12)
            monthly_chg  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total_chg    = st.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_chg), step=10.0)

        predict_btn = st.button("🔮  Predict Churn Risk")

    with col_result:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            # Build input row matching training columns
            inp = {
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "tenure": tenure,
                "MonthlyCharges": monthly_chg,
                "TotalCharges": total_chg,
            }
            # One-hot encode the same way training did
            cat_vals = {
                "gender": gender,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone_svc,
                "MultipleLines": multi_lines,
                "InternetService": internet_svc,
                "OnlineSecurity": online_sec,
                "OnlineBackup": online_bkp,
                "DeviceProtection": device_prot,
                "TechSupport": tech_support,
                "StreamingTV": stream_tv,
                "StreamingMovies": stream_mov,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
            }
            for col, val in cat_vals.items():
                for option in df[col].unique():
                    inp[f"{col}_{option}"] = 1 if val == option else 0

            input_df = pd.DataFrame([inp])
            # Align columns
            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X.columns]

            prob = model.predict_proba(input_df)[0]
            churn_prob = prob[1]
            pred = "Churn" if churn_prob >= 0.5 else "Stay"

            if pred == "Churn":
                pct_fill = int(churn_prob * 100)
                st.markdown(f"""
                <div class="result-card churn">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title churn">HIGH RISK</div>
                    <div class="result-prob">Churn probability: <b>{churn_prob*100:.1f}%</b></div>
                    <div class="prob-bar-wrap" style="margin:1rem 0;">
                        <div class="prob-bar-fill churn" style="width:{pct_fill}%;"></div>
                    </div>
                    <div style="font-family:JetBrains Mono,monospace; font-size:0.75rem; color:#6e6e9a;">
                        This customer is likely to leave. Consider retention offers.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                pct_fill = int((1 - churn_prob) * 100)
                st.markdown(f"""
                <div class="result-card stay">
                    <div class="result-icon">✅</div>
                    <div class="result-title stay">LOW RISK</div>
                    <div class="result-prob">Retention probability: <b>{(1-churn_prob)*100:.1f}%</b></div>
                    <div class="prob-bar-wrap" style="margin:1rem 0;">
                        <div class="prob-bar-fill stay" style="width:{pct_fill}%;"></div>
                    </div>
                    <div style="font-family:JetBrains Mono,monospace; font-size:0.75rem; color:#6e6e9a;">
                        This customer is likely to stay. Keep up engagement.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Risk factors
            st.markdown('<div class="section-title">Risk Factors</div>', unsafe_allow_html=True)
            risk_signals = []
            if contract == "Month-to-month":
                risk_signals.append(("Month-to-month contract", "high"))
            if internet_svc == "Fiber optic":
                risk_signals.append(("Fiber optic internet", "medium"))
            if tenure < 12:
                risk_signals.append(("Short tenure < 12 months", "high"))
            if online_sec == "No":
                risk_signals.append(("No Online Security", "medium"))
            if tech_support == "No":
                risk_signals.append(("No Tech Support", "medium"))
            if payment == "Electronic check":
                risk_signals.append(("Electronic check payment", "low"))

            if risk_signals:
                rows_html = "".join(
                    f'<tr><td>{s}</td><td><span class="badge badge-{l}">{l.upper()}</span></td></tr>'
                    for s, l in risk_signals
                )
                st.markdown(f"""
                <table class="styled-table">
                  <thead><tr><th>Signal</th><th>Risk Level</th></tr></thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box"><p>✅ No strong risk signals detected.</p></div>',
                            unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align:center; padding:2rem;">
            <p style="font-size:2rem; margin-bottom:0.5rem;">🎯</p>
            <p style="color:#6e6e9a;">Fill in the customer profile on the left<br>and hit <b style="color:#c3b8ff;">Predict Churn Risk</b>.</p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ███  PAGE: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "📋  Data Explorer":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Data Explorer</div>
        <div class="hero-sub">Browse & filter the raw dataset · 300 customers · 20 features</div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.error("Dataset not found."); st.stop()

    # Filters
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        f_churn    = st.selectbox("Churn Filter", ["All", "Yes", "No"])
    with col_f2:
        f_contract = st.selectbox("Contract", ["All"] + df["Contract"].unique().tolist())
    with col_f3:
        f_internet = st.selectbox("Internet", ["All"] + df["InternetService"].unique().tolist())
    with col_f4:
        f_payment  = st.selectbox("Payment", ["All"] + df["PaymentMethod"].unique().tolist())

    filtered = df.copy()
    if f_churn    != "All": filtered = filtered[filtered["Churn"]           == f_churn]
    if f_contract != "All": filtered = filtered[filtered["Contract"]        == f_contract]
    if f_internet != "All": filtered = filtered[filtered["InternetService"] == f_internet]
    if f_payment  != "All": filtered = filtered[filtered["PaymentMethod"]   == f_payment]

    st.markdown(f"""
    <div style="font-family:JetBrains Mono,monospace; font-size:0.75rem; color:#6e6e9a;
                margin-bottom:0.8rem;">
        Showing <b style="color:#c3b8ff;">{len(filtered)}</b> of {len(df)} records
    </div>
    """, unsafe_allow_html=True)

    # Render table (first 50 rows)
    display_cols = ["customerID","gender","SeniorCitizen","tenure","Contract",
                    "InternetService","MonthlyCharges","TotalCharges","Churn"]
    chunk = filtered[display_cols].head(50)

    header = "".join(f"<th>{c}</th>" for c in chunk.columns)
    rows_html = []
    for _, row in chunk.iterrows():
        churn_val = row["Churn"]
        badge = f'<span class="badge badge-{"high" if churn_val=="Yes" else "low"}">{churn_val}</span>'
        cells = []
        for c in chunk.columns:
            v = row[c]
            if c == "Churn":
                cells.append(f"<td>{badge}</td>")
            elif c in ("MonthlyCharges","TotalCharges"):
                cells.append(f"<td>${float(v):.2f}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="styled-table">
      <thead><tr>{header}</tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>
    </div>
    <div style="font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#6e6e9a; margin-top:0.5rem;">
      Displaying first 50 rows of filtered results.
    </div>
    """, unsafe_allow_html=True)

    # Summary stats
    st.markdown('<div class="section-title">Numeric Summary</div>', unsafe_allow_html=True)
    num_cols = ["tenure","MonthlyCharges","TotalCharges"]
    desc = filtered[num_cols].describe().round(2)

    header2 = "<th>Stat</th>" + "".join(f"<th>{c}</th>" for c in num_cols)
    rows2 = []
    for stat in desc.index:
        cells2 = [f"<td><b style='color:#c3b8ff;'>{stat}</b></td>"]
        for col in num_cols:
            cells2.append(f"<td>{desc.loc[stat, col]}</td>")
        rows2.append(f"<tr>{''.join(cells2)}</tr>")

    st.markdown(f"""
    <table class="styled-table">
      <thead><tr>{header2}</tr></thead>
      <tbody>{''.join(rows2)}</tbody>
    </table>
    """, unsafe_allow_html=True)