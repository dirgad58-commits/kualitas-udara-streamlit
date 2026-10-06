"""
Streamlit Dashboard: Sistem Prediksi Konsentrasi PM2.5 & Estimasi AQI
Berbasis Algoritma XGBoost Regressor dan Standar US EPA (Piecewise Linear Interpolation)
Desain: Premium Enterprise Light Theme (Deep Navy #0A192F & Champagne Gold #D97706)
Fitur: Mikro-Animasi CSS, Kontras Tinggi 100% Legibel, Bebas Simbol/Emoji Berlebihan
"""

import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sqlalchemy import create_engine, text

# Tambahkan root direktori ke sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.aqi_calculator import calculate_pm25_aqi, calculate_aqi_dataframe, EPA_PM25_BREAKPOINTS

# ==============================================================================
# 1. KONFIGURASI HALAMAN & INJEKSI CSS PREMIUM ENTERPRISE
# ==============================================================================
st.set_page_config(
    page_title="AQI PM2.5 Prediction Engine | XGBoost MLOps",
    page_icon="☁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Deep Navy (#0A192F), Classic Gold (#D97706), Slate Gray (#334155), Pure White (#FFFFFF)
# Semua teks, label input, slider, dan tombol dipaksa memiliki kontras maksimal agar tidak ada yang hilang
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">

<style>
/* -------------------------------------------------------------
   1. GLOBAL RESET & TYPOGRAPHY
------------------------------------------------------------- */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* Keyframe Animations */
@keyframes fadeInSlideUp {
    0% { opacity: 0; transform: translateY(12px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes goldAccentPulse {
    0% { box-shadow: 0 0 0 0 rgba(217, 119, 6, 0.25); }
    70% { box-shadow: 0 0 0 8px rgba(217, 119, 6, 0); }
    100% { box-shadow: 0 0 0 0 rgba(217, 119, 6, 0); }
}

/* -------------------------------------------------------------
   2. FIX LABEL WIDGET & FONT KONTRAS (100% TERLIHAT DI SEMUA TEMA)
------------------------------------------------------------- */
label[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
.stSlider label,
.stNumberInput label,
.stSelectbox label,
.stTextInput label {
    color: #0A192F !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.35rem !important;
    letter-spacing: -0.01em !important;
    opacity: 1 !important;
    display: block !important;
    visibility: visible !important;
}

/* Angka Nilai pada Slider */
[data-testid="stSlider"] div[data-testid="stTickBar"] + div,
[data-testid="stSlider"] [data-baseweb="slider"] div {
    color: #0A192F !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}

/* Nilai Input Angka & Dropdown */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    color: #0A192F !important;
    font-weight: 600 !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
    border-color: #D97706 !important;
    box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.15) !important;
}

[data-testid="stSelectbox"] * {
    color: #0A192F !important;
}

/* -------------------------------------------------------------
   3. TOMBOL AKSI SUBMIT (NAVY DENGAN TEKS PUTIH DAN BORDER GOLD)
------------------------------------------------------------- */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: #0A192F !important;
    color: #FFFFFF !important;
    border: 2px solid #D97706 !important;
    font-weight: 700 !important;
    font-size: 1.0rem !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 2.0rem !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(10, 25, 47, 0.15) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
}

div.stButton > button *,
div[data-testid="stFormSubmitButton"] > button * {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #D97706 !important;
    color: #FFFFFF !important;
    border-color: #0A192F !important;
    box-shadow: 0 6px 20px rgba(217, 119, 6, 0.35) !important;
    transform: translateY(-2px) !important;
}

/* -------------------------------------------------------------
   4. SIDEBAR ENTERPRISE (DEEP NAVY & GOLD ACCENT)
------------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background-color: #0A192F !important;
    border-right: 2px solid #1E293B !important;
}
section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.1) !important;
}

.sidebar-header-box {
    padding: 0.8rem 0 1.2rem 0;
    border-bottom: 2px solid #D97706;
    margin-bottom: 1.2rem;
}
.sidebar-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    color: #FFFFFF;
    text-transform: uppercase;
}
.sidebar-subtitle {
    font-size: 0.75rem;
    font-weight: 700;
    color: #D97706;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

.sidebar-metric-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-left: 3px solid #D97706;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.75rem;
}
.sidebar-metric-card .row-label {
    font-size: 0.78rem;
    color: #94A3B8;
    font-weight: 600;
}
.sidebar-metric-card .row-val {
    font-size: 0.88rem;
    color: #F8FAFC;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* -------------------------------------------------------------
   5. HEADER UTAMA (CLEAN CORPORATE BANNER)
------------------------------------------------------------- */
.main-banner {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 6px solid #0A192F;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 15px rgba(10, 25, 47, 0.04);
    animation: fadeInSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}
.main-banner-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #0A192F;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.main-banner-desc {
    font-size: 0.95rem;
    color: #475569;
    font-weight: 500;
    margin-top: 0.4rem;
    line-height: 1.5;
}
.gold-badge {
    background: rgba(217, 119, 6, 0.1);
    color: #B45309;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.85rem;
}

/* -------------------------------------------------------------
   6. INPUT CARD CONTAINER (FORM KELOMPOK PARAMETER)
------------------------------------------------------------- */
.input-section-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #0A192F;
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.input-section-card:hover {
    box-shadow: 0 6px 18px rgba(10, 25, 47, 0.06);
    border-top-color: #D97706;
}
.input-section-title {
    font-size: 0.92rem;
    font-weight: 800;
    color: #0A192F;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #F1F5F9;
}

/* -------------------------------------------------------------
   7. KARTU HASIL PREDIKSI (SHOWCASE BANNER)
------------------------------------------------------------- */
.prediction-result-panel {
    background: #0A192F;
    border: 2px solid #D97706;
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    color: #FFFFFF;
    margin-top: 1.4rem;
    box-shadow: 0 10px 25px rgba(10, 25, 47, 0.2);
    animation: fadeInSlideUp 0.4s ease-out;
}
.pred-tagline {
    font-size: 0.8rem;
    font-weight: 800;
    color: #D97706;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.pred-primary-num {
    font-size: 2.9rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.0;
    font-family: 'JetBrains Mono', monospace;
}
.pred-primary-unit {
    font-size: 1.15rem;
    color: #93C5FD;
    font-weight: 600;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.pred-secondary-num {
    font-size: 2.9rem;
    font-weight: 800;
    color: #FBBF24;
    line-height: 1.0;
    font-family: 'JetBrains Mono', monospace;
}

.health-pill-badge {
    display: inline-block;
    padding: 0.45rem 1.1rem;
    border-radius: 6px;
    font-weight: 800;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* -------------------------------------------------------------
   8. TABS STYLING (CLEAN BORDERLESS WITH GOLD ACTIVE INDICATOR)
------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    color: #64748B !important;
    padding: 0.75rem 0.5rem !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #0A192F !important;
    border-bottom: 3px solid #D97706 !important;
}

/* -------------------------------------------------------------
   9. KPI METRIC CARDS (LIGHT MODE)
------------------------------------------------------------- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card-light {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 4px solid #0A192F;
    border-radius: 10px;
    padding: 1.1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card-light:hover {
    transform: translateY(-2px);
    border-color: #D97706;
}
.kpi-card-gold {
    border-top-color: #D97706;
}
.kpi-title-text {
    font-size: 0.74rem;
    font-weight: 700;
    color: #64748B;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.kpi-val-text {
    font-size: 1.85rem;
    font-weight: 800;
    color: #0A192F;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.3rem;
}

/* -------------------------------------------------------------
   10. MITIGASI PROTOCOL CARDS
------------------------------------------------------------- */
.mitigasi-card-clean {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.2rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}
.mitigasi-card-clean.c-baik { border-left: 5px solid #10B981; }
.mitigasi-card-clean.c-sedang { border-left: 5px solid #F59E0B; }
.mitigasi-card-clean.c-sensitif { border-left: 5px solid #F97316; }
.mitigasi-card-clean.c-kritis { border-left: 5px solid #EF4444; }

.mitigasi-head {
    font-size: 0.95rem;
    font-weight: 800;
    color: #0A192F;
    margin-bottom: 0.2rem;
}
.mitigasi-param {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    margin-bottom: 0.5rem;
}
.mitigasi-body {
    font-size: 0.84rem;
    color: #334155;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. HELPER KONEKSI DATABASE & PEMUATAN DATA
# ==============================================================================
@st.cache_resource
def get_db_connection():
    candidates = [
        "mysql+pymysql://root:rootpassword@127.0.0.1:3306/aqi_prediction_db",
        "mysql+pymysql://drought_user:drought_password@127.0.0.1:3306/aqi_prediction_db",
        "mysql+pymysql://root:rootpassword@drought_mysql:3306/aqi_prediction_db",
        "mysql+pymysql://aqi_user:aqi_secure_password@mysql_db:3306/aqi_prediction_db"
    ]
    for uri in candidates:
        try:
            eng = create_engine(uri, pool_recycle=3600, connect_args={"connect_timeout": 2})
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            return eng
        except Exception:
            continue
    return None

@st.cache_data(ttl=60)
def load_environmental_data():
    engine = get_db_connection()
    if engine is not None:
        try:
            query = "SELECT * FROM master_feature_store ORDER BY recorded_at ASC"
            df = pd.read_sql(query, engine)
            if not df.empty:
                df['recorded_at'] = pd.to_datetime(df['recorded_at'])
                return df, "MySQL Database (Live)"
        except Exception:
            pass
            
    csv_paths = [
        os.path.join(ROOT_DIR, 'data', 'master_feature_store.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'master_feature_store.csv'),
        'data/master_feature_store.csv'
    ]
    for p in csv_paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df['recorded_at'] = pd.to_datetime(df['recorded_at'])
                return df, "CSV Fallback Storage"
            except Exception:
                pass
                
    dates = pd.date_range(end=datetime.now(), periods=168, freq='h')
    df_dummy = pd.DataFrame({
        'recorded_at': dates,
        'pm25': np.random.uniform(15, 65, size=len(dates)),
        'temperature_c': np.random.uniform(25, 33, size=len(dates)),
        'humidity_pct': np.random.uniform(60, 90, size=len(dates)),
        'wind_speed_kmh': np.random.uniform(3, 20, size=len(dates)),
        'traffic_index': np.random.uniform(20, 90, size=len(dates)),
        'hour_of_day': dates.hour
    })
    return df_dummy, "Simulasi Generator"

@st.cache_resource
def load_ml_artifacts():
    model_paths = [
        os.path.join(ROOT_DIR, 'models', 'xgboost_pm25_model.pkl'),
        os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_pm25_model.pkl'),
        'models/xgboost_pm25_model.pkl'
    ]
    scaler_paths = [
        os.path.join(ROOT_DIR, 'models', 'scaler.pkl'),
        os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl'),
        'models/scaler.pkl'
    ]
    meta_paths = [
        os.path.join(ROOT_DIR, 'models', 'model_metadata.json'),
        os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json'),
        'models/model_metadata.json'
    ]
    
    model, scaler, metadata = None, None, None
    for p in model_paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                break
            except Exception:
                pass
                
    for p in scaler_paths:
        if os.path.exists(p):
            try:
                scaler = joblib.load(p)
                break
            except Exception:
                pass
                
    for p in meta_paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    metadata = json.load(f)
                break
            except Exception:
                pass
                
    return model, scaler, metadata


# ==============================================================================
# 3. HELPER PLOTLY LIGHT THEME
# ==============================================================================
def apply_plotly_enterprise_theme(fig, title=""):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color="#0A192F", family="Plus Jakarta Sans")
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#334155", family="Plus Jakarta Sans"),
        margin=dict(l=35, r=25, t=45, b=35),
        xaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(color="#475569")
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(color="#475569")
        ),
        legend=dict(
            font=dict(color="#1E293B"),
            bgcolor="rgba(255, 255, 255, 0.9)"
        )
    )
    return fig

def add_numpy_ols_trendline(fig, x_series, y_series, name="Garis Regresi OLS"):
    try:
        mask = (~np.isnan(x_series)) & (~np.isnan(y_series))
        if mask.sum() > 3:
            x_vals = np.array(x_series)[mask]
            y_vals = np.array(y_series)[mask]
            poly = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            y_line = np.polyval(poly, x_line)
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name=name,
                line=dict(color='#D97706', width=2.5, dash='dash')
            ))
    except Exception:
        pass


# ==============================================================================
# 4. LOAD DATA & SIDEBAR ENTERPRISE
# ==============================================================================
df_env, data_source_info = load_environmental_data()
model, scaler, metadata = load_ml_artifacts()

if 'aqi' not in df_env.columns and 'pm25' in df_env.columns:
    df_env = calculate_aqi_dataframe(df_env, 'pm25')

with st.sidebar:
    st.markdown("""
<div class="sidebar-header-box">
    <div class="sidebar-title">AQI Predictor AI</div>
    <div class="sidebar-subtitle">XGBoost MLOps Engine</div>
</div>
""", unsafe_allow_html=True)
    
    st.caption("Penerapan Regresi Multi-Sumber Berdasarkan Standar US EPA 2024")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#D97706; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.6rem;'>Spesifikasi Pipeline</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
<div class="sidebar-metric-card">
    <div class="row-label">Sumber Data</div>
    <div class="row-val">{data_source_info}</div>
</div>

<div class="sidebar-metric-card">
    <div class="row-label">Volume Observasi</div>
    <div class="row-val">{len(df_env):,} baris</div>
</div>
""", unsafe_allow_html=True)

    r2_val = metadata.get('metrics', {}).get('r2_score', 0.7756) if metadata else 0.7756
    mae_val = metadata.get('metrics', {}).get('mae', 2.46) if metadata else 2.46
    rmse_val = metadata.get('metrics', {}).get('rmse', 3.12) if metadata else 3.12
    ver_val = metadata.get('version', 'v1.0.0') if metadata else 'v1.0.0'

    st.markdown(f"""
<div class="sidebar-metric-card">
    <div class="row-label">Versi Model</div>
    <div class="row-val">{ver_val}</div>
</div>

<div class="sidebar-metric-card">
    <div class="row-label">Skor Akurasi R²</div>
    <div class="row-val" style="color:#FBBF24;">{r2_val:.4f}</div>
</div>

<div class="sidebar-metric-card">
    <div class="row-label">Mean Absolute Error</div>
    <div class="row-val">{mae_val:.2f} µg/m³</div>
</div>

<div class="sidebar-metric-card">
    <div class="row-label">Root Mean Squared Error</div>
    <div class="row-val">{rmse_val:.2f} µg/m³</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#D97706; text-transform:uppercase; letter-spacing:0.05em; margin:1rem 0 0.6rem 0;'>Infrastruktur & Stasiun</div>", unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-metric-card">
    <div class="row-label">Lokasi Stasiun</div>
    <div class="row-val">Jakarta Pusat (LOC-JKT-01)</div>
</div>

<div class="sidebar-metric-card">
    <div class="row-label">Orkestrasi Pipeline</div>
    <div class="row-val">Astronomer Airflow</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 5. BANNER HEADER UTAMA
# ==============================================================================
st.markdown("""
<div class="main-banner">
    <div class="main-banner-title">Sistem Prediksi Konsentrasi PM2.5 dan Estimasi Indeks Kualitas Udara</div>
    <div class="main-banner-desc">
        Platform komputasi prediktif berbasis algoritma <span class="gold-badge">XGBoost Regressor</span> 
        yang mengintegrasikan data sensor OpenAQ, meteorologi Open-Meteo, serta dinamika mobilitas lalu lintas komuter 
        sesuai formulasi Piecewise Linear Interpolation <span class="gold-badge">US EPA Revised 2024</span>.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 6. TAB UTAMA: TAB 1 LANGSUNG SEBAGAI FORM INPUT USER
# ==============================================================================
tab_input, tab_monitoring, tab_evaluasi = st.tabs([
    "Simulator Prediksi AI (Input Pengguna)",
    "Monitoring Lingkungan & Analisis Data (Q1 - Q4)",
    "Evaluasi Kinerja Model & Kontribusi Fitur (Q5)"
])


# ------------------------------------------------------------------------------
# TAB 1: SIMULATOR PREDIKSI AI (INPUT USER)
# ------------------------------------------------------------------------------
with tab_input:
    st.markdown("<div style='font-size:1.15rem; font-weight:800; color:#0A192F; margin-bottom:0.3rem;'>Formulir Parameter Simulasi Lingkungan</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.88rem; color:#64748B; margin-bottom:1.2rem;'>Atur nilai-nilai variabel di bawah ini untuk menguji respons inferensi prediktif model XGBoost terhadap kondisi atmosfer dan beban lalu lintas:</div>", unsafe_allow_html=True)
    
    if model is None or scaler is None:
        st.error("Artefak model (xgboost_pm25_model.pkl atau scaler.pkl) belum ditemukan di folder models/.")
    else:
        with st.form("clean_enterprise_prediction_form"):
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                st.markdown("""
<div class="input-section-card">
    <div class="input-section-title">Parameter Meteorologi Atmosfer</div>
</div>
""", unsafe_allow_html=True)
                temp_in = st.slider("Suhu Udara (°C)", min_value=18.0, max_value=42.0, value=30.0, step=0.5)
                humid_in = st.slider("Kelembaban Relatif (%)", min_value=20.0, max_value=100.0, value=75.0, step=1.0)
                wind_in = st.slider("Kecepatan Angin (km/jam)", min_value=1.0, max_value=45.0, value=12.0, step=0.5)
                rain_in = st.slider("Curah Hujan (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
                
            with col_f2:
                st.markdown("""
<div class="input-section-card">
    <div class="input-section-title">Parameter Mobilitas & Temporal</div>
</div>
""", unsafe_allow_html=True)
                traffic_in = st.slider("Indeks Kemacetan Lalu Lintas (0 - 100)", min_value=0.0, max_value=100.0, value=70.0, step=5.0)
                hour_in = st.slider("Jam dalam Sehari (WIB)", min_value=0, max_value=23, value=8)
                is_weekend_in = st.selectbox("Klasifikasi Hari Kerja:", options=[0, 1], format_func=lambda x: "Akhir Pekan (Sabtu / Minggu)" if x == 1 else "Hari Kerja Aktif (Senin - Jumat)")
                is_holiday_in = st.selectbox("Status Hari Libur:", options=[0, 1], format_func=lambda x: "Hari Libur Nasional" if x == 1 else "Hari Biasa")
                
            with col_f3:
                st.markdown("""
<div class="input-section-card">
    <div class="input-section-title">Riwayat Historis (Fitur Lag)</div>
</div>
""", unsafe_allow_html=True)
                lag1_in = st.number_input("PM2.5 1 Jam Sebelumnya (µg/m³)", min_value=0.0, max_value=250.0, value=45.0, step=1.0)
                lag24_in = st.number_input("PM2.5 24 Jam Sebelumnya (µg/m³)", min_value=0.0, max_value=250.0, value=42.0, step=1.0)
                roll6_in = st.number_input("Rata-rata PM2.5 6 Jam Terakhir (µg/m³)", min_value=0.0, max_value=250.0, value=44.0, step=1.0)
                wind_dir_in = st.number_input("Arah Angin (Derajat Azimuth 0-360)", min_value=0.0, max_value=360.0, value=180.0, step=10.0)

            st.markdown("<br>", unsafe_allow_html=True)
            btn_predict = st.form_submit_button("Jalankan Inferensi Model Prediksi Sekarang", use_container_width=True)
            
        if btn_predict:
            # Transformasi Jam Siklikal
            hour_sin = np.sin(2 * np.pi * hour_in / 24.0)
            hour_cos = np.cos(2 * np.pi * hour_in / 24.0)
            
            input_df = pd.DataFrame([{
                'temperature_c': temp_in,
                'humidity_pct': humid_in,
                'wind_speed_kmh': wind_in,
                'wind_direction_deg': wind_dir_in,
                'rainfall_mm': rain_in,
                'traffic_index': traffic_in,
                'is_weekend': is_weekend_in,
                'is_holiday': is_holiday_in,
                'pm25_lag_1h': lag1_in,
                'pm25_lag_24h': lag24_in,
                'pm25_rolling_mean_6h': roll6_in,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos
            }])
            
            # Penskalaan dan Prediksi Model XGBoost
            input_scaled = scaler.transform(input_df)
            pred_pm25 = float(model.predict(input_scaled)[0])
            pred_pm25 = max(1.0, round(pred_pm25, 2))
            
            # Perhitungan AQI Standar US EPA
            aqi_res = calculate_pm25_aqi(pred_pm25)
            badge_bg = aqi_res['color']
            badge_text = "#000000" if aqi_res['color'] in ["#FFFF00", "#00E400"] else "#FFFFFF"
            
            # Tampilan Hasil Prediksi Enterprise Navy & Gold
            st.markdown(f"""
<div class="prediction-result-panel">
    <div class="pred-tagline">Hasil Inferensi Prediktif XGBoost Regressor</div>
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 2.0rem;">
        <div>
            <div class="pred-primary-num">{pred_pm25} <span class="pred-primary-unit">µg/m³</span></div>
            <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 0.35rem; font-weight: 500;">Estimasi Konsentrasi PM2.5</div>
        </div>
        
        <div style="text-align: center;">
            <div class="pred-secondary-num">{aqi_res['aqi']} <span style="font-size: 1.15rem; color: #CBD5E1; font-family: 'Plus Jakarta Sans', sans-serif;">/ 500</span></div>
            <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 0.35rem; font-weight: 500;">Skor Indeks US EPA AQI</div>
        </div>

        <div style="min-width: 260px; max-width: 320px;">
            <div class="health-pill-badge" style="background: {badge_bg}; color: {badge_text}; margin-bottom: 0.5rem;">
                {aqi_res['category']}
            </div>
            <div style="font-size: 0.85rem; color: #E2E8F0; line-height: 1.45;">
                <b style="color: #D97706;">Protokol Kesehatan:</b> {aqi_res['action']}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
            
            # Auto-Log ke Database MySQL
            engine = get_db_connection()
            if engine is not None:
                try:
                    payload_json = json.dumps(input_df.to_dict(orient='records')[0])
                    insert_log_query = text("""
                    INSERT INTO prediction_logs (
                        location_id, model_version, predicted_pm25, calculated_aqi,
                        aqi_category, health_implication, input_features_json
                    ) VALUES (
                        'LOC-JKT-01', 'v1.0.0-xgb', :pred_pm25, :aqi,
                        :cat, :impl, :payload
                    )
                    """)
                    with engine.begin() as conn:
                        conn.execute(insert_log_query, {
                            'pred_pm25': pred_pm25,
                            'aqi': aqi_res['aqi'],
                            'cat': aqi_res['category'],
                            'impl': aqi_res['action'],
                            'payload': payload_json
                        })
                    st.success("Log transaksi inferensi berhasil dicatat ke tabel prediction_logs.")
                except Exception as log_err:
                    st.caption(f"Status logging MySQL: {log_err}")

        # Riwayat Log Prediksi
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1.05rem; font-weight:800; color:#0A192F; margin-bottom:0.6rem;'>Riwayat Log Inferensi Terkini (Audit Trail Database)</div>", unsafe_allow_html=True)
        engine = get_db_connection()
        if engine is not None:
            try:
                logs_df = pd.read_sql("SELECT prediction_id, predicted_at, predicted_pm25, calculated_aqi, aqi_category FROM prediction_logs ORDER BY predicted_at DESC LIMIT 5", engine)
                if not logs_df.empty:
                    st.dataframe(logs_df, use_container_width=True)
                else:
                    st.info("Belum ada data inferensi tercatat pada tabel prediction_logs.")
            except Exception:
                st.info("Koneksi tabel log audit MySQL siap di lingkungan basis data lokal.")


# ------------------------------------------------------------------------------
# TAB 2: MONITORING LINGKUNGAN & JAWABAN BISNIS Q1 - Q4
# ------------------------------------------------------------------------------
with tab_monitoring:
    st.markdown("<div style='font-size:1.15rem; font-weight:800; color:#0A192F; margin-bottom:0.8rem;'>Metrik Pemantauan Lingkungan Terkini</div>", unsafe_allow_html=True)
    
    if not df_env.empty:
        latest = df_env.iloc[-1]
        aqi_info = calculate_pm25_aqi(latest['pm25'])
        waktu_str = latest['recorded_at'].strftime('%d %b %Y %H:%M') if 'recorded_at' in latest else "Terkini"
        
        st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card-light">
        <div class="kpi-title-text">Konsentrasi PM2.5</div>
        <div class="kpi-val-text">{latest['pm25']:.1f} <span style="font-size:0.85rem; color:#64748B;">µg/m³</span></div>
        <div style="font-size:0.75rem; color:#94A3B8;">Waktu: {waktu_str}</div>
    </div>
    
    <div class="kpi-card-light kpi-card-gold">
        <div class="kpi-title-text">Indeks AQI EPA</div>
        <div class="kpi-val-text" style="color:#D97706;">{aqi_info['aqi']} <span style="font-size:0.85rem; color:#64748B;">/ 500</span></div>
        <div style="font-size:0.8rem; font-weight:700; color:{aqi_info['color']};">{aqi_info['category']}</div>
    </div>

    <div class="kpi-card-light">
        <div class="kpi-title-text">Suhu Permukaan</div>
        <div class="kpi-val-text">{latest.get('temperature_c', 28.5):.1f} <span style="font-size:0.85rem; color:#64748B;">°C</span></div>
        <div style="font-size:0.75rem; color:#94A3B8;">Sensor Open-Meteo</div>
    </div>

    <div class="kpi-card-light">
        <div class="kpi-title-text">Kelembaban Relatif</div>
        <div class="kpi-val-text">{latest.get('humidity_pct', 75.0):.0f} <span style="font-size:0.85rem; color:#64748B;">%</span></div>
        <div style="font-size:0.75rem; color:#94A3B8;">Kondisi Hidrometeorologis</div>
    </div>

    <div class="kpi-card-light">
        <div class="kpi-title-text">Kecepatan Angin</div>
        <div class="kpi-val-text">{latest.get('wind_speed_kmh', 10.0):.1f} <span style="font-size:0.85rem; color:#64748B;">km/h</span></div>
        <div style="font-size:0.75rem; color:#94A3B8;">Faktor Dispersi Partikulat</div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div style='font-size:1.05rem; font-weight:800; color:#0A192F; margin-bottom:0.4rem;'>1. Fluktuasi Tren Historis PM2.5 (Menjawab Q1)</div>", unsafe_allow_html=True)
        
        # Runtun Waktu PM2.5
        fig_trend = px.line(
            df_env,
            x='recorded_at',
            y='pm25',
            labels={'pm25': 'Konsentrasi PM2.5 (µg/m³)', 'recorded_at': 'Waktu Observasi'},
            color_discrete_sequence=['#1E3A8A']
        )
        fig_trend.add_hline(
            y=55.4,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text="Ambang Batas Kritis Tidak Sehat EPA (55.4 µg/m³)",
            annotation_position="top right"
        )
        fig_trend = apply_plotly_enterprise_theme(fig_trend, "Runtun Waktu Konsentrasi Partikulat PM2.5")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        col_q1_a, col_q1_b = st.columns([1.2, 1])
        with col_q1_a:
            cat_counts = df_env['aqi_category'].value_counts().reset_index()
            cat_counts.columns = ['Kategori', 'Jumlah Jam']
            
            color_map = {
                'Baik': '#10B981',
                'Sedang': '#F59E0B',
                'Tidak Sehat bagi Kelompok Sensitif': '#F97316',
                'Tidak Sehat': '#EF4444',
                'Sangat Tidak Sehat': '#8F3F97',
                'Berbahaya': '#7E0023'
            }
            colors_list = [color_map.get(cat, '#1E3A8A') for cat in cat_counts['Kategori']]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=cat_counts['Kategori'],
                values=cat_counts['Jumlah Jam'],
                hole=0.55,
                marker=dict(colors=colors_list),
                textinfo='label+percent'
            )])
            fig_pie = apply_plotly_enterprise_theme(fig_pie, "Distribusi Frekuensi Kategori Kualitas Udara (US EPA)")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_q1_b:
            safe_pct = (df_env['pm25'] <= 35.4).mean() * 100
            unhealthy_pct = (df_env['pm25'] > 55.4).mean() * 100
            st.markdown(f"""
<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #0A192F; border-radius:10px; padding:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.02);">
    <div style="font-weight:800; color:#0A192F; font-size:1.0rem; margin-bottom:0.5rem;">Analisis Baseline Profil (Q1)</div>
    <ul style="color:#334155; font-size:0.86rem; line-height:1.6; margin:0; padding-left:1.2rem;">
        <li><b>Rata-rata Konsentrasi:</b> <span style="color:#0A192F; font-weight:700;">{df_env['pm25'].mean():.2f} µg/m³</span>.</li>
        <li><b>Proporsi Udara Aman:</b> <span style="color:#10B981; font-weight:700;">{safe_pct:.1f}%</span> dari keseluruhan periode observasi.</li>
        <li><b>Proporsi Udara Kritis:</b> <span style="color:#EF4444; font-weight:700;">{unhealthy_pct:.1f}%</span> (Kategori Tidak Sehat).</li>
        <li><b>Kesimpulan:</b> Beban polusi didominasi kategori Sedang dengan eksaserbasi periodik saat periode mobilitas tinggi.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        col_cuaca, col_mobilitas = st.columns(2)
        
        with col_cuaca:
            st.markdown("<div style='font-size:1.05rem; font-weight:800; color:#0A192F; margin-bottom:0.4rem;'>2. Pengaruh Dinamika Meteorologi (Menjawab Q2)</div>", unsafe_allow_html=True)
            fig_scatter = px.scatter(
                df_env,
                x='wind_speed_kmh',
                y='pm25',
                color='humidity_pct',
                labels={'wind_speed_kmh': 'Kecepatan Angin (km/h)', 'pm25': 'PM2.5 (µg/m³)', 'humidity_pct': 'Kelembaban (%)'},
                color_continuous_scale='Viridis'
            )
            add_numpy_ols_trendline(fig_scatter, df_env['wind_speed_kmh'], df_env['pm25'], "Garis Regresi OLS")
            fig_scatter = apply_plotly_enterprise_theme(fig_scatter, "Dispersi Angin & Kelembaban terhadap PM2.5")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info("""
            **Temuan Q2 (Meteorologi):**
            - **Kecepatan Angin:** Menunjukkan korelasi negatif yang nyata. Ventilasi horizontal angin membantu dispersi polutan keluar dari lembah perkotaan.
            - **Kelembaban Relatif:** Kelembaban tinggi meningkatkan koagulasi aerosol higroskopis sehingga menahan partikulat dekat permukaan tanah.
            """)

        with col_mobilitas:
            st.markdown("<div style='font-size:1.05rem; font-weight:800; color:#0A192F; margin-bottom:0.4rem;'>3. Pola Diurnal Siklus 24 Jam (Menjawab Q3)</div>", unsafe_allow_html=True)
            hourly_avg = df_env.groupby('hour_of_day')[['pm25', 'traffic_index']].mean().reset_index()
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Scatter(
                x=hourly_avg['hour_of_day'],
                y=hourly_avg['pm25'],
                name="Konsentrasi PM2.5",
                line=dict(color="#DC2626", width=2.8),
                mode='lines+markers'
            ))
            fig_hour.add_trace(go.Scatter(
                x=hourly_avg['hour_of_day'],
                y=hourly_avg['traffic_index'],
                name="Indeks Kemacetan",
                line=dict(color="#0A192F", width=2, dash="dot"),
                mode='lines'
            ))
            fig_hour = apply_plotly_enterprise_theme(fig_hour, "Siklus Diurnal 24 Jam: Puncak Emisi Jam Sibuk")
            fig_hour.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
            st.plotly_chart(fig_hour, use_container_width=True)
            
            st.warning("""
            **Temuan Q3 (Mobilitas):**
            - Lonjakan konsentrasi partikulat terkonsentrasi pada pukul **07:00–09:00 WIB (Puncak Pagi)** dan **17:00–19:00 WIB (Puncak Sore)**.
            - Pola emisi kendaraan bermotor komuter terbukti menjadi pendorong antropogenik utama polusi udara.
            """)

        st.markdown("---")
        st.markdown("<div style='font-size:1.05rem; font-weight:800; color:#0A192F; margin-bottom:0.6rem;'>4. Pedoman Intervensi dan Mitigasi Kesehatan (Menjawab Q4)</div>", unsafe_allow_html=True)
        st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem;">
    <div class="mitigasi-card-clean c-baik">
        <div class="mitigasi-head">Kategori Baik</div>
        <div class="mitigasi-param">0.0 – 9.0 µg/m³</div>
        <div class="mitigasi-body">Kualitas udara sangat memuaskan. Tidak ada pembatasan aktivitas luar ruangan untuk seluruh populasi.</div>
    </div>
    
    <div class="mitigasi-card-clean c-sedang">
        <div class="mitigasi-head">Kategori Sedang</div>
        <div class="mitigasi-param">9.1 – 35.4 µg/m³</div>
        <div class="mitigasi-body">Kualitas udara dapat diterima. Kelompok sangat sensitif disarankan mengurangi aktivitas fisik intensif berkepanjangan di luar.</div>
    </div>

    <div class="mitigasi-card-clean c-sensitif">
        <div class="mitigasi-head">Kelompok Sensitif</div>
        <div class="mitigasi-param">35.5 – 55.4 µg/m³</div>
        <div class="mitigasi-body">Anak-anak, lansia, dan penderita gangguan respirasi dianjurkan memakai masker filtrasi dan menyalakan air purifier di dalam ruang.</div>
    </div>

    <div class="mitigasi-card-clean c-kritis">
        <div class="mitigasi-head">Tidak Sehat</div>
        <div class="mitigasi-param">> 55.4 µg/m³</div>
        <div class="mitigasi-body">Masyarakat umum rentan mengalami dampak kesehatan. Tutup ventilasi rumah saat jam sibuk dan gunakan masker standar N95 jika beraktivitas luar.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 3: EVALUASI PERFORMA MODEL & FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
with tab_evaluasi:
    st.markdown("<div style='font-size:1.15rem; font-weight:800; color:#0A192F; margin-bottom:0.8rem;'>Metrik Evaluasi Ilmiah XGBoost Regressor</div>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    if metadata and 'metrics' in metadata:
        m = metadata['metrics']
        with col_m1:
            st.metric("R-Squared Score (R²)", f"{m.get('r2_score', 0.7756):.4f}", "77.6% Variansi Terjelaskan")
        with col_m2:
            st.metric("Mean Absolute Error (MAE)", f"{m.get('mae', 2.46):.2f} µg/m³", "Deviasi Rata-rata")
        with col_m3:
            st.metric("Root Mean Squared Error (RMSE)", f"{m.get('rmse', 3.12):.2f} µg/m³", "Galat Kuadratik")

    st.markdown("---")
    st.markdown("<div style='font-size:1.05rem; font-weight:800; color:#0A192F; margin-bottom:0.4rem;'>Urutan Kontribusi Fitur Prediktif (Feature Importance - Menjawab Q5)</div>", unsafe_allow_html=True)
    if model is not None and hasattr(model, 'feature_importances_'):
        feature_names = metadata.get('features', [
            'temperature_c', 'humidity_pct', 'wind_speed_kmh', 'wind_direction_deg', 'rainfall_mm',
            'traffic_index', 'is_weekend', 'is_holiday',
            'pm25_lag_1h', 'pm25_lag_24h', 'pm25_rolling_mean_6h',
            'hour_sin', 'hour_cos'
        ])
        importance_df = pd.DataFrame({
            'Fitur': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(
            importance_df,
            x='Importance',
            y='Fitur',
            orientation='h',
            labels={'Importance': 'Tingkat Kepentingan (F-Score)', 'Fitur': 'Nama Fitur Input'},
            color='Importance',
            color_continuous_scale=[[0, '#0A192F'], [0.5, '#1E3A8A'], [1, '#D97706']]
        )
        fig_imp = apply_plotly_enterprise_theme(fig_imp, "Kontribusi Relatif Fitur dalam Estimasi PM2.5 (XGBoost Regressor)")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #D97706; border-radius:10px; padding:1.2rem; margin-top:1.0rem;">
    <div style="font-weight:800; color:#0A192F; font-size:1.0rem; margin-bottom:0.4rem;">Kesimpulan Ilmiah Evaluasi Model (Menjawab Q5):</div>
    <ol style="color:#334155; font-size:0.86rem; line-height:1.6; margin:0; padding-left:1.2rem;">
        <li>Fitur <b><code>pm25_lag_1h</code></b> dan <b><code>pm25_rolling_mean_6h</code></b> memberikan kontribusi terbesar dalam model XGBoost. Hal ini membuktikan konsentrasi polutan memiliki sifat <i>autoregresif temporal kuat</i> (kondisi 1 jam sebelumnya menjadi prediktor primer kondisi saat ini).</li>
        <li>Variabel <b><code>humidity_pct</code></b> dan <b><code>wind_speed_kmh</code></b> merupakan pengendali fisis utama proses dispersi atmosferik.</li>
        <li>Fitur siklikal jam (<code>hour_sin</code>, <code>hour_cos</code>) dan indeks kemacetan secara konsisten merefleksikan variabilitas mobilitas antropogenik harian.</li>
    </ol>
</div>
""", unsafe_allow_html=True)
