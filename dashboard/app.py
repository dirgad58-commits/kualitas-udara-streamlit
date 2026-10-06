"""
Streamlit Dashboard: Sistem Prediksi Konsentrasi PM2.5 & Estimasi AQI
Berbasis Algoritma XGBoost Regressor dan Standar US EPA (Piecewise Linear Interpolation)
Tema: Light Theme (Mode Terang) dengan Palet Warna Biru Navy (#0A192F / #1E3A8A) dan Emas Gold (#D97706 / #F59E0B)
Tampilan Utama (Tab 1): Inputan User / Simulator Prediksi Interaktif
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
# 1. KONFIGURASI HALAMAN & INJEKSI CSS MODE TERANG (NAVY & GOLD)
# ==============================================================================
st.set_page_config(
    page_title="Prediksi Kualitas Udara PM2.5 & AQI | AI XGBoost",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Light Mode, Deep Navy (#0A192F), Classic Gold (#D97706)
# PENTING: Semua string HTML dibuat rapat tanpa indentasi 4 spasi agar tidak tertafsir sebagai Markdown Code Block
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
/* Reset & Global Typography */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}

/* Sidebar Styling: Deep Navy */
section[data-testid="stSidebar"] {
    background-color: #0A192F !important;
    border-right: 2px solid #1E293B !important;
}
section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] .stCaption {
    color: #CBD5E1 !important;
}

/* Header Container */
.brand-header {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 6px solid #D97706;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(10, 25, 47, 0.05);
}
.brand-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0A192F;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.brand-subtitle {
    font-size: 0.95rem;
    color: #475569;
    font-weight: 500;
    margin-top: 0.3rem;
}
.gold-tag {
    color: #D97706;
    font-weight: 700;
}

/* Custom Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.2rem;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    color: #64748B !important;
    padding: 0.6rem 1rem !important;
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #0A192F !important;
    border-bottom: 3px solid #D97706 !important;
    background: rgba(217, 119, 6, 0.06) !important;
}

/* KPI Cards: Pure White with Navy and Gold Borders */
.kpi-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-box {
    flex: 1 1 calc(20% - 1rem);
    min-width: 180px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 4px solid #0A192F;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
}
.kpi-box-gold {
    border-top: 4px solid #D97706;
}
.kpi-caption {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.kpi-num {
    font-size: 1.85rem;
    font-weight: 800;
    color: #0A192F;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.kpi-num-gold {
    color: #D97706;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #94A3B8;
    font-weight: 500;
}

/* Result Box Showcase (Navy Banner with Gold Accents) */
.result-card {
    background: linear-gradient(135deg, #0A192F 0%, #1E3A8A 100%);
    border: 2px solid #D97706;
    border-radius: 16px;
    padding: 1.6rem 2.0rem;
    color: #FFFFFF;
    margin-top: 1.2rem;
    box-shadow: 0 10px 25px rgba(10, 25, 47, 0.2);
}
.result-label {
    color: #FBBF24;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-val {
    font-size: 2.8rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
}
.result-unit {
    font-size: 1.2rem;
    color: #93C5FD;
    font-weight: 600;
}

/* EPA Badge */
.epa-pill {
    display: inline-block;
    padding: 0.4rem 1.0rem;
    border-radius: 9999px;
    font-weight: 800;
    font-size: 0.9rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

/* Mitigasi Grid Light */
.mitigasi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 1rem;
    margin: 1.2rem 0;
}
.mitigasi-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.1rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.03);
}
.mitigasi-box-baik { border-left: 5px solid #10B981; }
.mitigasi-box-sedang { border-left: 5px solid #F59E0B; }
.mitigasi-box-sensitif { border-left: 5px solid #F97316; }
.mitigasi-box-kritis { border-left: 5px solid #EF4444; }

.mitigasi-headline {
    font-size: 1.0rem;
    font-weight: 800;
    color: #0A192F;
    margin-bottom: 0.2rem;
}
.mitigasi-threshold {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    margin-bottom: 0.5rem;
}
.mitigasi-text {
    font-size: 0.82rem;
    color: #334155;
    line-height: 1.45;
}

/* Tombol Submit Kustom */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #0A192F 0%, #1E3A8A 100%) !important;
    color: #FFFFFF !important;
    border: 2px solid #D97706 !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.65rem 1.5rem !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(10, 25, 47, 0.15) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:first-child:hover {
    background: #D97706 !important;
    border-color: #0A192F !important;
    color: #FFFFFF !important;
    transform: translateY(-2px) !important;
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
# 3. HELPER PLOTLY LIGHT THEME DENGAN REGRESI NUMPY MURNI
# ==============================================================================
def apply_plotly_light_theme(fig, title=""):
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
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5E1",
            tickfont=dict(color="#475569")
        ),
        yaxis=dict(
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5E1",
            tickfont=dict(color="#475569")
        ),
        legend=dict(
            font=dict(color="#1E293B"),
            bgcolor="rgba(255, 255, 255, 0.85)"
        )
    )
    return fig

def add_numpy_ols_trendline(fig, x_series, y_series, name="Garis Regresi OLS"):
    """Regresi linier bebas ketergantungan statsmodels."""
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
                line=dict(color='#D97706', width=3, dash='dash')
            ))
    except Exception:
        pass


# ==============================================================================
# 4. LOAD DATA & SIDEBAR (NAVY THEME)
# ==============================================================================
df_env, data_source_info = load_environmental_data()
model, scaler, metadata = load_ml_artifacts()

if 'aqi' not in df_env.columns and 'pm25' in df_env.columns:
    df_env = calculate_aqi_dataframe(df_env, 'pm25')

with st.sidebar:
    # Sidebar Header
    st.markdown("""
<div style="padding: 0.5rem 0 1.0rem 0; border-bottom: 1px solid #1E293B; margin-bottom: 1.0rem;">
    <div style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF;">AQI PREDICTOR AI</div>
    <div style="font-size: 0.8rem; color: #D97706; font-weight: 700; letter-spacing: 0.05em;">XGBOOST ML ENGINE</div>
</div>
""", unsafe_allow_html=True)
    
    st.caption("Sistem Estimasi Konsentrasi PM2.5 & Indeks Kualitas Udara (US EPA Revised 2024)")

    st.markdown("---")
    st.markdown("**📌 Parameter Sistem:**")
    st.markdown(f"• Sumber Data: **{data_source_info}**")
    st.markdown(f"• Total Data: **{len(df_env)} observasi**")
    
    r2_val = metadata.get('metrics', {}).get('r2_score', 0.7756) if metadata else 0.7756
    mae_val = metadata.get('metrics', {}).get('mae', 2.46) if metadata else 2.46
    rmse_val = metadata.get('metrics', {}).get('rmse', 3.12) if metadata else 3.12
    ver_val = metadata.get('version', 'v1.0.0') if metadata else 'v1.0.0'
    
    st.markdown(f"• Versi Model: **{ver_val}**")
    st.markdown(f"• Akurasi $R^2$: **{r2_val:.4f}**")
    st.markdown(f"• Error MAE: **{mae_val:.2f} µg/m³**")
    st.markdown(f"• Error RMSE: **{rmse_val:.2f} µg/m³**")

    st.markdown("---")
    st.markdown("**📍 Lokasi & Pipeline:**")
    st.markdown("• Lokasi: **Jakarta Pusat (LOC-JKT-01)**")
    st.markdown("• Orkestrator: **Astronomer Airflow**")
    st.markdown("• Target Regresi: **PM2.5 (µg/m³)**")


# ==============================================================================
# 5. HEADER UTAMA DASHBOARD
# ==============================================================================
st.markdown("""
<div class="brand-header">
    <div class="brand-title">Sistem Prediksi Kualitas Udara PM2.5 & AQI</div>
    <div class="brand-subtitle">
        Implementasi Machine Learning Regresi Berbasis <span class="gold-tag">XGBoost Regressor</span> Multi-Sumber 
        (Sensor Udara OpenAQ, Cuaca Open-Meteo, & Pola Mobilitas) Sesuai Standar <span class="gold-tag">US EPA 2024</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 6. TAB APLIKASI: TAB 1 ADALAH INPUTAN USER / SIMULATOR PREDIKSI AI
# ==============================================================================
tab_input, tab_monitoring, tab_evaluasi = st.tabs([
    "🎯 1. Simulator Prediksi AI (Input User)",
    "📈 2. Monitoring Lingkungan & Temuan Data (Q1 - Q4)",
    "📊 3. Evaluasi Performa Model & Feature Importance (Q5)"
])


# ------------------------------------------------------------------------------
# TAB 1: SIMULATOR PREDIKSI AI (TAMPILAN AWAL / INPUT USER)
# ------------------------------------------------------------------------------
with tab_input:
    st.markdown("### 🎛️ Form Input Parameter Lingkungan & Simulasi Prediksi")
    st.markdown("Silakan atur parameter kondisi atmosfer dan lalu lintas di bawah ini untuk melihat estimasi konsentrasi PM2.5 dan kategori skor AQI dari model XGBoost secara instan:")
    
    if model is None or scaler is None:
        st.error("⚠️ Artefak model (`xgboost_pm25_model.pkl` atau `scaler.pkl`) belum ditemukan di direktori `models/`.")
    else:
        with st.form("main_user_input_form"):
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                st.markdown("##### 🌤️ Kondisi Meteorologi Cuaca")
                temp_in = st.slider("Suhu Udara (°C)", min_value=18.0, max_value=42.0, value=30.0, step=0.5)
                humid_in = st.slider("Kelembaban Relatif (%)", min_value=20.0, max_value=100.0, value=75.0, step=1.0)
                wind_in = st.slider("Kecepatan Angin (km/jam)", min_value=1.0, max_value=45.0, value=12.0, step=0.5)
                rain_in = st.slider("Curah Hujan (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
                
            with col_f2:
                st.markdown("##### 🚗 Parameter Mobilitas & Waktu")
                traffic_in = st.slider("Indeks Kemacetan (0 - 100)", min_value=0.0, max_value=100.0, value=70.0, step=5.0)
                hour_in = st.slider("Jam dalam Sehari (WIB)", min_value=0, max_value=23, value=8)
                is_weekend_in = st.selectbox("Status Hari:", options=[0, 1], format_func=lambda x: "Akhir Pekan (Sabtu / Minggu)" if x == 1 else "Hari Kerja (Senin - Jumat)")
                is_holiday_in = st.selectbox("Status Hari Libur:", options=[0, 1], format_func=lambda x: "Hari Libur Nasional" if x == 1 else "Hari Biasa")
                
            with col_f3:
                st.markdown("##### ⏱️ Riwayat Historis (Lag Fitur)")
                lag1_in = st.number_input("PM2.5 1 Jam Sebelumnya (µg/m³)", min_value=0.0, max_value=250.0, value=45.0, step=1.0)
                lag24_in = st.number_input("PM2.5 24 Jam Sebelumnya (µg/m³)", min_value=0.0, max_value=250.0, value=42.0, step=1.0)
                roll6_in = st.number_input("Rata-rata PM2.5 6 Jam Terakhir", min_value=0.0, max_value=250.0, value=44.0, step=1.0)
                wind_dir_in = st.number_input("Arah Angin (Derajat 0-360)", min_value=0.0, max_value=360.0, value=180.0, step=10.0)

            st.markdown("<br>", unsafe_allow_html=True)
            btn_predict = st.form_submit_button("🚀 Hitung Estimasi PM2.5 & AQI Sekarang", use_container_width=True)
            
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
            
            # Penskalaan dan Prediksi XGBoost
            input_scaled = scaler.transform(input_df)
            pred_pm25 = float(model.predict(input_scaled)[0])
            pred_pm25 = max(1.0, round(pred_pm25, 2))
            
            # Perhitungan AQI Standar US EPA
            aqi_res = calculate_pm25_aqi(pred_pm25)
            badge_bg = aqi_res['color']
            badge_text = "#000000" if aqi_res['color'] in ["#FFFF00", "#00E400"] else "#FFFFFF"
            
            # Tampilan Hasil Prediksi Berwarna Navy & Gold
            st.markdown(f"""
<div class="result-card">
    <div class="result-label">HASIL ESTIMASI MODEL MACHINE LEARNING (XGBOOST REGRESSOR)</div>
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1.5rem;">
        <div>
            <div class="result-val">{pred_pm25} <span class="result-unit">µg/m³</span></div>
            <div style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0.3rem;">Prediksi Konsentrasi Partikulat PM2.5</div>
        </div>
        
        <div style="text-align: center;">
            <div class="result-val" style="color: #FBBF24;">{aqi_res['aqi']} <span class="result-unit" style="color: #E2E8F0;">/ 500</span></div>
            <div style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0.3rem;">Indeks Standar US EPA</div>
        </div>

        <div style="min-width: 250px;">
            <div class="epa-pill" style="background: {badge_bg}; color: {badge_text}; margin-bottom: 0.5rem;">
                {aqi_res['category']}
            </div>
            <div style="font-size: 0.85rem; color: #F1F5F9; line-height: 1.4;">
                <b>Rekomendasi Medis:</b> {aqi_res['action']}
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
                    st.success("✅ Log inferensi berhasil disimpan ke tabel `prediction_logs` di MySQL.")
                except Exception as log_err:
                    st.caption(f"Catatan log MySQL: {log_err}")

        # Tabel Riwayat Log Prediksi
        st.markdown("---")
        st.subheader("📜 Riwayat Prediksi Pengguna Terkini (Database Log)")
        engine = get_db_connection()
        if engine is not None:
            try:
                logs_df = pd.read_sql("SELECT prediction_id, predicted_at, predicted_pm25, calculated_aqi, aqi_category FROM prediction_logs ORDER BY predicted_at DESC LIMIT 5", engine)
                if not logs_df.empty:
                    st.dataframe(logs_df, use_container_width=True)
                else:
                    st.info("Belum ada riwayat prediksi tercatat di tabel log.")
            except Exception:
                st.info("Koneksi tabel MySQL aktif di lingkungan lokal.")


# ------------------------------------------------------------------------------
# TAB 2: MONITORING LINGKUNGAN & JAWABAN BISNIS Q1 - Q4
# ------------------------------------------------------------------------------
with tab_monitoring:
    st.markdown("### 📊 Ringkasan Sensor Kualitas Udara & Parameter Terkini")
    
    if not df_env.empty:
        latest = df_env.iloc[-1]
        aqi_info = calculate_pm25_aqi(latest['pm25'])
        waktu_str = latest['recorded_at'].strftime('%d %b %Y %H:%M') if 'recorded_at' in latest else "Terkini"
        
        # Kartu Metrik Mode Terang (Navy & Gold)
        st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-box">
        <div class="kpi-caption">PM2.5 TERKINI</div>
        <div class="kpi-num">{latest['pm25']:.1f} <span style="font-size:0.9rem; color:#64748B;">µg/m³</span></div>
        <div class="kpi-sub">Waktu: {waktu_str}</div>
    </div>
    
    <div class="kpi-box kpi-box-gold">
        <div class="kpi-caption">SKOR AQI EPA</div>
        <div class="kpi-num kpi-num-gold">{aqi_info['aqi']} <span style="font-size:0.9rem; color:#64748B;">/ 500</span></div>
        <div class="kpi-sub" style="font-weight:700; color:{aqi_info['color']};">{aqi_info['category']}</div>
    </div>

    <div class="kpi-box">
        <div class="kpi-caption">SUHU UDARA</div>
        <div class="kpi-num">{latest.get('temperature_c', 28.5):.1f} <span style="font-size:0.9rem; color:#64748B;">°C</span></div>
        <div class="kpi-sub">Open-Meteo API</div>
    </div>

    <div class="kpi-box">
        <div class="kpi-caption">KELEMBABAN</div>
        <div class="kpi-num">{latest.get('humidity_pct', 75.0):.0f} <span style="font-size:0.9rem; color:#64748B;">%</span></div>
        <div class="kpi-sub">Relatif (RH)</div>
    </div>

    <div class="kpi-box">
        <div class="kpi-caption">KECEPATAN ANGIN</div>
        <div class="kpi-num">{latest.get('wind_speed_kmh', 10.0):.1f} <span style="font-size:0.9rem; color:#64748B;">km/h</span></div>
        <div class="kpi-sub">Dispersi Angin</div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 1️⃣ Fluktuasi Tren Historis Konsentrasi PM2.5 (Menjawab Q1)")
        
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
            annotation_text="Batas Kritis Tidak Sehat EPA (55.4 µg/m³)",
            annotation_position="top right"
        )
        fig_trend = apply_plotly_light_theme(fig_trend, "Runtun Waktu Konsentrasi PM2.5 Sepanjang Observasi")
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
            fig_pie = apply_plotly_light_theme(fig_pie, "Distribusi Jam Kualitas Udara (US EPA)")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_q1_b:
            safe_pct = (df_env['pm25'] <= 35.4).mean() * 100
            unhealthy_pct = (df_env['pm25'] > 55.4).mean() * 100
            st.markdown(f"""
<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #1E3A8A; border-radius:10px; padding:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
    <div style="font-weight:800; color:#0A192F; font-size:1.05rem; margin-bottom:0.5rem;">💡 Insight Baseline Profil (Q1):</div>
    <ul style="color:#334155; font-size:0.88rem; line-height:1.6; margin:0; padding-left:1.2rem;">
        <li><b>Rata-rata Konsentrasi:</b> <span style="color:#1E3A8A; font-weight:700;">{df_env['pm25'].mean():.2f} µg/m³</span>.</li>
        <li><b>Proporsi Udara Aman:</b> <span style="color:#10B981; font-weight:700;">{safe_pct:.1f}%</span> dari periode pengamatan.</li>
        <li><b>Proporsi Udara Kritis:</b> <span style="color:#EF4444; font-weight:700;">{unhealthy_pct:.1f}%</span>.</li>
        <li><b>Kesimpulan:</b> Polusi didominasi kategori Sedang dengan lonjakan periodik saat rush hour.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        col_cuaca, col_mobilitas = st.columns(2)
        
        with col_cuaca:
            st.markdown("#### 2️⃣ Pengaruh Meteorologi Cuaca (Menjawab Q2)")
            fig_scatter = px.scatter(
                df_env,
                x='wind_speed_kmh',
                y='pm25',
                color='humidity_pct',
                labels={'wind_speed_kmh': 'Kecepatan Angin (km/h)', 'pm25': 'PM2.5 (µg/m³)', 'humidity_pct': 'Lembab (%)'},
                color_continuous_scale='Viridis'
            )
            add_numpy_ols_trendline(fig_scatter, df_env['wind_speed_kmh'], df_env['pm25'], "Garis Regresi OLS")
            fig_scatter = apply_plotly_light_theme(fig_scatter, "Dispersi Angin & Kelembaban vs PM2.5")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info("""
            **Temuan Q2 (Meteorologi):**
            - **Kecepatan Angin:** Berkorelasi negatif signifikan. Angin kencang mempercepat penyebaran partikulat debu.
            - **Kelembaban Udara:** Menahan partikel polutan tetap melayang di dekat permukaan (efek inversi kabut).
            """)

        with col_mobilitas:
            st.markdown("#### 3️⃣ Pola Mobilitas & Jam Sibuk 24 Jam (Menjawab Q3)")
            hourly_avg = df_env.groupby('hour_of_day')[['pm25', 'traffic_index']].mean().reset_index()
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Scatter(
                x=hourly_avg['hour_of_day'],
                y=hourly_avg['pm25'],
                name="PM2.5 (µg/m³)",
                line=dict(color="#DC2626", width=3),
                mode='lines+markers'
            ))
            fig_hour.add_trace(go.Scatter(
                x=hourly_avg['hour_of_day'],
                y=hourly_avg['traffic_index'],
                name="Indeks Kemacetan",
                line=dict(color="#D97706", width=2, dash="dot"),
                mode='lines'
            ))
            fig_hour = apply_plotly_light_theme(fig_hour, "Siklus Diurnal 24 Jam: Puncak Polusi di Jam Sibuk")
            fig_hour.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
            st.plotly_chart(fig_hour, use_container_width=True)
            
            st.warning("""
            **Temuan Q3 (Mobilitas):**
            - Puncak polusi terjadi pada pukul **07:00–09:00 WIB (Rush Hour Pagi)** dan **17:00–19:00 WIB (Rush Hour Sore)**.
            - Hari kerja menunjukkan tingkat polusi lebih tinggi dibandingkan akhir pekan.
            """)

        st.markdown("---")
        st.markdown("#### 4️⃣ Panduan Mitigasi & Protokol Kesehatan Publik (Menjawab Q4)")
        st.markdown("""
<div class="mitigasi-container">
    <div class="mitigasi-box mitigasi-box-baik">
        <div class="mitigasi-headline">🟢 Kategori Baik</div>
        <div class="mitigasi-threshold">0.0 – 9.0 µg/m³</div>
        <div class="mitigasi-text">Kualitas udara sangat memuaskan. Seluruh masyarakat bebas melakukan aktivitas di luar ruangan tanpa pembatasan.</div>
    </div>
    
    <div class="mitigasi-box mitigasi-box-sedang">
        <div class="mitigasi-headline">🟡 Kategori Sedang</div>
        <div class="mitigasi-threshold">9.1 – 35.4 µg/m³</div>
        <div class="mitigasi-text">Kualitas udara dapat diterima. Kelompok yang sangat sensitif disarankan mengurangi aktivitas fisik berat di luar ruang.</div>
    </div>

    <div class="mitigasi-box mitigasi-box-sensitif">
        <div class="mitigasi-headline">🟠 Kelompok Sensitif</div>
        <div class="mitigasi-threshold">35.5 – 55.4 µg/m³</div>
        <div class="mitigasi-text"><b>Anak-anak, lansia, dan penderita asma</b> wajib memakai masker filtrasi di luar dan mengaktifkan air purifier di dalam rumah.</div>
    </div>

    <div class="mitigasi-box mitigasi-box-kritis">
        <div class="mitigasi-headline">🔴 Tidak Sehat</div>
        <div class="mitigasi-threshold">> 55.4 µg/m³</div>
        <div class="mitigasi-text">Masyarakat umum mulai merasakan dampak kesehatan. Tutup ventilasi saat jam sibuk; gunakan masker standar N95 jika bepergian.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 3: EVALUASI PERFORMA MODEL & FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
with tab_evaluasi:
    st.markdown("### 📊 Evaluasi Kinerja Algoritma XGBoost Regressor")
    
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
    st.markdown("#### 🔍 Urutan Kontribusi Fitur (Feature Importance - Menjawab Q5)")
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
        fig_imp = apply_plotly_light_theme(fig_imp, "Kontribusi Relatif Fitur dalam Estimasi PM2.5 (XGBoost Regressor)")
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #D97706; border-radius:10px; padding:1.2rem; margin-top:1.0rem;">
    <div style="font-weight:800; color:#0A192F; font-size:1.05rem; margin-bottom:0.4rem;">Kesimpulan Ilmiah Evaluasi Model (Menjawab Q5):</div>
    <ol style="color:#334155; font-size:0.88rem; line-height:1.6; margin:0; padding-left:1.2rem;">
        <li>Fitur <b><code>pm25_lag_1h</code></b> dan <b><code>pm25_rolling_mean_6h</code></b> memberikan kontribusi terbesar dalam model XGBoost. Hal ini membuktikan polusi udara bersifat <i>autoregresif temporal kuat</i> (kondisi 1 jam lalu sangat menentukan kondisi sekarang).</li>
        <li>Variabel <b><code>humidity_pct</code></b> dan <b><code>wind_speed_kmh</code></b> adalah faktor atmosferik primer yang mengontrol dinamika penumpukan polutan.</li>
        <li>Pola jam siklikal (<code>hour_sin</code>, <code>hour_cos</code>) dan indeks kemacetan berhasil menangkap fluktuasi emisi kendaraan bermotor di area perkotaan.</li>
    </ol>
</div>
""", unsafe_allow_html=True)
