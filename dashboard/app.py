"""
Streamlit Dashboard: Sistem Prediksi Konsentrasi PM2.5 & Estimasi AQI
Berbasis Algoritma XGBoost Regressor dan Standar US EPA (Piecewise Linear Interpolation)
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

# Tambahkan root proyek ke sys.path untuk import src.aqi_calculator
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.aqi_calculator import calculate_pm25_aqi, calculate_aqi_dataframe, EPA_PM25_BREAKPOINTS

# ==============================================================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS
# ==============================================================================
st.set_page_config(
    page_title="AQI PM2.5 AI Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        color: #FFFFFF;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. HELPER KONEKSI DATABASE & PEMUATAN MODEL
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
            eng = create_engine(uri, pool_recycle=3600, connect_args={"connect_timeout": 3})
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
        except Exception as e:
            pass
            
    # Fallback to local CSV
    csv_path = os.path.join(ROOT_DIR, "data", "master_feature_store.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['recorded_at'] = pd.to_datetime(df['recorded_at'])
        return df, "CSV Backup Storage"
        
    # Dummy minimal if not found
    return pd.DataFrame(), "No Data"

@st.cache_resource
def load_ml_artifacts():
    model_path = os.path.join(ROOT_DIR, "models", "xgboost_pm25_model.pkl")
    scaler_path = os.path.join(ROOT_DIR, "models", "scaler.pkl")
    meta_path = os.path.join(ROOT_DIR, "models", "model_metadata.json")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        return model, scaler, metadata
    return None, None, None

# Load Resource
df_env, data_source_info = load_environmental_data()
model, scaler, metadata = load_ml_artifacts()

# Tambahkan kalkulasi AQI ke DataFrame observasi
if not df_env.empty and 'pm25' in df_env.columns:
    df_env = calculate_aqi_dataframe(df_env, pm25_col='pm25')


# ==============================================================================
# 3. SIDEBAR INFORMASI PROYEK & KONTROL
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/air-element.png", width=110)
    st.title("Proyek Data Science")
    st.markdown("**Prediksi PM2.5 & Estimasi AQI**")
    st.caption("Algoritma: **XGBoost Regressor**  \nStandar: **US EPA Revised 2024**")
    
    st.divider()
    st.subheader("ℹ️ Informasi Sistem")
    st.markdown(f"**Sumber Data:** `{data_source_info}`")
    st.markdown(f"**Total Baris Data:** `{len(df_env)} observasi`")
    if metadata:
        st.markdown(f"**Versi Model:** `{metadata.get('version', 'v1.0.0')}`")
        st.markdown(f"**Akurasi R² Score:** `{metadata.get('metrics', {}).get('r2_score', 0):.4f}`")
        st.markdown(f"**Error MAE:** `{metadata.get('metrics', {}).get('mae', 0):.2f} µg/m³`")
    
    st.divider()
    st.subheader("📍 Lokasi Observasi")
    st.markdown("**Stasiun:** `LOC-JKT-01 (Jakarta Pusat)`")
    st.markdown("**Orkestrator:** `Astronomer Airflow (Astro CLI)`")


# ==============================================================================
# 4. HEADER UTAMA & KARTU METRIK REAL-TIME
# ==============================================================================
st.markdown('<div class="main-header">🌫️ Sistem Prediksi Konsentrasi PM2.5 & Estimasi AQI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Penerapan Machine Learning Regresi Berbasis Data Multi-Sumber (Sensor Kualitas Udara, Meteorologi, & Mobilitas Lalu Lintas)</div>', unsafe_allow_html=True)

# Kartu Metrik Terkini
if not df_env.empty:
    latest = df_env.iloc[-1]
    aqi_info = calculate_pm25_aqi(latest['pm25'])
    
    c1, c2, c3, c4, c5 = st.columns([1.5, 1.5, 1.2, 1.2, 1.2])
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748B; font-size:0.9rem;">KONSENTRASI PM2.5 TERKINI</span>
            <h2 style="color:#0F172A; margin:0.3rem 0;">{latest['pm25']:.1f} <span style="font-size:1rem; color:#64748B;">µg/m³</span></h2>
            <span style="font-size:0.8rem; color:#94A3B8;">Waktu: {latest['recorded_at'].strftime('%d %b %Y %H:%M')}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        badge_bg = aqi_info['color']
        text_color = "#000000" if aqi_info['color'] in ["#FFFF00", "#00E400"] else "#FFFFFF"
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748B; font-size:0.9rem;">SKOR AQI (STANDAR US EPA)</span>
            <h2 style="color:#0F172A; margin:0.3rem 0;">{aqi_info['aqi']} <span style="font-size:1rem; color:#64748B;">/ 500</span></h2>
            <div class="risk-badge" style="background:{badge_bg}; color:{text_color};">{aqi_info['category']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748B; font-size:0.9rem;">SUHU UDARA</span>
            <h2 style="color:#0284C7; margin:0.3rem 0;">{latest.get('temperature_c', 28.5):.1f}°C</h2>
            <span style="font-size:0.8rem; color:#94A3B8;">Open-Meteo API</span>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748B; font-size:0.9rem;">KELEMBABAN</span>
            <h2 style="color:#0D9488; margin:0.3rem 0;">{latest.get('humidity_pct', 75.0):.0f}%</h2>
            <span style="font-size:0.8rem; color:#94A3B8;">Relatif (%)</span>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#64748B; font-size:0.9rem;">KECEPATAN ANGIN</span>
            <h2 style="color:#6366F1; margin:0.3rem 0;">{latest.get('wind_speed_kmh', 10.0):.1f}</h2>
            <span style="font-size:0.8rem; color:#94A3B8;">km / jam</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# 5. TAB APLIKASI
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Monitoring Lingkungan & Temuan Data (Q1 - Q4)",
    "🤖 Simulator Prediksi Interaktif AI XGBoost (Q5)",
    "📊 Evaluasi Akurasi Model & Feature Importance"
])


# ------------------------------------------------------------------------------
# TAB 1: MONITORING LINGKUNGAN & JAWABAN Q1 - Q4
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Analisis Eksploratif Multi-Sumber Kualitas Udara")
    
    # [Q1]: Tren Historis PM2.5 & AQI
    st.markdown("### 1️⃣ Fluktuasi Tren Historis Konsentrasi PM2.5 & AQI (Menjawab Q1)")
    if not df_env.empty:
        fig_trend = px.line(
            df_env,
            x='recorded_at',
            y='pm25',
            title='Runtun Waktu Konsentrasi Partikulat PM2.5 (µg/m³)',
            labels={'pm25': 'Konsentrasi PM2.5 (µg/m³)', 'recorded_at': 'Waktu'},
            color_discrete_sequence=['#2563EB']
        )
        fig_trend.add_hline(y=55.4, line_dash="dash", line_color="#DC2626", annotation_text="Batas Kritis Tidak Sehat EPA (55.4 µg/m³)")
        fig_trend.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Proporsi Kategori AQI
        col_q1_a, col_q1_b = st.columns(2)
        with col_q1_a:
            cat_counts = df_env['aqi_category'].value_counts().reset_index()
            cat_counts.columns = ['Kategori', 'Jumlah Jam']
            fig_pie = px.pie(
                cat_counts,
                values='Jumlah Jam',
                names='Kategori',
                title='Proporsi Kategori Kualitas Udara (US EPA)',
                color_discrete_sequence=px.colors.sequential.Sunsetdark
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_q1_b:
            st.markdown("#### 💡 Insight Bisnis Q1 (Baseline Profile):")
            safe_pct = (df_env['pm25'] <= 35.4).mean() * 100
            unhealthy_pct = (df_env['pm25'] > 55.4).mean() * 100
            st.info(f"""
            - **Rata-rata Konsentrasi PM2.5:** `{df_env['pm25'].mean():.2f} µg/m³`
            - **Proporsi Udara Aman (Baik & Sedang):** `{safe_pct:.1f}%` dari total waktu observasi.
            - **Proporsi Udara Berisiko Kritis (> 55.4 µg/m³):** `{unhealthy_pct:.1f}%`.
            - Kesimpulan: Kualitas udara perkotaan didominasi kategori Sedang dengan lonjakan periodik saat jam sibuk.
            """)

    st.divider()

    # [Q2 & Q3]: Faktor Cuaca & Pola Jam Sibuk
    col_cuaca, col_mobilitas = st.columns(2)
    
    with col_cuaca:
        st.markdown("### 2️⃣ Pengaruh Meteorologi Cuaca (Menjawab Q2)")
        if not df_env.empty:
            fig_scatter = px.scatter(
                df_env,
                x='wind_speed_kmh',
                y='pm25',
                color='humidity_pct',
                trendline="ols",
                title="Efek Dispersi Kecepatan Angin & Kelembaban terhadap PM2.5",
                labels={'wind_speed_kmh': 'Kecepatan Angin (km/h)', 'pm25': 'PM2.5 (µg/m³)', 'humidity_pct': 'Lembab (%)'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.success("""
            **Temuan Q2 (Meteorologi):**
            - Kecepatan angin memiliki **korelasi negatif kuat**: Angin kencang mempercepat dispersi partikulat debu.
            - Kelembaban tinggi menahan polutan di dekat permukaan tanah (inversi kabut).
            """)

    with col_mobilitas:
        st.markdown("### 3️⃣ Pola Mobilitas 24 Jam (Menjawab Q3)")
        if not df_env.empty:
            hourly_avg = df_env.groupby('hour_of_day')[['pm25', 'traffic_index']].mean().reset_index()
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Scatter(x=hourly_avg['hour_of_day'], y=hourly_avg['pm25'], name="PM2.5 (µg/m³)", line=dict(color="#EF4444", width=3)))
            fig_hour.add_trace(go.Scatter(x=hourly_avg['hour_of_day'], y=hourly_avg['traffic_index'], name="Indeks Kemacetan", line=dict(color="#64748B", dash="dot")))
            fig_hour.update_layout(title="Siklus Diurnal 24 Jam: Puncak Polusi di Jam Sibuk", xaxis_title="Jam (0 - 23)", hovermode="x")
            st.plotly_chart(fig_hour, use_container_width=True)
            st.warning("""
            **Temuan Q3 (Mobilitas Antropogenik):**
            - Lonjakan tajam PM2.5 terjadi pada **pukul 07:00–09:00 WIB (Rush Hour Pagi)** dan **17:00–19:00 WIB (Rush Hour Sore)**.
            - Hari kerja (*weekdays*) memiliki rata-rata polusi lebih tinggi dibandingkan akhir pekan (*weekends*).
            """)

    st.divider()

    # [Q4]: Urgensi Mitigasi
    st.markdown("### 4️⃣ Periode Kritis & Rekomendasi Mitigasi (Menjawab Q4)")
    st.markdown("""
    | Tingkat Risiko | Rentang PM2.5 | Rekomendasi Mitigasi Intervensi Publik |
    |---|:---:|---|
    | **🟢 Baik** | $0.0 - 9.0\ \mu g/m^3$ | Bebas beraktivitas di luar ruangan tanpa pembatasan. |
    | **🟡 Sedang** | $9.1 - 35.4\ \mu g/m^3$ | Masyarakat umum aman; kelompok sangat sensitif kurangi aktivitas berat berkepanjangan di luar. |
    | **🟠 Sensitif** | $35.5 - 55.4\ \mu g/m^3$ | **Anak-anak, lansia, dan penderita asma** wajib memakai masker di luar dan menyalakan *air purifier*. |
    | **🔴 Tidak Sehat** | $> 55.4\ \mu g/m^3$ | Tutup jendela ventilasi luar saat jam sibuk; hindari olahraga luar ruangan; gunakan masker standar N95. |
    """)


# ------------------------------------------------------------------------------
# TAB 2: SIMULATOR PREDIKSI INTERAKTIF AI (MENJAWAB Q5)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Simulasi & Inferensi Model Machine Learning (XGBoost Regressor)")
    st.markdown("Ubah parameter kondisi lingkungan dan lalu lintas di bawah ini untuk menguji estimasi prediktif model AI:")
    
    if model is None or scaler is None:
        st.error(" Artefak model (`xgboost_pm25_model.pkl` atau `scaler.pkl`) belum ditemukan di folder `models/`.")
    else:
        with st.form("prediction_form"):
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                st.markdown("**🌤️ Parameter Meteorologi:**")
                temp_in = st.slider("Suhu Udara (°C)", min_value=20.0, max_value=40.0, value=29.5, step=0.5)
                humid_in = st.slider("Kelembaban Relatif (%)", min_value=30.0, max_value=100.0, value=75.0, step=1.0)
                wind_in = st.slider("Kecepatan Angin (km/jam)", min_value=1.0, max_value=40.0, value=10.0, step=0.5)
                rain_in = st.slider("Curah Hujan (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
                
            with col_f2:
                st.markdown("**🚗 Parameter Mobilitas & Waktu:**")
                traffic_in = st.slider("Indeks Kemacetan Lalu Lintas (0 - 100)", min_value=0.0, max_value=100.0, value=75.0, step=5.0)
                hour_in = st.slider("Jam dalam Sehari (WIB)", min_value=0, max_value=23, value=8)
                is_weekend_in = st.selectbox("Status Akhir Pekan:", options=[0, 1], format_func=lambda x: "Akhir Pekan (Sabtu/Minggu)" if x == 1 else "Hari Kerja (Senin - Jumat)")
                is_holiday_in = st.selectbox("Status Hari Libur:", options=[0, 1], format_func=lambda x: "Hari Libur Nasional" if x == 1 else "Hari Biasa")
                
            with col_f3:
                st.markdown("**⏱️ Riwayat Nilai Historis (Lag Fitur):**")
                lag1_in = st.number_input("PM2.5 1 Jam Sebelumnya (µg/m³)", min_value=0.0, max_value=200.0, value=35.0, step=1.0)
                lag24_in = st.number_input("PM2.5 24 Jam Sebelumnya (µg/m³)", min_value=0.0, max_value=200.0, value=38.0, step=1.0)
                roll6_in = st.number_input("PM2.5 Rata-rata 6 Jam Terakhir", min_value=0.0, max_value=200.0, value=34.0, step=1.0)
                wind_dir_in = st.number_input("Arah Angin (Derajat 0-360)", min_value=0.0, max_value=360.0, value=180.0, step=10.0)

            btn_predict = st.form_submit_button("🚀 Hitung Estimasi Kualitas Udara Sekarang", use_container_width=True)
            
        if btn_predict:
            # Transformasi Fitur Siklikal Jam
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
            
            # Penskalaan & Prediksi Model
            input_scaled = scaler.transform(input_df)
            pred_pm25 = float(model.predict(input_scaled)[0])
            pred_pm25 = max(1.0, round(pred_pm25, 2))
            
            # Konversi Standar US EPA AQI
            aqi_res = calculate_pm25_aqi(pred_pm25)
            
            # Tampilan Hasil Prediksi
            st.markdown("---")
            st.markdown("### 🎯 Hasil Estimasi Model AI:")
            
            res_c1, res_c2, res_c3 = st.columns([1.5, 1.5, 2.5])
            with res_c1:
                st.metric("Estimasi Konsentrasi PM2.5", f"{pred_pm25} µg/m³")
            with res_c2:
                st.metric("Skor AQI US EPA", f"{aqi_res['aqi']} / 500")
            with res_c3:
                st.markdown(f"""
                <div style="background:{aqi_res['color']}; padding:0.8rem; border-radius:10px; color:{'#000' if aqi_res['color'] in ['#FFFF00', '#00E400'] else '#FFF'}; font-weight:700;">
                    Kategori: {aqi_res['category']}
                </div>
                <div style="margin-top:0.5rem; font-size:0.9rem; color:#475569;">
                    <b>Rekomendasi Tindakan:</b> {aqi_res['action']}
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
                    st.caption("✅ Hasil prediksi berhasil dicatat secara otomatis ke tabel `prediction_logs` di MySQL.")
                except Exception as log_err:
                    st.caption(f"Catatan log MySQL: {log_err}")

        # Tampilkan Riwayat Log Prediksi Terkini
        st.markdown("---")
        st.subheader("📜 Riwayat Log Prediksi Terakhir (Tabel `prediction_logs`)")
        engine = get_db_connection()
        if engine is not None:
            try:
                logs_df = pd.read_sql("SELECT prediction_id, predicted_at, predicted_pm25, calculated_aqi, aqi_category FROM prediction_logs ORDER BY predicted_at DESC LIMIT 5", engine)
                if not logs_df.empty:
                    st.dataframe(logs_df, use_container_width=True)
                else:
                    st.info("Belum ada catatan log prediksi di database.")
            except Exception:
                st.info("Koneksi tabel log tidak aktif.")


# ------------------------------------------------------------------------------
# TAB 3: EVALUASI AKURASI MODEL & FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Evaluasi Performa Ilmiah Model XGBoost Regressor")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    if metadata and 'metrics' in metadata:
        m = metadata['metrics']
        col_m1.metric("R-Squared Score (R²)", f"{m.get('r2_score', 0.7756):.4f}", "77.6% Variansi Terjelaskan")
        col_m2.metric("Mean Absolute Error (MAE)", f"{m.get('mae', 2.46):.2f} µg/m³", "Deviasi Rata-rata")
        col_m3.metric("Root Mean Squared Error (RMSE)", f"{m.get('rmse', 3.12):.2f} µg/m³", "Galat Kuadratik")
        
    st.markdown("### 🔍 Urutan Kontribusi Fitur (Feature Importance)")
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
            title='Kontribusi Relatif Fitur dalam Estimasi PM2.5 (XGBoost Regressor)',
            color='Importance',
            color_continuous_scale='Bluered'
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
        **Kesimpulan Evaluasi Model (Menjawab Q5):**
        1. Fitur **`pm25_lag_1h`**, **`pm25_rolling_mean_6h`**, dan **`humidity_pct`** memberikan kontribusi terbesar dalam model XGBoost.
        2. Hal ini secara empiris membuktikan bahwa polusi udara memiliki sifat autoregresif temporal kuat serta dipengaruhi secara nyata oleh faktor kelembaban atmosferik dan kemacetan jam sibuk.
        """)
